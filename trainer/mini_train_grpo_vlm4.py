import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from copy import deepcopy

from model.model_vlm import VLMConfig
from dataset.lm_dataset import VLMDataset
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    init_distributed_mode,
    setup_seed,
    init_vlm_model,
    vlm_checkpoint,
    SkipBatchSampler,
    vlm_collate_fn
)

warnings.filterwarnings('ignore')


@torch.no_grad()
def _generate_completions(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    pixel_values,
    num_samples=4,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
):
    """对同一个prompt采样生成 num_samples 个completion，返回(list[Tensor])仅新token部分。"""
    device = input_ids.device
    B, T = input_ids.shape

    input_ids_rep = input_ids.unsqueeze(1).expand(B, num_samples, T).reshape(B * num_samples, T)
    attn_rep = attention_mask.unsqueeze(1).expand(B, num_samples, T).reshape(B * num_samples, T)

    if pixel_values is not None:##===================================
        if isinstance(pixel_values, dict):##===================================
            pv = {k: v.unsqueeze(1).expand(B, num_samples, *v.shape[1:]).reshape(B * num_samples, *v.shape[1:])
                  for k, v in pixel_values.items()}
        else:
            pv = pixel_values.unsqueeze(1).expand(B, num_samples, *pixel_values.shape[1:])##===================================
            pv = pv.reshape(B * num_samples, *pixel_values.shape[1:])##===================================
    else:
        pv = None

    gen_ids = model.generate(
        inputs=input_ids_rep,
        attention_mask=attn_rep,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        top_p=top_p,
        pixel_values=pv,
    )

    completions = []
    for i in range(B * num_samples):
        comp = gen_ids[i, T:]
        if tokenizer.eos_token_id is not None:
            eos_pos = (comp == tokenizer.eos_token_id).nonzero(as_tuple=False)
            if eos_pos.numel() > 0:
                comp = comp[: eos_pos[0].item() + 1]
        completions.append(comp.to(device))
    return completions


def _compute_logprobs(model, input_ids, attention_mask, labels, pad_token_id, pixel_values=None):
    """返回每token logp（与labels对齐的next-token logp），并在mask位置(label=-100)置0。

    关键点：labels 里会有 -100（ignore_index），不能直接作为 gather 的 index。
    需要先把 -100 替换为一个合法 token（如 pad_token_id），再用 mask 把这些位置的 logp 置 0。
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=None, pixel_values=pixel_values)
    logits = out.logits  # [B,T,V]
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    # mask: 仅对有效 label 位置计算 logp
    mask = (shifted_labels != -100)

    # gather 不能吃 -100，先替换为合法 id
    safe_labels = shifted_labels.clone()
    safe_labels = torch.where(mask, safe_labels, torch.full_like(safe_labels, pad_token_id))

    logp = F.log_softmax(shifted_logits, dim=-1)
    token_logp = torch.gather(logp, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)  # [B,T-1]

    token_logp = token_logp * mask.float()
    return token_logp, mask.float()


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8):
    """在 mask==1 的位置做 mean。x/mask shape 相同。"""
    denom = mask.sum(dim=dim).clamp_min(eps)
    return (x * mask).sum(dim=dim) / denom


def _compute_token_logprobs(model, input_ids, attention_mask, pad_token_id, pixel_values=None):
    """计算每个 token 的 next-token logp，返回 shape [B, T-1]，不含 label mask。"""
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=None, pixel_values=pixel_values)
    logits = out.logits[:, :-1, :]  # [B,T-1,V]
    logp = F.log_softmax(logits, dim=-1)
    next_ids = input_ids[:, 1:]  # [B,T-1]
    token_logp = torch.gather(logp, dim=-1, index=next_ids.unsqueeze(-1)).squeeze(-1)
    # padding 位置置 0（可选），mask 由外部传入
    if pad_token_id is not None:
        token_logp = token_logp * (next_ids != pad_token_id).float()
    return token_logp


@torch.no_grad()
def _kl_on_generated_tokens(
    model,
    ref_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_mask: torch.Tensor,
    pad_token_id: int,
    pixel_values: torch.Tensor | None = None,
):
    """DeepSeek风格的近似KL：E_{tokens~pi}[ logp_pi(token) - logp_ref(token) ]，只在生成token(gen_mask==1)上做平均。

    input_ids: [B,T]  完整序列（prompt+completion+pad）
    attention_mask: [B,T]
    gen_mask: [B,T]  仅生成completion位置为1（与labels一致）。

    返回：标量 kl_mean。
    """
    # next-token logp 对齐到 [B, T-1]
    pi_logp = _compute_token_logprobs(model, input_ids, attention_mask, pad_token_id, pixel_values=pixel_values)
    ref_logp = _compute_token_logprobs(ref_model, input_ids, attention_mask, pad_token_id, pixel_values=pixel_values)

    # gen_mask 对齐到 next-token 位置
    m = gen_mask[:, 1:].float()
    kl_tok = (pi_logp - ref_logp) * m
    kl_mean = (kl_tok.sum(dim=1) / m.sum(dim=1).clamp_min(1.0)).mean()
    return kl_mean


def _grpo_advantages(rewards, group_size, eps=1e-8):
    """GRPO: 对每个prompt的group内reward做标准化得到advantage。rewards shape [B*G]."""
    rewards = rewards.view(-1, group_size)
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    adv = (rewards - mean) / (std + eps)
    return adv.view(-1)


def _reward_fn(texts, return_details: bool = False):
    """严格 format 奖励（0/1）：
    必须严格输出：
      <reasoning>非空</reasoning><answer>非空</answer>
    约束：
    - reasoning/answer 标签必须各出现且只出现一次（开/关都为1）
    - 顺序必须 reasoning 在 answer 前
    - reasoning/answer 内容去掉空白后不能为空
    - 禁止在内容区域再次出现 reasoning/answer 任意标签（防 nested / tag spam）
    """
    import re
    import torch

    reasoning_re = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.IGNORECASE | re.DOTALL)
    answer_re = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)

    open_reasoning_re = re.compile(r"<reasoning\b[^>]*>", re.IGNORECASE)
    close_reasoning_re = re.compile(r"</reasoning>", re.IGNORECASE)
    open_answer_re = re.compile(r"<answer\b[^>]*>", re.IGNORECASE)
    close_answer_re = re.compile(r"</answer>", re.IGNORECASE)

    # 任意 reasoning/answer 标签（用于检测内容里是否夹带标签）
    any_reasoning_tag_re = re.compile(r"</?\s*reasoning\b", re.IGNORECASE)
    any_answer_tag_re = re.compile(r"</?\s*answer\b", re.IGNORECASE)

    rs = []
    details = []

    for t in texts:
        t = t or ""

        open_r = len(open_reasoning_re.findall(t))
        close_r = len(close_reasoning_re.findall(t))
        open_a = len(open_answer_re.findall(t))
        close_a = len(close_answer_re.findall(t))

        m_r = reasoning_re.search(t)
        m_a = answer_re.search(t)

        has_r = m_r is not None
        has_a = m_a is not None

        # 必须都存在且只出现一次（开关都为1）+ regex能匹配出内容
        single_pair_ok = (open_r == 1 and close_r == 1 and open_a == 1 and close_a == 1 and has_r and has_a)

        # 顺序：reasoning 在 answer 之前
        order_ok = False
        if has_r and has_a:
            order_ok = m_r.start() < m_a.start()

        reasoning_txt = (m_r.group(1) if has_r else "")
        answer_txt = (m_a.group(1) if has_a else "")

        # 内容非空（去掉空白）
        reasoning_non_empty = len(reasoning_txt.strip()) > 0
        answer_non_empty = len(answer_txt.strip()) > 0
        content_ok = reasoning_non_empty and answer_non_empty

        # 禁止嵌套/重复标签出现在内容里
        nested_ok = True
        if has_r and (any_reasoning_tag_re.search(reasoning_txt) or any_answer_tag_re.search(reasoning_txt)):
            nested_ok = False
        if has_a and (any_reasoning_tag_re.search(answer_txt) or any_answer_tag_re.search(answer_txt)):
            nested_ok = False

        well_formed = bool(single_pair_ok and order_ok and content_ok and nested_ok)
        r = 1.0 if well_formed else 0.0

        rs.append(r)
        details.append({
            "reward": r,
            "has_reasoning": has_r,
            "has_answer": has_a,
            "open_reasoning": open_r,
            "close_reasoning": close_r,
            "open_answer": open_a,
            "close_answer": close_a,
            "single_pair_ok": single_pair_ok,
            "order_ok": order_ok,
            "reasoning_non_empty": reasoning_non_empty,
            "answer_non_empty": answer_non_empty,
            "nested_ok": nested_ok,
            "text_len": len(t),
        })

    rewards = torch.tensor(rs, dtype=torch.float32)
    if return_details:
        return rewards, details
    return rewards


def _infer_prompt_lens(input_ids, labels, pad_id):
    """逐样本推断prompt长度（assistant答案开始位置）。

    VLMDataset 的 labels 只在 assistant 段为 token id，其余为 -100。
    因此 prompt_len = 首个非 -100 的位置。
    """
    lens = []
    for b in range(labels.size(0)):
        pos = (labels[b] != -100).nonzero(as_tuple=False)
        if pos.numel() == 0:
            valid = (input_ids[b] != pad_id).nonzero(as_tuple=False)
            lens.append(int(valid[-1].item() + 1) if valid.numel() else int(input_ids.size(1)))
        else:
            lens.append(int(pos[0].item()))
    return lens


def _decode_ground_truth_from_labels(tokenizer, input_ids: torch.Tensor, labels: torch.Tensor) -> list[str]:
    """从 labels!=-100 的区域还原每条样本的 ground truth（assistant 段）。

    返回 list[str]，长度为 batch_size。

    注意：VLMDataset 的 labels 在 assistant 段为 token id，其余为 -100。
    这里使用 input_ids 对齐，抽取 labels 有效位置对应的 token 序列。
    """
    gts: list[str] = []
    bsz = labels.size(0)
    for b in range(bsz):
        pos = (labels[b] != -100).nonzero(as_tuple=False).view(-1)
        if pos.numel() == 0:
            gts.append("")
            continue
        # 直接从 input_ids 抽取对应 tokens（与 labels 在该位置一致）
        ids = input_ids[b, pos].detach().cpu().tolist()
        gts.append(tokenizer.decode(ids, skip_special_tokens=False))
    return gts


def _extract_answer_text(text: str) -> str:
    """从输出中抽取 <answer>...</answer> 内的内容；若不存在则返回空串。"""
    import re

    if not text:
        return ""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    return (m.group(1) or "").strip()


def _reward_accuracy(
    gen_texts: list[str],
    ground_truth_texts: list[str],
    group_size: int,
    return_details: bool = False,
):
    """准确性奖励：使用外部 LLM 裁判对 reasoning/answer 打分（0~1）。

    约定（重要）：
    - gen_texts 与 ground_truth_texts 必须一一对应，长度相同（都为 N）。
      这样 compute_reward 在做 gating/subset 时不会破坏对齐关系。

    裁判 prompt：

    标准答案:
    {ground_truth}
    当前模型回答:
    {texts}

    请根据标准答案，为我的reasoning和answer部分打分，按如下格式回答
    {"reasoning":...,"answer":....}分数0到1

    返回：torch.FloatTensor shape [N]
    """
    import json
    import re
    import time
    import torch

    try:
        from scripts.api import call_llm  # type: ignore
    except Exception as e:
        rewards = torch.zeros(len(gen_texts), dtype=torch.float32)
        if return_details:
            return rewards, [{"reward": 0.0, "error": f"import call_llm failed: {e}"} for _ in gen_texts]
        return rewards

    N = len(gen_texts)
    if len(ground_truth_texts) != N:
        raise ValueError(f"ground_truth_texts 长度应与 gen_texts 相同: {len(ground_truth_texts)} vs {N}")

    if not hasattr(_reward_accuracy, "_judge_cache"):
        _reward_accuracy._judge_cache = {}  # type: ignore[attr-defined]

    def _build_prompt(gt: str, pred_full: str) -> str:
        return (
            "标准答案:\n"
            f"{gt}\n"
            "当前模型回答:\n"
            f"{pred_full}\n\n"
            "请根据标准答案，为我的reasoning和answer部分打分，按如下格式回答\n"
            '{"reasoning":...,"answer":....}分数0到1'
        )

    def _parse_scores(resp: str) -> tuple[float, float, str | None]:
        if resp is None:
            return 0.0, 0.0, "empty_response"
        s = str(resp).strip()
        if not s:
            return 0.0, 0.0, "empty_response"

        payload = None
        try:
            payload = json.loads(s)
        except Exception:
            m = re.search(r"\{[\s\S]*?\}", s)
            if m:
                try:
                    payload = json.loads(m.group(0))
                except Exception:
                    payload = None

        if not isinstance(payload, dict):
            return 0.0, 0.0, "json_parse_failed"

        r = payload.get("reasoning", 0.0)
        a = payload.get("answer", 0.0)
        try:
            r = float(r)
        except Exception:
            r = 0.0
        try:
            a = float(a)
        except Exception:
            a = 0.0

        r = max(0.0, min(1.0, r))
        a = max(0.0, min(1.0, a))
        return r, a, None

    gt_texts = [(gt or "").strip() for gt in ground_truth_texts]

    rs: list[float] = []
    details: list[dict] = []

    min_interval_sec = 0.0
    last_call_ts = 0.0

    cache = _reward_accuracy._judge_cache  # type: ignore[attr-defined]

    for i, pred_full in enumerate(gen_texts):
        gt = gt_texts[i]
        pred_full = (pred_full or "").strip()

        if not gt or not pred_full:
            r_tot = 0.0
            rs.append(r_tot)
            if return_details:
                details.append({"reward": r_tot, "reasoning": 0.0, "answer": 0.0, "error": "empty_gt_or_pred"})
            continue

        cache_key = (gt, pred_full)
        if cache_key in cache:
            r_score, a_score, raw = cache[cache_key]
            err = None
        else:
            if min_interval_sec > 0:
                now = time.time()
                dt = now - last_call_ts
                if dt < min_interval_sec:
                    time.sleep(min_interval_sec - dt)
            prompt = _build_prompt(gt, pred_full)
            try:
                raw = call_llm(prompt)
                last_call_ts = time.time()
                r_score, a_score, err = _parse_scores(raw)
            except Exception as e:
                raw = f"call_llm_failed: {e}"
                r_score, a_score, err = 0.0, 0.0, "call_llm_failed"
            cache[cache_key] = (r_score, a_score, raw)

        # 合成：answer 更重要（可按需调整）
        r_tot = 1.0 * float(r_score) + 1.0 * float(a_score)
        rs.append(r_tot)

        if return_details:
            details.append({
                "reward": r_tot,
                "reasoning": float(r_score),
                "answer": float(a_score),
                "error": err,
                "judge_raw": (raw or "")[:500],
            })

    rewards = torch.tensor(rs, dtype=torch.float32)
    if return_details:
        return rewards, details
    return rewards


def compute_reward(
    tokenizer,
    gen_texts: list[str],
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    group_size: int,
    w_format: float = 1.0,
    w_acc: float = 1.0,
    return_details: bool = False,
):
    """统一 reward 入口：格式奖励 + 准确性奖励。

    返回：
    - total_rewards: torch.FloatTensor shape [B*G]
    - details: dict（可选）
    """
    fmt_r, fmt_details = _reward_fn(gen_texts, return_details=True)

    import torch
    acc_r = torch.zeros_like(fmt_r)
    acc_details: list[dict] = [{"reward": 0.0, "skipped": True, "reason": "format_reward_zero"} for _ in gen_texts]

    if w_acc != 0:
        idx = (fmt_r > 0).nonzero(as_tuple=False).view(-1)
        if idx.numel() > 0:
            gt_texts = _decode_ground_truth_from_labels(tokenizer, input_ids, labels)

            # 把 GT 从 [B] 展开到 [B*G]，与 gen_texts/completions 的顺序一致：i // G 对应 b
            gt_rep: list[str] = []
            B = len(gt_texts)
            for b in range(B):
                gt_rep.extend([gt_texts[b]] * group_size)

            sub_idx = idx.detach().cpu().tolist()
            sub_gen = [gen_texts[i] for i in sub_idx]
            sub_gt = [gt_rep[i] for i in sub_idx]

            sub_acc_r, sub_acc_details = _reward_accuracy(
                sub_gen,
                sub_gt,
                group_size=group_size,
                return_details=True,
            )
            acc_r[idx] = sub_acc_r

            if return_details:
                for j, i in enumerate(sub_idx):
                    acc_details[i] = sub_acc_details[j] | {"skipped": False}

    total = w_format * fmt_r + w_acc * acc_r

    if return_details:
        gt_texts = _decode_ground_truth_from_labels(tokenizer, input_ids, labels)
        return total, {
            "format": fmt_details,
            "accuracy": acc_details,
            "ground_truth_texts": gt_texts,
            "w_format": w_format,
            "w_acc": w_acc,
        }
    return total


def _grpo_token_loss(
    model,
    ref_model,
    full_ids: torch.Tensor,
    full_attn: torch.Tensor,
    gen_mask: torch.Tensor,
    advantages: torch.Tensor,
    pad_token_id: int,
    beta: float,
    pixel_values: torch.Tensor | None = None,
):
    """token-level GRPO loss + 非负KL（与文本版训练脚本一致，稳定性更好）。

    - full_ids/full_attn: [V, T]（prompt+completion+pad）
    - gen_mask: [V, T]，仅completion区域为1
    - advantages: [V]

    返回：(loss, kl_mean)
    - loss: 标量
    - kl_mean: 标量（completion token上的平均 per-token KL）

    说明：
    - ratio trick: exp(logp - stopgrad(logp))，数值恒为1但梯度正确
    - KL: kl_div = ref_logp - pi_logp; per_token_kl = exp(kl_div) - kl_div - 1 (>=0)
    """
    # next-token 对齐到 [V, T-1]
    pi_logp = _compute_token_logprobs(model, full_ids, full_attn, pad_token_id, pixel_values=pixel_values)
    with torch.no_grad():
        ref_logp = _compute_token_logprobs(ref_model, full_ids, full_attn, pad_token_id, pixel_values=pixel_values)

    m = gen_mask[:, 1:].float()  # [V,T-1]
    denom = m.sum(dim=1).clamp_min(1.0)  # [V]

    ratio = torch.exp(pi_logp - pi_logp.detach())  # [V,T-1]

    kl_div = (ref_logp - pi_logp)  # [V,T-1]
    per_token_kl = torch.exp(kl_div) - kl_div - 1.0  # >=0

    per_token_loss = -(ratio * advantages.unsqueeze(1) - beta * per_token_kl)
    loss_per_seq = (per_token_loss * m).sum(dim=1) / denom
    loss = loss_per_seq.mean()

    kl_mean = ((per_token_kl * m).sum(dim=1) / denom).mean()
    return loss, kl_mean


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    model.train()

    for step, (input_ids, labels, pixel_values) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # pixel_values = pixel_values.to(args.device)
        pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()} if isinstance(pixel_values, dict) else pixel_values.to(args.device)

        attn_mask = (input_ids != tokenizer.pad_token_id).long()

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # prompt裁剪：逐样本只用prompt部分做生成
        with torch.no_grad():
            prompt_lens = _infer_prompt_lens(input_ids, labels, tokenizer.pad_token_id)
            prompt_len_max = max(prompt_lens)

            prompt_ids = torch.full((input_ids.size(0), prompt_len_max), tokenizer.pad_token_id, dtype=torch.long, device=args.device)
            prompt_attn = torch.zeros_like(prompt_ids)
            for b, pl in enumerate(prompt_lens):
                prompt_ids[b, :pl] = input_ids[b, :pl]
                prompt_attn[b, :pl] = 1

        # 1) 采样生成
        with torch.no_grad():
            with autocast_ctx:
                completions = _generate_completions(
                    model,
                    tokenizer,
                    prompt_ids,
                    prompt_attn,
                    pixel_values,
                    num_samples=args.num_samples,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

        gen_texts = [tokenizer.decode(c, skip_special_tokens=False) for c in completions]

        # --- 每一步输出/保存生成内容（主进程） ---
        if is_main_process() and getattr(args, "print_generations", False):
            try:
                interval = int(getattr(args, "print_gen_interval", 1) or 1)
                if interval <= 0:
                    interval = 1
                if (step % interval) == 0 or step == iters - 1:
                    max_show = int(getattr(args, "print_gen_max", args.num_samples) or args.num_samples)
                    trunc = int(getattr(args, "print_gen_trunc", 0) or 0)

                    Logger(f"[gen] epoch={epoch+1} step={step} showing={min(max_show, len(gen_texts))}/{len(gen_texts)}")
                    for i, t in enumerate(gen_texts[:max_show]):
                        t_show = (t[:trunc] + "...<trunc>") if (trunc and len(t) > trunc) else t
                        Logger(f"[gen] #{i}: {t_show}")

                    save_path = getattr(args, "print_gen_save_path", "")
                    if save_path:
                        import json
                        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                        rec = {
                            "epoch": int(epoch + 1),
                            "step": int(step),
                            "num_samples": int(args.num_samples),
                            "gen_texts": gen_texts,
                        }
                        with open(save_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                try:
                    Logger(f"[gen] logging failed: {e}")
                except Exception:
                    pass
        # -----------------------------------------

        rewards, reward_bundle = compute_reward(
            tokenizer=tokenizer,
            gen_texts=gen_texts,
            input_ids=input_ids,
            labels=labels,
            group_size=args.num_samples,
            w_format=1.0,
            w_acc=1.0,
            return_details=True,
        )
        reward_details = reward_bundle.get("format", [])

        rewards = rewards.to(args.device)
        advantages = _grpo_advantages(rewards, args.num_samples).to(args.device)
        # 防止adv过大导致更新过猛（文本版也会做clamp/标准化）
        advantages = torch.clamp(advantages, -10.0, 10.0)

        # 若一个 prompt 的组内 reward 全相同(全0或全1)，GRPO 标准化后 advantage 会全0，loss 也会变成0。
        # 这种 step 反传没有意义，直接跳过可节省算力并避免日志出现 -0.000e+00。
        if advantages.abs().sum().item() == 0.0:
            if is_main_process() and (step % args.log_interval == 0):
                try:
                    Logger(f"[skip] step={step} advantages are all zero. rewards={rewards.detach().float().cpu().tolist()}")
                except Exception:
                    pass

            # 关键：即便 skip 也允许触发保存（尤其是 epoch 最后一步）
            if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
                model.eval()
                moe_suffix = '_moe' if vlm_config.use_moe else ''
                ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
                raw_model = model.module if isinstance(model, DistributedDataParallel) else model
                raw_model = getattr(raw_model, '_orig_mod', raw_model)
                state_dict = raw_model.state_dict()
                clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vision_encoder.')}
                clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
                torch.save(clean_state_dict, ckp)
                vlm_checkpoint(
                    vlm_config,
                    weight=args.save_weight,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=step,
                    wandb=wandb,
                    save_dir='../checkpoints',
                    scaler=scaler,
                )
                model.train()
                del state_dict, clean_state_dict

            continue                    

        # 2) 组装序列并计算policy gradient loss（按每个样本自身prompt_len对齐）
        B = prompt_ids.size(0)
        G = args.num_samples
        V = B * G
        max_comp = max([c.numel() for c in completions]) if completions else 1

        full_ids = torch.full((V, prompt_len_max + max_comp), tokenizer.pad_token_id, dtype=torch.long, device=args.device)
        full_attn = torch.zeros_like(full_ids)
        full_labels = torch.full_like(full_ids, -100)

        for i in range(V):
            b = i // G
            pl = prompt_lens[b]
            comp = completions[i]

            full_ids[i, :pl] = input_ids[b, :pl]
            full_ids[i, pl:pl + comp.numel()] = comp

            full_attn[i, :pl + comp.numel()] = 1
            full_labels[i, pl:pl + comp.numel()] = full_ids[i, pl:pl + comp.numel()]


        if pixel_values is not None:##===================================
            if isinstance(pixel_values, dict):##===================================
                pv_rep = {k: v.repeat_interleave(G, dim=0) for k, v in pixel_values.items()}
            else:
                pv_rep = pixel_values.repeat_interleave(G, dim=0)##===================================
        else:
            pv_rep = None


        with autocast_ctx:
            # completion token mask（与labels一致）
            gen_mask = (full_labels != -100).to(full_attn.dtype)

            beta = float(args.kl_beta) if (args.kl_beta and args.kl_beta > 0) else 0.0
            pg_loss, kl_mean = _grpo_token_loss(
                model=model,
                ref_model=ref_model,
                full_ids=full_ids,
                full_attn=full_attn,
                gen_mask=gen_mask,
                advantages=advantages,
                pad_token_id=tokenizer.pad_token_id,
                beta=beta,
                pixel_values=pv_rep,
            )
            # 这里 pg_loss 已经包含 KL 项；为了保持原日志字段，保留 kl_loss 便于展示
            kl_loss = torch.tensor(beta, device=args.device) * kl_mean

            loss = pg_loss
            loss = loss / args.accumulation_steps

        # ---- debug (只在前几个 step、主进程打印一次) ----
        if is_main_process():
            try:
                Logger(f"[debug] step={step} prompt_lens={prompt_lens} prompt_len_max={prompt_len_max} max_comp={max_comp}")
                Logger(f"[debug] rewards={rewards.detach().float().cpu().tolist()} reward_mean={rewards.mean().item():.4f} reward_std={rewards.std().item():.4f}")
                Logger(f"[debug] advantages mean={advantages.mean().item():.6f} min={advantages.min().item():.6f} max={advantages.max().item():.6f}")
            except Exception as e:
                Logger(f"[debug] failed: {e}")
        # ----------------------------------------------

        # ---- debug / log ----
        if step % args.log_interval == 0 or step == iters - 1:
            if wandb:
                wandb.log({
                    'kl_mean': kl_mean.item(),
                    'kl_loss': kl_loss.item(),
                    'pg_loss': pg_loss.item(),
                })

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {loss.item() * args.accumulation_steps:.3e}, '
                f'reward: {rewards.mean().item():.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min'
            )
            if wandb:
                wandb.log({
                    'loss': loss.item() * args.accumulation_steps,
                    'reward_mean': rewards.mean().item(),
                    'reward_std': rewards.std().item(),
                    'learning_rate': current_lr,
                    'epoch_time': eta_min,
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if vlm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vision_encoder.')}
            clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
            torch.save(clean_state_dict, ckp)
            vlm_checkpoint(
                vlm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir='../checkpoints',
                scaler=scaler,
            )
            model.train()
            del state_dict, clean_state_dict

        del input_ids, labels, pixel_values, full_ids, full_attn, full_labels, loss

# #训练
# 本项目训练流程分为两步：SFT 冷启动 → GRPO 强化学习。
# 1) SFT 冷启动（Supervised Fine-Tuning）
# 使用监督数据对模型进行冷启动，得到后续 GRPO 的初始化权重：
# python train_sft_vlm.py \
#   --epochs 1 \
#   --from_weight sft_vlm \
#   --save_weight reasoning_sft1 \
#   --data_path dataset/train_data/vlm_reasoning_3149_train.parquet \
#   --save_dir ../out_vlm_reasoning \
#   --use_wandb \
#   --wandb_project vlm_reasoning
#
# 2) GRPO 训练（Reinforcement Learning）
# 在 SFT 权重基础上进行 GRPO 强化学习训练：
# python train_grpo_vlm.py \
#   --save_dir ../out_vlm_reasoning \
#   --save_weight reasoning_grpo \
#   --epochs 1 \
#   --batch_size 1 \
#   --learning_rate 5e-6 \
#   --data_path dataset/train_data/vlm_reasoning_2149_grpo.parquet \
#   --from_weight reasoning_sft1 \
#   --num_samples 4 \
#   --temperature 0.65 \
#   --top_p 0.85 \
#   --accumulation_steps 1 \
#   --log_interval 20 \
#   --save_interval 1000 \
#   --use_wandb \
#   --wandb_project vlm_reasoning_GRPO_v1
#
# 3) 直接下载已训练模型（可选）
# 如果不想本地训练，可以直接下载已训练好的 SFT / GRPO 权重：
# https://huggingface.co/JinshengWei/Minimind-v-RL/tree/main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V VLM GRPO")
    parser.add_argument("--save_dir", type=str, default="../outown", help="模型保存目录")##===================================
    parser.add_argument('--save_weight', default='cot_grpo', type=str, help="保存权重的前缀名")##===================================


    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")##===================================
    parser.add_argument("--data_path", type=str, default="../dataset/cot_reasoning[JinshengWei]__Minimind-v-RL-thinking/vlm_reasoning_2149_grpo.parquet", help="训练数据路径")##===================================
    parser.add_argument('--from_weight', default='cot_sft', type=str, help="基于哪个权重训练")##===================================##===================================
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")##===================================##===================================
    ##默认 --freeze_llm 1即训练vision_proj 和LLM 首尾层，保留中间层原有语言能力：
    parser.add_argument('--freeze_llm', default=0, type=int, choices=[0, 1, 2], help="冻结策略（0=完全可训练，   1=冻结proj+解冻首尾层，   2=完全冻结仅训练proj）")  ##Visual Encoder一直完全冻结##===================================##===================================
    # ##参数说明：
    # --save_dir: 保存权重的目录
    # --save_weight: 保存权重的前缀名
    #
    # --from_weight: 基础权重名称（none  llm----, pretrain_vlm, sft_vlm等）
    # --from_resume: 是否续训（0=从头开始，1=从检查点继续）
    # --freeze_llm: 冻结策略（0=全参可训，1=proj + LLM 首尾层，2=仅训 proj）。----Pretrain默认2----SFT默认1----grpo默认0
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")##===================================
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    # parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度")
    parser.add_argument('--max_seq_len', default=1536, type=int, help="训练的最大截断长度")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    ##
    parser.add_argument('--num_samples', type=int, default=4, help='每个prompt采样数')
    parser.add_argument('--max_new_tokens', type=int, default=1536)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--kl_beta', type=float, default=0.02, help='KL正则系数(beta)，0表示不使用')
    # ---- generation printing/saving ----
    parser.add_argument('--print_generations', type=int, default=1, choices=[0, 1], help='是否输出每一步生成的内容（仅主进程）')
    parser.add_argument('--print_gen_interval', type=int, default=1, help='输出生成内容的步频（默认每步）')
    parser.add_argument('--print_gen_max', type=int, default=4, help='每步最多输出多少条生成（从gen_texts前面截取）')
    parser.add_argument('--print_gen_trunc', type=int, default=5000, help='单条生成输出截断长度，<=0 表示不截断')
    parser.add_argument('--print_gen_save_path', type=str, default='temp.jsonl', help='可选：把每步生成写入 jsonl 文件路径（包含完整 gen_texts）')
    # -----------------------------------

    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="vlm_reasoning_GRPO", help="wandb项目名")
    args = parser.parse_args()
    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    vlm_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, max_seq_len=args.max_seq_len, use_moe=bool(args.use_moe))##===================================
    ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None##===================================##===================================

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        run_name = f"MiniMind-V-GRPO-E{args.epochs}-BS{args.batch_size}-LR{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=run_name, id=wandb_id, resume=resume)

    # 注意：init_vlm_model 默认从 ../out 加载权重；这里将 save_dir 显式传入，保持与你的命令一致
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer, preprocess = init_vlm_model(vlm_config, from_weight=args.from_weight, device=args.device, freeze_llm=args.freeze_llm)#1=冻结proj+解冻首尾层##===================================##===================================
    ##
    ##
    # ---- reference model for KL (frozen) ----
    ref_model = deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # 重要：ref_model 不 compile，避免额外开销/兼容性问题；同时放到同一device
    ref_model.to(args.device)




    ##<|image_pad|>*64
    train_ds = VLMDataset(args.data_path, tokenizer, preprocess=preprocess, image_special_token=vlm_config.image_special_token, image_token_len=vlm_config.image_token_len, max_length=vlm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None




    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=vlm_collate_fn)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()


# #Model Params: 65.09M
# Trainable Params: 65.095M
# [gen] epoch=1 step=1 showing=4/16 ##===================================##===================================
# [gen] #0: 是公园里的“游客”或“游客”类型，通常在游客前或游客区进行户外活动。公园里有
# [gen] #1: 上方的公园场景是户外野餐。以下是详细描述：
# [gen] #2: 为访客设计的户外运动器材是野餐或集会式的。以下是详细描述：
# [gen] #3: 以自然、有机形式为主的户外运动包括：
# #[debug] step=1 prompt_lens=[89, 89, 100, 101] prompt_len_max=101 max_comp=1536
# # [debug] rewards=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0] reward_mean=0.1875 reward_std=0.4031
# # [debug] advantages mean=0.000000 min=-1.500000 max=0.500000
#
#
# [gen] epoch=1 step=2 showing=4/16  ##===================================##===================================
# [gen] #0: 拍摄照片时，人们常会因视觉干扰、遮挡视线或忽略主体。图中两位
# [gen] #1: ------ 这张图片显示室内环境，有玻璃门和金属门，门左侧有一面
# [gen] #2: 故意留下“插图”一词，但实际情况可能因人而异。图中人物
# [gen] #3: 客：一名穿着深色西装、脏腰的男士，正在打开一台冰箱
#
#
# [gen] epoch=1 step=3 showing=4/16  ##===================================##===================================
# [gen] #0: <reasoning>
# 1. 观察图像：图像显示一个女性站在室内环境中，身着浅色长袍，头发短。她穿着白色长袖上衣，胸前有装饰元素（如蕾丝边饰），表情放松。
# 2. 分析环境：背景中有观众和摄像机，环境看起来像是一个公共活动场所，有人在交谈或表演。
# 3. 推断主要要素：
#    - 头发和肩膀**：女性头发短，造型时尚，面部表情专注。
#    - 背景中有模糊的人物，可能与一场现场表演有关（如音乐剧、电影、电视剧），但无法辨认。
#    - 人群模糊：背景有人在交谈或表演，整体氛围随意、非正式。
# 4. 综合判断：该图片属于“公共活动场所”或“广场/社区活动场所”，属于公共空间环境。
# </reasoning>##===================================
# <answer>##===================================
# 这是一张公共活动场所，人群聚集在场所，有人在交谈或表演，背景有观众和摄像机，整体氛围轻松。
# </answer><|im_end|>
#
#
# [gen] #1: <reasoning>
# 1. 观察图片：这是一个室内场景，有人站在街道旁，背景有模糊的人影和其他模糊的人影，可能是在街角或人群区。
# 2. 分析场景：一名女性正面向镜头微笑，表情中立，似乎在交谈或拍照；一名女性手持手机，面带微笑，姿态放松。
# 3. 推断场景：画面未显示人物在活动中，但部分人影和周围环境为非人造场景，可能暗示非机械场景。
# 4. 推断潜在用途：此类场景常见于社交活动、节日或公共聚会，或作为艺术装置、艺术装置等。
# 5. 综合判断：这是一张带有“人像照片”和“场景画面”元素的室内摄影作品，常见于公共场所、展览或拍摄广播。
# </reasoning>##===================================
# <answer>##===================================
# 这是一张带有人像照片和场景元素的室内摄影作品，可能用于公共活动或艺术展示。
# </answer><|im_end|>
#
#
# [gen] #2: <think>
# </think>
# <reasoning>
# 1. 图像显示一个女性在室内进行活动，背景有人和行走的人，环境看起来像是非正式或非正式的场合。
# 2. 前景中有两位女性，一位穿着休闲长袖上衣，另一位穿着浅色上衣，面带微笑，似乎在交谈或参与某种互动活动。背景中有其他人在场，整体氛围轻松愉快。
# 3. 综合判断：她们似乎在参与一场非正式的交谈或社交活动，背景有人在交谈，环境轻松但非正式，整体氛围轻松愉快。
# 4. 综合判断：这张照片很可能用于公共活动、社交聚会或休闲聚会，氛围轻松愉快。
# </reasoning>##===================================
# <answer>##===================================
# 这是一张非正式社交场景，背景有人在场，环境轻松，氛围轻松愉快。
# </answer><|im_end|>
#
#
# [gen] #3: <reasoning>
# 1. 图像显示一名女性站在室内场景中，背景有其他人和一只手机屏幕，部分被遮挡。
# 2. 她面前有两名女性，背景是模糊的人影，表情略显紧张或不悦。
# 3. 她穿着休闲，双手叉腰，姿态放松。
# 4. 背景有人影和一扇窗户，环境昏暗，光线较暗。
# 5. 她们身后有模糊的人影，部分人影靠近前景。
# 6. 前景中有另一人，但内容不清晰；背景中还有另一人，但模糊。
# 7. 整体氛围轻松愉快，可能是在社交场合或工作场合。
# </reasoning>##===================================
# <answer>##===================================
# 图像显示一名女性在室内场景中，背景有人影和另一人影。
# </answer><|im_end|>
#
#
# [gen] epoch=1 step=4 showing=4/16 ##===================================##===================================
# [gen] #0: CanCanCanCan Can Can Can Can Account incident, as indicated by the ima
# [gen] #1: 用品（如花瓶）的绿色花瓶可能象征成长、成长或美化生活。粉色花瓶可能象征
# [gen] #2: 为花卉设计的花瓶中粉色花朵通常具有象征意义。粉色花朵象征爱与纯洁，常与纯洁、纯洁或成长有关。它也可能作为装饰品或装饰品，增添视觉趣味和情感表达。结合花朵的自然象征意义，可推断它可能代表自然、生命、成长或情感的延续。然而，仅凭图像无法确定具体意义，因此需结合视觉构图、象征意义等多维度分析。
# <answer>##===================================
# 粉色花朵可能象征纯洁、爱与启迪，也可能象征成长与希望。结合花朵的自然象征意义，可推断其重要性在于自然循环与生命循环，或是象征生命短暂或生命短暂时刻的延续。
# </answer><|im_end|>
#
# [gen] #3:
# <reasoning>
# 1. 图像中显示一朵粉色花瓶，花瓶里插着一朵粉色花朵（可能是玫瑰或类似花朵），周围有白色花朵。
# 2. 花瓶整体颜色较浅，可能是粉色或粉色，而白色花朵更显粉红和白色。
# 3. 背景是朦胧的自然光，可能为窗户或窗帘，增强了自然光线。
# 4. 可见花瓶内有花朵或小花瓶，但若未完全展开，可能为花瓶内的花朵。
# 5. 整体环境看起来像窗台或窗帘，但没有明显装饰或装饰，无法推断其具体意义。
# </reasoning>##===================================
# <answer>##===================================
# 花瓶中的粉色花朵可能象征纯洁、宁静或情感联想。
# </answer><|im_end|>
#
#
# [debug] step=4 prompt_lens=[91, 102, 97, 81] prompt_len_max=102 max_comp=1536
# [debug] rewards=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] reward_mean=0.3125 reward_std=0.4787
# [debug] advantages mean=0.000000 min=-0.500000 max=1.500000
# [gen] epoch=1 step=5 showing=4/16 ##===================================##===================================##===================================



