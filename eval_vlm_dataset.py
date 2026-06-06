# import os
# import io
# import json
# import time
# import argparse
# import warnings
# from typing import Any, Dict, List, Optional, Tuple
#
# import torch
# import pyarrow.parquet as pq
# from PIL import Image
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# # tqdm（可选依赖：未安装时自动降级）
# try:
#     from tqdm import tqdm
# except Exception:  # pragma: no cover
#     tqdm = None
#
# from model.model_vlm import MiniMindVLM, VLMConfig
# from trainer.trainer_utils import setup_seed, get_model_params
#
# warnings.filterwarnings('ignore')
#
#
# def init_model(args):
#     tokenizer = AutoTokenizer.from_pretrained(args.load_from)
#     if 'model' in args.load_from:
#         moe_suffix = '_moe' if args.use_moe else ''
#         ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
#         model = MiniMindVLM(
#             VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)),
#             vision_model_path=args.vision_model_path
#         )
#         state_dict = torch.load(ckp, map_location=args.device)
#         model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
#     else:
#         model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
#         model.vision_encoder, model.processor = MiniMindVLM.get_vision_model(args.vision_model_path)
#
#     # 兼容：某些 transformers 模型没有 params 字段
#     if not hasattr(model, 'params'):
#         model.params = getattr(model, 'config', None)
#     get_model_params(model, model.params)
#     preprocess = model.processor
#     return model.eval().to(args.device), tokenizer, preprocess
#
#
# def _load_image_from_bytes(image_bytes: Any) -> List[Image.Image]:
#     """兼容 parquet 中 image_bytes 为 bytes 或 list[bytes] 的情况。"""
#     if image_bytes is None:
#         return []
#     if isinstance(image_bytes, list):
#         out = []
#         for b in image_bytes:
#             if b is None:
#                 continue
#             out.append(Image.open(io.BytesIO(b)).convert('RGB'))
#         return out
#     return [Image.open(io.BytesIO(image_bytes)).convert('RGB')]
#
#
# def _safe_json_loads(x: Any) -> Any:
#     if x is None:
#         return None
#     if isinstance(x, (dict, list)):
#         return x
#     if not isinstance(x, str):
#         x = str(x)
#     try:
#         return json.loads(x)
#     except Exception:
#         return None
#
#
# def _extract_eval_fields(row: Dict[str, Any], prompt_col: str, gt_col: str) -> Tuple[str, Optional[str], Optional[List[Dict[str, Any]]]]:
#     """从一行中取 prompt / gt / conversations(可选)。"""
#     conversations = None
#     if 'conversations' in row:
#         conversations = _safe_json_loads(row.get('conversations'))
#
#     prompt = row.get(prompt_col)
#     gt = row.get(gt_col) if gt_col else None
#
#     # 若指定列不存在/为空：尝试从 conversations 中抽取
#     if (prompt is None or str(prompt).strip() == '') and conversations:
#         # 常见格式：[{role:'user', content:'...'}, {role:'assistant', content:'...'}]
#         user_msgs = [m for m in conversations if (m.get('role') == 'user' or m.get('from') == 'human')]
#         if user_msgs:
#             prompt = user_msgs[-1].get('content') or user_msgs[-1].get('value')
#
#     if (gt is None or (isinstance(gt, str) and gt.strip() == '')) and gt_col is None and conversations:
#         asst_msgs = [m for m in conversations if (m.get('role') == 'assistant' or m.get('from') == 'gpt')]
#         if asst_msgs:
#             gt = asst_msgs[-1].get('content') or asst_msgs[-1].get('value')
#
#     if prompt is None:
#         prompt = ''
#     if not isinstance(prompt, str):
#         prompt = str(prompt)
#     if gt is not None and not isinstance(gt, str):
#         gt = str(gt)
#
#     return prompt, gt, conversations
#
#
# def _build_inputs(tokenizer, model, prompt: str) -> str:
#     # 支持 prompt 中包含 <image> 占位符
#     if hasattr(model, 'params') and getattr(model.params, 'image_special_token', None):
#         prompt = prompt.replace('<image>', model.params.image_special_token)
#
#     messages = [{"role": "user", "content": prompt}]
#     try:
#         return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     except Exception:
#         # 兼容不支持 chat_template 的 tokenizer
#         return prompt
#
#
# def _simple_score(pred: str, gt: Optional[str]) -> Dict[str, Any]:
#     """一个可落地的轻量打分：1) 若无 gt，则 score=None；2) 有 gt 计算字符级 F1/BLEU-like(简化)。"""
#     if gt is None:
#         return {"score": None, "metric": "none"}
#
#     pred = (pred or '').strip()
#     gt = (gt or '').strip()
#     if not gt:
#         return {"score": None, "metric": "none"}
#
#     # 字符级 F1（对中文/无空格文本也相对稳）
#     from collections import Counter
#
#     p = list(pred)
#     g = list(gt)
#     pc = Counter(p)
#     gc = Counter(g)
#     inter = sum((pc & gc).values())
#     if inter == 0:
#         return {"score": 0.0, "metric": "char_f1"}
#     precision = inter / max(1, len(p))
#     recall = inter / max(1, len(g))
#     f1 = 2 * precision * recall / max(1e-12, (precision + recall))
#     return {"score": float(f1), "metric": "char_f1"}
#
#
# @torch.no_grad()
# def generate_one(model, tokenizer, preprocess, device: str, prompt: str, images: List[Image.Image], args) -> str:
#     # 图像 -> pixel_values: (bs=1, num, c, h, w)
#     pixel_values = None
#     if images:
#         img_tensors = [MiniMindVLM.image2tensor(im, preprocess).to(device) for im in images]
#         # 每个 tensor: (1, c, h, w) -> 堆叠成 (1, num, c, h, w)
#         pixel_values = torch.stack([t.squeeze(0) for t in img_tensors], dim=0).unsqueeze(0)
#
#     inputs_text = _build_inputs(tokenizer, model, prompt)
#     inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(device)
#
#     generated_ids = model.generate(
#         inputs=inputs["input_ids"],
#         attention_mask=inputs.get("attention_mask", None),
#         max_new_tokens=args.max_new_tokens,
#         do_sample=bool(args.do_sample),
#         top_p=args.top_p,
#         temperature=args.temperature,
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#         pixel_values=pixel_values,
#     )
#
#     gen = generated_ids[0][inputs["input_ids"].shape[1]:]
#     return tokenizer.decode(gen, skip_special_tokens=True)
#
#
# def iter_parquet_rows(parquet_path: str):
#     table = pq.read_table(parquet_path)
#     cols = table.column_names
#     for i in range(len(table)):
#         row = {}
#         for c in cols:
#             row[c] = table[c][i].as_py()
#         yield i, row
#
#
# def main():
#     parser = argparse.ArgumentParser(description="MiniMind-V Dataset Eval")
#     parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
#     parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
#     parser.add_argument('--weight', default='sft_vlm', type=str, help="权重名称前缀（pretrain_vlm, sft_vlm）")
#     parser.add_argument('--hidden_size', default=512, type=int)
#     parser.add_argument('--num_hidden_layers', default=8, type=int)
#     parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
#     parser.add_argument('--vision_model_path', default='./model/vision_model/clip-vit-base-patch16', type=str)
#
#     parser.add_argument('--parquet_path', default='./dataset/sft_i2t.parquet', type=str, help='要评测的parquet数据集路径')
#     parser.add_argument('--out_jsonl', default='./eval_results.jsonl', type=str, help='输出jsonl路径')
#     parser.add_argument('--prompt_col', default='prompt', type=str, help='parquet中prompt列名；若不存在可从conversations抽取')
#     parser.add_argument('--gt_col', default='', type=str, help='parquet中ground-truth列名（可空；空则从conversations抽取assistant）')
#     parser.add_argument('--image_col', default='image_bytes', type=str, help='parquet中图片bytes列名')
#
#     parser.add_argument('--max_new_tokens', default=8192, type=int)
#     parser.add_argument('--temperature', default=0.65, type=float)
#     parser.add_argument('--top_p', default=0.85, type=float)
#     parser.add_argument('--do_sample', default=1, type=int)
#     parser.add_argument('--seed', default=2026, type=int)
#     parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
#     parser.add_argument('--limit', default=-1, type=int, help='只评测前N条；-1表示全量')
#     args = parser.parse_args()
#
#     setup_seed(args.seed)
#     model, tokenizer, preprocess = init_model(args)
#
#     gt_col = args.gt_col.strip() or None
#
#     os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or '.', exist_ok=True)
#     n = 0
#     t0 = time.time()
#
#     row_iter = iter_parquet_rows(args.parquet_path)
#     if tqdm is not None:
#         row_iter = tqdm(row_iter, desc='Evaluating', unit='sample')
#
#     with open(args.out_jsonl, 'w', encoding='utf-8') as fw:
#         for idx, row in row_iter:
#             if args.limit > 0 and n >= args.limit:
#                 break
#
#             prompt, gt, conversations = _extract_eval_fields(row, args.prompt_col, gt_col)
#
#             image_bytes = row.get(args.image_col)
#             images = _load_image_from_bytes(image_bytes)
#
#             st = time.time()
#             pred = generate_one(model, tokenizer, preprocess, args.device, prompt, images, args)
#             latency = time.time() - st
#
#             # score_obj = _simple_score(pred, gt)
#
#             record = {
#                 "idx": idx,
#                 "parquet_path": args.parquet_path,
#                 "prompt": prompt,
#                 "pred": pred,
#                 "gt": gt,
#                 # "score": score_obj.get("score"),
#                 # "metric": score_obj.get("metric"),
#                 "latency_s": latency,
#                 "has_image": bool(images),
#             }
#             if conversations is not None:
#                 record["conversations"] = conversations
#
#             fw.write(json.dumps(record, ensure_ascii=False) + "\n")
#             fw.flush()
#             n += 1
#
#     dt = time.time() - t0
#     print(f"Done. samples={n}, time={dt:.2f}s, avg={dt/max(1,n):.3f}s/sample, out={args.out_jsonl}")
#
#
# if __name__ == '__main__':
#     main()
