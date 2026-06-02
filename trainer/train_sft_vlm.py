import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import datasets
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, VLMConfig##===================================
from dataset.lm_dataset import VLMDataset##===================================
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, init_vlm_model, vlm_checkpoint, SkipBatchSampler, vlm_collate_fn

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels, pixel_values) in enumerate(loader, start=start_step + 1):##===================================
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()} if isinstance(pixel_values, dict) else pixel_values.to(args.device)



        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels, pixel_values=pixel_values)##===================================



            loss = res.loss + res.aux_loss##===================================
            loss = loss / args.accumulation_steps##===================================

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps##===================================
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if vlm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            clean_state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith('vision_encoder.')
            }
            clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}  # 半精度保存并移到CPU
            torch.save(clean_state_dict, ckp)
            vlm_checkpoint(vlm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict, clean_state_dict

        del input_ids, labels, pixel_values, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")##===================================
    parser.add_argument('--save_weight', default='sft_vlm', type=str, help="保存权重的前缀名")##===================================

    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")##===================================
    parser.add_argument("--data_path", type=str, default="../dataset/sft_i2t.parquet", help="训练数据路径")##===================================
    parser.add_argument('--from_weight', default='pretrain_vlm', type=str, help="基于哪个权重训练，为none则不基于任何权重训练none  llm----pretrain_vlm  sft_vlm")##===================================##===================================
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")##===================================##===================================
    ##默认 --freeze_llm 1即训练vision_proj 和LLM 首尾层，保留中间层原有语言能力：
    parser.add_argument('--freeze_llm', default=1, type=int, choices=[0, 1, 2], help="冻结策略（0=完全可训练，   1=冻结proj+解冻首尾层，   2=完全冻结仅训练proj）")##Visual Encoder一直完全冻结##===================================##===================================
# ##参数说明：
# --save_dir: 保存权重的目录
# --save_weight: 保存权重的前缀名
#
# --from_weight: 基础权重名称（none  llm----, pretrain_vlm, sft_vlm等）
# --from_resume: 是否续训（0=从头开始，1=从检查点继续）
# --freeze_llm: 冻结策略（0=全参可训，1=proj + LLM 首尾层，2=仅训 proj）。Pretrain 默认 2，SFT 默认 1
# 更多可直接参考代码
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
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
    parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-SFT", help="wandb项目名")
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
        wandb_run_name = f"MiniMind-V-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer, preprocess = init_vlm_model(vlm_config, from_weight=args.from_weight, device=args.device, freeze_llm=args.freeze_llm)#1=冻结proj+解冻首尾层##===================================##===================================



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

# 开始训练
# 推荐直接执行 SFT。默认 --freeze_llm 1，即训练 vision_proj 和 LLM 首尾层，保留中间层原有语言能力：
# python train_sft_vlm.py --epochs 2 --from_weight llm##===================================
# ##
#         如果希望让 Projector 先完成一轮图文对齐，再进入 SFT，可额外执行 Pretrain：
#         python train_pretrain_vlm.py --epochs 2 --from_weight llm##===================================
#         python train_sft_vlm.py --epochs 2 --from_weight pretrain_vlm##===================================
#         执行完成后，out/ 下会生成 sft_vlm_*.pth 作为 SFT 权重。
#
#
#         注：训练须知
#         支持断点续训：添加--from_resume 1参数可从上次中断处继续训练
#         支持GPU数量变化：续训时GPU数量改变会自动转换step
#         原子性保存：使用临时文件+替换机制，防止保存过程中断导致权重损坏
#         每次保存同时生成out/**.pth（模型权重）和checkpoints/**_resume.pth（训练状态）文件
#         # 训练中断后，使用相同命令并添加 --from_resume 1
#         python train_sft_vlm.py --epochs 4 --from_resume 1
# ##参数说明：
# --save_dir: 保存权重的目录
# --save_weight: 保存权重的前缀名
#
# --from_weight: 基础权重名称（none  llm----, pretrain_vlm, sft_vlm等）
# --from_resume: 是否续训（0=从头开始，1=从检查点继续）
# --freeze_llm: 冻结策略（0=全参可训，1=proj + LLM 首尾层，2=仅训 proj）。Pretrain 默认 2，SFT 默认 1
# 更多可直接参考代码


# Tip
# 训练脚本均为 PyTorch 原生框架，均支持多卡加速，假设你的设备有 N (N＞1) 张显卡：
# 单机N卡启动训练方式 (DDP, 支持多机多卡集群)
# torchrun --nproc_per_node N train_xxx.py
# ##
# 可根据需要开启wandb记录训练过程
# # 需要登录: wandb login
# torchrun --nproc_per_node N train_xxx.py --use_wandb
# # and
# python train_xxx.py --use_wandb
# 通过添加--use_wandb参数，可以记录训练过程，训练完成后，可以在wandb网站上查看训练过程。通过修改wandb_project 和wandb_run_name参数，可以指定项目名称和运行名称。
#

##===================================##===================================##===================================##===================================
##===================================##===================================##===================================##===================================
##===================================##===================================##===================================##===================================
##===================================##===================================##===================================##===================================
# 失败----数据的问题##===================================


