# minimind-v-GRPO

基于 **minimind-v** 多模态预训练模型进行 **GRPO** 强化学习训练（Reinforcement Learning），以提升小模型的“思考/推理”能力。

> 本仓库目标：在多模态（VLM）基础上，通过 GRPO 进行 RL 微调，让模型在推理类数据上表现更好。
> **使用前提**：请先完整跑通 **minimind-v** 的预训练/基础流程后再使用本仓库进行 GRPO 训练：  
> https://github.com/jingyaogong/minimind-v

---

## 目录

- [示例](#示例)
- [环境依赖](#环境依赖)
- [数据集准备](#数据集准备)
- [训练](#训练)
- [评测](#评测)



---
## 示例

| 图片 | 训练方法 | Prompt | 输出（含 `<reasoning>` / `<answer>`） |
| --- | --- | --- | --- |
| <img src="dataset/eval_images/彩虹瀑布-Rainbow-Falls.jpg" alt="彩虹瀑布-Rainbow-Falls" width="260" /> | **SFT** | 图片描述了什么 | `<reasoning>` 图像显示一群人站在水边，背景是山坡和一条河。背景中有彩虹和蓝色彩虹，暗示这是一个充满活力的水域。背景中有山坡，表明它可能是山区或山区。 `</reasoning>`<br><br>`<answer>` 这是一个充满活力的水域，有山坡和水流。 `</answer>` |
| <img src="dataset/eval_images/彩虹瀑布-Rainbow-Falls.jpg" alt="彩虹瀑布-Rainbow-Falls" width="260" /> | **GRPO** | 图片描述了什么 | `<reasoning>` 图像显示有绿色的山峦，有岩壁和彩虹，有大水流和红叶。水和阳光照射到山脉的中心，形成美丽的风景，有巨大的水景和景观。 `</reasoning>`<br><br>`<answer>` 图像显示有绿色的山峦，有山脉，表明天气多风、水流变化，象征着自然界的自然界特征。 `</answer>` |


---

## 数据集准备

本项目使用 `Minimind-v-RL-thinking` 数据集中的 parquet 文件。

1. 进入数据目录：
   ```bash
   cd dataset/train_data
   ```

2. 下载数据（使用 hf-mirror）：
   ```bash
   wget "https://hf-mirror.com/datasets/JinshengWei/Minimind-v-RL-thinking/resolve/main/vlm_reasoning_2149_grpo.parquet"
   wget "https://hf-mirror.com/datasets/JinshengWei/Minimind-v-RL-thinking/resolve/main/vlm_reasoning_3149_train.parquet"
   wget "https://hf-mirror.com/datasets/JinshengWei/Minimind-v-RL-thinking/resolve/main/vlm_reasoning_500_test.parquet"
   ```
---

## 训练

本项目训练流程分为两步：**SFT 冷启动** → **GRPO 强化学习**。

### 1) SFT 冷启动（Supervised Fine-Tuning）

使用监督数据对模型进行冷启动，得到后续 GRPO 的初始化权重：

```bash
python train_sft_vlm.py \
  --epochs 1 \
  --from_weight sft_vlm \
  --save_weight reasoning_sft1 \
  --data_path dataset/train_data/vlm_reasoning_3149_train.parquet \
  --save_dir ../out_vlm_reasoning \
  --use_wandb \
  --wandb_project vlm_reasoning
```

### 2) GRPO 训练（Reinforcement Learning）

在 SFT 权重基础上进行 GRPO 强化学习训练：

```bash
python train_grpo_vlm.py \
  --save_dir ../out_vlm_reasoning \
  --save_weight reasoning_grpo \
  --epochs 1 \
  --batch_size 1 \
  --learning_rate 5e-6 \
  --data_path dataset/train_data/vlm_reasoning_2149_grpo.parquet \
  --from_weight reasoning_sft1 \
  --num_samples 4 \
  --temperature 0.65 \
  --top_p 0.85 \
  --accumulation_steps 1 \
  --log_interval 20 \
  --save_interval 1000 \
  --use_wandb \
  --wandb_project vlm_reasoning_GRPO_v1
```

### 3) 直接下载已训练模型（可选）

如果不想本地训练，可以直接下载已训练好的 SFT / GRPO 权重：

- https://huggingface.co/JinshengWei/Minimind-v-RL/tree/main

下载对应 **SFT 模型** 与 **GRPO 模型** 后，按脚本的 `--from_weight` / 加载权重方式使用即可。
---

## 评测

本项目评测会输出模型的推理格式（`format`）与答案正确性（`accuracy`）两类分数，并提供单样本的预测/标注对比，便于快速定位问题。

### 评测样本示例（单条）

> 字段说明：`prompt` 为输入；`pred` 为模型输出；`gt` 为参考答案；`score` 为该样本的打分。

```json
{
  "idx": 0,
  "prompt": "与图像中的场景有关联的潜在风险是什么？<image>",
  "pred": "<reasoning>\n图像显示一个男人在厨房里用手机，周围是厨房用具、烤箱、炉灶、水槽、冰箱和微波炉等厨具，表明该场景可能发生在烤箱、炉灶、水槽、烤箱、电器等热源附近。附近有桌面和椅子，表明该场景可能发生在厨房环境中，可能是厨房休闲或休闲娱乐活动。\n</reasoning>\n\n<answer>\n与图像中的场景有关潜在风险包括：\n1. 潜在风险：与烹饪或烤箱相关的电气安全问题，如火灾危险（如烫伤）、烫伤或烧伤。\n2. 设备故障：烤箱、水槽、冰箱、电器设备可能损坏或损坏。\n3. 安全风险：在烤箱附近，可能伤害烤箱，导致损坏或损坏。\n4. 空间限制：厨房空间有限，可能导致意外摔倒或损坏，增加意外发生的风险。\n</answer>",
  "gt": "<reasoning>\n图像显示一个室内场景，有几个人围绕着一个发光的壁炉或加热设备。其中一人正俯身靠近该设备，可能在进行操作或添加燃料。潜在风险包括：\n\n1. 火灾风险：如果设备未正确维护或使用不当（如放置易燃物过近），可能引发火灾。\n2. 烫伤风险：靠近高温表面的人（尤其是儿童）可能被烫伤。\n3. 一氧化碳中毒：若为燃烧型取暖器且通风不良，可能导致有害气体积聚。\n4. 电气安全：若为电暖器，线路老化或过载也可能导致短路或起火。\n\n此外，房间内人员密集，一旦发生意外，疏散和救援难度增加。\n</reasoning>\n\n<answer>\n潜在风险包括火灾、烫伤、一氧化碳中毒及电气安全隐患，尤其在设备使用不当或通风不良时更易发生。\n</answer>",
  "latency_s": 1.5716335773468018,
  "has_image": true,
  "conversations": [
    {
      "role": "user",
      "content": "与图像中的场景有关联的潜在风险是什么？<image>"
    },
    {
      "role": "assistant",
      "content": "<reasoning>\n图像显示一个室内场景，有几个人围绕着一个发光的壁炉或加热设备。其中一人正俯身靠近该设备，可能在进行操作或添加燃料。潜在风险包括：\n\n1. 火灾风险：如果设备未正确维护或使用不当（如放置易燃物过近），可能引发火灾。\n2. 烫伤风险：靠近高温表面的人（尤其是儿童）可能被烫伤。\n3. 一氧化碳中毒：若为燃烧型取暖器且通风不良，可能导致有害气体积聚。\n4. 电气安全：若为电暖器，线路老化或过载也可能导致短路或起火。\n\n此外，房间内人员密集，一旦发生意外，疏散和救援难度增加。\n</reasoning>\n\n<answer>\n潜在风险包括火灾、烫伤、一氧化碳中毒及电气安全隐患，尤其在设备使用不当或通风不良时更易发生。\n</answer>"
    }
  ],
  "score": {
    "format": 100.0,
    "accuracy": 45.0
  }
}
```

### 汇总得分（平均）

| 方法 | avg_format_score | avg_accuracy_score |
| --- | ---:| ---:|
| SFT | 58.4000 | 22.1500 |
| GRPO | 99.0000 | 38.7300 |



---



## 致谢

- minimind-v 预训练模型与相关工作
- 数据集：JinshengWei/Minimind-v-RL-thinking