import time
import argparse
import os
import warnings
import torch
import random
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_vlm import MiniMindVLM, VLMConfig
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from, trust_remote_code=True)
    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniMindVLM(
            VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)),
            vision_model_path="./model/siglip2-base-p32-256-ve"
        )
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    else:##hugface格式
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        model.vision_encoder, model.processor = MiniMindVLM.get_vision_model("./model/siglip2-base-p32-256-ve")##===================================
    get_model_params(model, model.config)##========
    return model.half().eval().to(args.device), tokenizer, \
        model.processor


def main():
    parser = argparse.ArgumentParser(description="MiniMind-V Chat")
    # parser.add_argument('--load_from', default='./minimind-3v-moe', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")#加载transformers格式 ----ModuleNotFoundError: No module named 'transformers_modules.minimind_hyphen_3v_hyphen_moe'
    parser.add_argument('--load_from', default='./minimind-3v', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")#加载transformers格式



    # parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")#加载原生PyTorch权重
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='sft_vlm', type=str, help="权重名称前缀（pretrain_vlm, sft_vlm）")##===================================##===================================##===================================##===================================



    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")##===================================
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.7, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--image_dir', default='./dataset/eval_images/', type=str, help="测试图像目录")##===================================
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    # parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")##===================================##===================================##===================================##===================================
    parser.add_argument('--open_thinking', default=1, type=int, help="是否开启自适应思考（0=否，1=是）")##===================================##===================================##===================================##===================================
    args = parser.parse_args()
    model, tokenizer, preprocess = init_model(args)##===================================



    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # 自动测试image_dir中的所有图像
    prompt = "<image>\n请描述这张图中的主要物体和场景。"
    # prompt = "<image>\nPlease illustrate the image through your words."
    for image_file in sorted(os.listdir(args.image_dir)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            setup_seed(random.randint(1, 31415926))
            image_path = os.path.join(args.image_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            pixel_values = {k: v.to(args.device) for k, v in MiniMindVLM.image2tensor(image, preprocess).items()}##===================================
            ##
            messages = [{"role": "user",                     ##<|image_pad|>*64image_token_len
                         "content": prompt.replace('<image>', model.config.image_special_token * model.config.image_token_len)}]
            inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
            inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)
            print(f'[图像]: {image_file}')
            print(f"💬: {repr(prompt)}")



            print('🤖: ', end='')
            st = time.time()
            generated_ids = model.generate(
                inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p, temperature=args.temperature,
                pixel_values=pixel_values##===================================
            )
            gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
            print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')##[Speed]: 22.00 tokens/s
if __name__ == "__main__":
    main()
# modelscope download --model gongjy/minimind-3v README.md --local_dir ./minimind-3v  ##===================================


# #Model Params: 65.09M
# [图像]: image-01-golden-dog-balloons.jpg
# 💬: '<image>\n请描述这张图中的主要物体和场景。'
# 🤖:
# </think>
# 这张图片中的主要物体是金色的金色狗，可能是一只金毛寻回犬，它正在一个阳光明媚的日子里跳跃。这只狗被描绘在草地上，暗示着一个轻松愉快的场景，可能是在一个公园或农场。狗的姿势和表情，头部略微转向前方，表明它正处于运动中。背景是一个晴朗的天空，有几朵散落的云彩，可能是一天中的景色。狗的姿势和姿态表明它正在进行积极的互动，可能正在享受一场愉快的户外活动。
# [Speed]: 26.95 tokens/s
#
#
# [图像]: image-02-rainbow-umbrella-street.jpg
# 💬: '<image>\n请描述这张图中的主要物体和场景。'
# 🤖:
# </think>
# 这张图片中的主要物体是一把彩色雨伞。雨伞设计用于防雨，其特点是光滑的质地和可见的彩色条纹，与雨伞的颜色相匹配。雨伞的形状是圆形的，带有多个开口，颜色从红色、绿色、黄色到蓝色不等。雨伞的外观表明它是新的或者保养良好，没有被风吹雨打。雨伞的大小和形状与雨伞相匹配，但它是为雨伞设计的。雨伞的位置在背景中，背景中有建筑物，这表明这是一个城市环境。
# [Speed]: 33.95 tokens/s
#
#
# [图像]: image-03-cherry-blossom-bike.jpg



