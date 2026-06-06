import torch
import re
text ='<reasoning>\n1. 图中是一个小孩坐在摩托车或电动车上，手握车把，看似在“驾驶”。\n2. 小孩头上戴着一个由许多小纸片或布条组成的奇特头饰，像狮子鬃毛或某种手工帽，造型夸张又童趣。\n3. 小孩表情专注甚至带点滑稽，与成人驾驶的严肃场景形成反差。\n4. “可爱”源于孩童天真模仿大人的行为 + 手工头饰的创意；“不寻常”是因为孩子不该开车、头饰非日常穿戴，组合出超现实感。\n</reasoning>\n\n<answer>\n孩子模仿大人骑车的认真模样搭配手工制作的夸张头饰，既显童真可爱，又因不合常理而显得荒诞有趣。\n</answer>\n'


@torch.no_grad()
def compute_format_reward(texts):
    """格式奖励：严格检查是否为
       ^<reasoning>...</reasoning><answer>...</answer>$ 这种格式。
    """
    # 非贪婪匹配，并允许中间有任意字符（包括换行）
    pattern = re.compile(r"^<reasoning>.*?</reasoning>\s*<answer>.*?</answer>$", re.DOTALL | re.IGNORECASE)
    rewards = []
    for t in texts:
        if not t:
            rewards.append(0.0)
            continue
        # 去掉首尾空白，避免多余换行/空格影响匹配
        s = t.strip()
        match = pattern.match(s)
        rewards.append(1.0 if match  else 0.0)
    return torch.tensor(rewards, dtype=torch.float32)
print(compute_format_reward([text]))
print(len('图中显示一只狗正坐在沙滩上，两只手牵着沙滩小狗，可能正在看东西。这只狗似乎正在享受一天中一天的休闲活动，可能是休闲活动或与家人共度时光。'))

