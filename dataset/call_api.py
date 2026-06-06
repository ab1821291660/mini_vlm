# 说明：把 DASHSCOPE_API_KEY / BASE_URL / MODEL_NAME 按需改成你自己的
import base64
from typing import Optional, Union

from openai import OpenAI


# ---------------- 配置区：按需修改 ----------------
DASHSCOPE_API_KEY = "sk-xxxx"  # 替换成你的 API Key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.5-plus"   # 文本/图文都可用的模型名
IMAGE_MIME_TYPE = "image/jpeg"
# -----------------------------------


def _get_client() -> OpenAI:
    return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=BASE_URL)


def _encode_image_to_data_url(image: Union[bytes, str], mime_type: str = IMAGE_MIME_TYPE) -> str:
    """
    把图片转成 data url：
    - image: bytes（图片二进制） 或 str（图片路径）
    """
    if isinstance(image, str):
        with open(image, "rb") as f:
            image_bytes = f.read()
    elif isinstance(image, (bytes, bytearray, memoryview)):
        image_bytes = bytes(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)} (expect bytes or file path str)")

    if len(image_bytes) < 100:
        raise ValueError("Image bytes too small; may be invalid.")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _extract_text_from_content(content) -> str:
    """
    兼容 OpenAI SDK：message.content 可能是 str 或富文本 list。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
        return "".join(parts)
    return str(content)


def call_llm(prompt: str, thinking: bool = False) -> str:
    """
    纯文本调用，返回回答的字符串内容。

    Args:
        prompt: 用户输入。
        thinking: 是否开启“思考/推理”模式（如后端支持）。
    """
    client = _get_client()

    # 兼容不同 OpenAI-compatible 后端：部分厂商使用 extra_body 透传自定义参数。
    # 这里优先使用通用字段名 thinking；若你的后端需要别名（如 enable_thinking / reasoning 等），
    # 可在此处按需调整。
    extra_body = {"enable_thinking": True,"thinking_budget": 81920} if thinking else {"enable_thinking": False}

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt},
        ],
        extra_body=extra_body,
    )
    return _extract_text_from_content(completion.choices[0].message.content)


def call_vlm(image: Union[bytes, str], prompt: str, mime_type: str = IMAGE_MIME_TYPE) -> str:
    """
    图文多模态调用：
    - image: bytes 或 图片路径(str)
    - prompt: 文本提示词
    返回回答的字符串内容。
    """
    client = _get_client()
    data_url = _encode_image_to_data_url(image, mime_type=mime_type)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return _extract_text_from_content(completion.choices[0].message.content)


if __name__ == "__main__":
    # 简单自测（避免把 key 写进代码仓库）
    print(call_llm("你好，简单介绍一下你自己。",False))
    # print(call_vlm("/path/to/your.jpg", "这张图里有什么？"))
    pass