import pyarrow.parquet as pq
import json
import os

import pyarrow.parquet as pq
import pyarrow as pa
import os
import json
def preview_parquet_data(input_parquet, n=3):
    if not os.path.exists(input_parquet):
        print(f"文件未找到: {input_parquet}")
        return
    print(f"正在读取 {input_parquet} ...")
    # 读取原始 parquet 文件
    table = pq.read_table(input_parquet)
    total_rows = len(table)
    print(f"数据集总条数: {total_rows}")


    n = min(n, total_rows)
    for i in range(n):
        print(table[i])##===================================


        print(f"=== 第 {i+1} 条数据 ===")
        # 读取并解析 conversations
        conv_raw = table['conversations'][i].as_py()##===================================
        if type(conv_raw) != str:
            conv_raw = str(conv_raw)##========
        try:
            conversations = json.loads(conv_raw)##========
            print("【对话内容】:")
            print(json.dumps(conversations, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"【对话内容 解析失败】: {conv_raw}")


        # 读取 image_bytes
        img_data = table['image_bytes'][i].as_py()##===================================
        if isinstance(img_data, list):
            print("list")
            print(f"【图片数据】: 包含 {len(img_data)} 张图片, 大小分别为 {[len(img) for img in img_data]} 字节")
        elif img_data is not None:
            print("not None")
            print(f"【图片数据】: 包含 1 张图片, 大小为 {len(img_data)} 字节")
        else:
            print("无图片")
            print("【图片数据】: 无图片")
        print("-" * 60 + "\n")


def extract_top_k_data(input_parquet, output_parquet, k=100):
    if not os.path.exists(input_parquet):
        print(f"文件未找到: {input_parquet}")
        return
    print(f"正在读取 {input_parquet} ...")
    # 读取原始 parquet 文件
    table = pq.read_table(input_parquet)
    total_rows = len(table)
    print(f"数据集总条数: {total_rows}")

    # 提取前 k 条数据
    sub_table = table.slice(0, k)

def extract_top_k_data(input_parquet, output_parquet, k=100):
    if not os.path.exists(input_parquet):
        print(f"文件未找到: {input_parquet}")
        return
    print(f"正在读取 {input_parquet} ...")
    # 读取原始 parquet 文件
    table = pq.read_table(input_parquet)
    total_rows = len(table)
    print(f"数据集总条数: {total_rows}")


    # 提取前 k 条数据
    sub_table = table.slice(0, k)
    ##
    ##
    # 写入新的 parquet 文件
    pq.write_table(sub_table, output_parquet)  ##===================================
    print(f"成功提取前 {k} 条数据！")
    print(f"新文件已保存至: {output_parquet}")


    # # 遍历并处理 conversations
    # conversations_list = sub_table['conversations'].to_pylist()  ##===================================
    # new_conversations = []
    # for conv_str in conversations_list:
    #     try:
    #         conv = json.loads(conv_str)##========
    #         for msg in conv:
    #             if msg.get("role") == "assistant":
    #                 msg["content"] = f"{msg.get('content', '')}"
    #         new_conversations.append(json.dumps(conv, ensure_ascii=False))##========
    #     except Exception as e:
    #         # 如果解析失败，则保留原样
    #         new_conversations.append(conv_str)
    # # 替换原本的 conversations 列
    # col_idx = sub_table.schema.get_field_index('conversations')
    # new_conv_array = pa.array(new_conversations, type=pa.string())
    # sub_table = sub_table.set_column(col_idx, 'conversations', new_conv_array)
    # ##
    # ##
    # # 写入新的 parquet 文件
    # pq.write_table(sub_table, output_parquet)  ##===================================
    # print(f"成功提取前 {k} 条数据！")
    # print(f"新文件已保存至: {output_parquet}")
if __name__ == '__main__':
    input_file = './sft_i2t.parquet'##===================================
    ## input_file = './vlm_reasoning_2149_grpo.parquet'
    ## output_file = '/home/bnw/WJS/pretrain/minimind-v/dataset/sft_cold_start_i2t_5000.parquet'
    output_file = './sft_i2t_mini1000.parquet'##===================================
    ##
    input_file = './Innovator-VL-RL-172K_RL_part000006.parquet'#数据集总条数: 8698##===================================
    output_file = './Innovator-VL-RL-172K_RL_part000006_mini1000.parquet'##===================================
    ##
    ##
    # preview_parquet_data(input_file, n=3)#10#可以修改n的值来查看更多条目##===================================
    extract_top_k_data(input_file, output_file, k=20)##1000##===================================




                # #数据集总条数: 2904511##===================================
                # === 第 1 条数据 ===
                # 【对话内容】:
                # [
                #   {
                #     "role": "user",
                #     "content": "<image>How does the use of negative space and color contrast contribute to the overall impact of the image?"
                #   },
                #   {
                #     "role": "assistant",
                #     "content": "The use of negative space, which is the white and unoccupied area surrounding the figure, emphasizes the subject and draws the viewer's attention directly to the character. This technique creates a stark contrast with the golden armor, making the figure the focal point of the composition. The limited use of color, primarily gold against the white background, adds a dramatic and regal quality to the image. It suggests that the character is perhaps a solitary and significant figure within the context of its environment. The high contrast between the character and the background also gives the impression of radiance and power, potentially reflective of the character's status or abilities within the game's narrative.",
                #     "reasoning_content": ""
                #   }
                # ]
                # 【图片数据】: 包含 1 张图片, 大小为 5415 字节
#数据集总条数: 2149##===================================
# === 第 1 条数据 ===
# 【对话内容】:
# [
#   {
#     "role": "user",
#     "content": "闹钟的位置对睡眠质量有什么影响？<image>"
#   },
#   {
#     "role": "assistant",
#     "content": "<reasoning>\n闹钟的位置影响睡眠质量主要体现在心理和生理层面。若闹钟离床太近，人容易在夜间无意识查看时间，增加焦虑感，干扰深度睡眠；若太远，则可能因听不清而错过闹铃，导致起床压力或迟到焦虑。理想位置是伸手可及但需稍起身才能关闭的地方，既避免频繁看时间，又能确保被唤醒。此外，数字显示光（如图中红色LED）若在黑暗中过亮，也可能抑制褪黑素分泌，影响入睡与睡眠连续性。\n</reasoning>\n\n<answer>\n闹钟应放在伸手可及但非紧邻床头的位置，以减少夜间看时间的焦虑并确保能被唤醒；同时避免强光直射眼睛，以免干扰睡眠节律。"
#   }
# ]
# 【图片数据】: 包含 1 张图片, 大小分别为 [2392] 字节
# ------------------------------------------------------------


