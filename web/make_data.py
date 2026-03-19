import json
import os

# 你的 5 个模型文件夹名称
models =["cosyvoice", "cosyvoice2", "emovoice", "indextts2", "my"]
output_data =[]

# 读取 jsonl 数据
with open("text/test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
            
        item = json.loads(line.strip())
        key = item["key"]
        text = item["target_text"]
        
        # 获取前 3 句上下文，如果不足 3 句则全取
        context_list = item.get("context", [])[:3]
        
        # 自动拼接音频路径
        audios = {}
        for model in models:
            # 假设你的音频是 wav 格式，并且以 key 命名
            audios[model] = f"./audio/{model}/{key}.wav"
            
        output_data.append({
            "id": key,
            "context": context_list,
            "text": text,
            "audios": audios
        })

# 将数据写入 data.js
with open("data.js", "w", encoding="utf-8") as f:
    # ensure_ascii=False 保证中文正常显示，不变成 Unicode 转义
    json_str = json.dumps(output_data, ensure_ascii=False, indent=4)
    f.write(f"const testData = {json_str};")

print(f"成功生成 data.js！共处理了 {len(output_data)} 条数据。")