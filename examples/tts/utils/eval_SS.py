import os
import re
import glob
import csv
import torch
import torchaudio
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WavLMForXVector
from tqdm import tqdm
from collections import defaultdict

# ================= 配置区域 =================
PRED_AUDIO_DIR = "/data/Shizihui/MyModel/ckp/UT-EN-23/EN-5/pred_audio/neutral_prompt_speech"  # 你的模型生成的音频文件夹
GT_AUDIO_DIR = None      # 真实的参考音频文件夹 (如果不算还原度，设为 None)
OUTPUT_CSV = "/data/Shizihui/MyModel/ckp/UT-EN-23/EN-5/pred_audio/neutral_prompt_speech/_ss_results.csv"  # 导出的结果文件名
MODEL_NAME = "microsoft/wavlm-base-plus-sv" # WavLM 预训练模型
# ============================================

def setup_wavlm():
    """初始化并加载 WavLM 模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading WavLM model on {device}...")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = WavLMForXVector.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return model, extractor, device

def extract_embedding(wav_path, model, extractor, device):
    """提取单条音频的 Speaker Embedding"""
    try:
        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)
        
        wav = wav.squeeze().numpy()
        # 极短音频保护 (WavLM需要一定长度的输入)
        if len(wav) < 8000: 
            return None 

        inputs = extractor(wav, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb = model(**inputs).embeddings
        return emb
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

def parse_filename(filename):
    """解析文件名，提取故事名和序号"""
    match = re.match(r"^(.*)[-_](\d+)\.wav$", filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None

def main():
    model, extractor, device = setup_wavlm()
    
    print("Scanning audio files...")
    pred_files = glob.glob(os.path.join(PRED_AUDIO_DIR, "*.wav"))
    story_dict = defaultdict(list)
    
    for path in pred_files:
        filename = os.path.basename(path)
        story_name, idx = parse_filename(filename)
        if story_name is not None:
            story_dict[story_name].append((idx, path))
            
    for story_name in story_dict:
        story_dict[story_name].sort(key=lambda x: x[0])

    # 用于保存写入 CSV 的数据行
    csv_data =[]
    intra_ss_scores = []
    global_ss_scores =[]  # ⭐️ 新增：全局防漂移度列表
    gt_ss_scores =[]
    
    print("Calculating Speaker Similarity...")
    for story_name, sentences in tqdm(story_dict.items(), desc="Processing Stories"):
        if not sentences:
            continue
            
        # 1. 批量提取当前故事的所有 Embedding
        embeddings = {}
        for idx, path in sentences:
            emb = extract_embedding(path, model, extractor, device)
            if emb is not None:
                embeddings[idx] = emb
                
        # 获取该故事的第一句话作为全局锚点 (Anchor)
        first_idx, first_path = sentences[0]
                
        # 2. 计算 上下文连贯度 (Intra-SS: 第 i 句 vs 第 i+1 句)
        for i in range(len(sentences) - 1):
            idx1, path1 = sentences[i]
            idx2, path2 = sentences[i+1]
            
            if idx1 in embeddings and idx2 in embeddings:
                sim = F.cosine_similarity(embeddings[idx1], embeddings[idx2]).item()
                intra_ss_scores.append(sim)
                
                # 记录到 CSV 数据列表中
                csv_data.append([
                    "Intra-SS (Local Context)", 
                    story_name, 
                    os.path.basename(path1), 
                    os.path.basename(path2), 
                    f"{sim:.4f}"
                ])
                
        # 3. ⭐️ 新增：计算 全局防漂移度 (Global-SS: 第 i 句 vs 第 1 句)
        for i in range(1, len(sentences)):
            curr_idx, curr_path = sentences[i]
            
            if first_idx in embeddings and curr_idx in embeddings:
                sim = F.cosine_similarity(embeddings[first_idx], embeddings[curr_idx]).item()
                global_ss_scores.append(sim)
                
                # 记录到 CSV 数据列表中
                csv_data.append([
                    "Global-SS (Anchor vs Curr)", 
                    story_name, 
                    os.path.basename(first_path), 
                    os.path.basename(curr_path), 
                    f"{sim:.4f}"
                ])
                
        # 4. 计算 还原度 (GT-SS: 预测 vs 真实)
        if GT_AUDIO_DIR and os.path.exists(GT_AUDIO_DIR):
            for idx, pred_path in sentences:
                filename = os.path.basename(pred_path)
                gt_path = os.path.join(GT_AUDIO_DIR, filename)
                
                if os.path.exists(gt_path) and idx in embeddings:
                    gt_emb = extract_embedding(gt_path, model, extractor, device)
                    if gt_emb is not None:
                        sim = F.cosine_similarity(embeddings[idx], gt_emb).item()
                        gt_ss_scores.append(sim)
                        
                        # 记录到 CSV 数据列表中
                        csv_data.append([
                            "GT-SS (Reconstruction)", 
                            story_name, 
                            filename, 
                            f"[GT] {filename}", 
                            f"{sim:.4f}"
                        ])

    # ================= 计算平均分 =================
    avg_intra = sum(intra_ss_scores) / len(intra_ss_scores) if intra_ss_scores else 0.0
    avg_global = sum(global_ss_scores) / len(global_ss_scores) if global_ss_scores else 0.0
    avg_gt = sum(gt_ss_scores) / len(gt_ss_scores) if gt_ss_scores else 0.0

    # ================= 写入 CSV 文件 =================
    print(f"\nWriting detailed results to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow(["Eval_Type", "Story_Name", "Audio_A", "Audio_B", "SS_Score"])
        
        # 写入每一对音频的详细得分
        writer.writerows(csv_data)
        
        # 留几个空行，然后写入平均分汇总
        writer.writerow([])
        writer.writerow(["--- SUMMARY ---", "", "", "", ""])
        if intra_ss_scores:
            writer.writerow(["Average Intra-SS (Local)", "All Stories", f"Total Pairs: {len(intra_ss_scores)}", "", f"{avg_intra:.4f}"])
        if global_ss_scores:
            writer.writerow(["Average Global-SS (Anchor)", "All Stories", f"Total Pairs: {len(global_ss_scores)}", "", f"{avg_global:.4f}"])
        if gt_ss_scores:
            writer.writerow(["Average GT-SS (GroundTruth)", "All Stories", f"Total Pairs: {len(gt_ss_scores)}", "", f"{avg_gt:.4f}"])

    # ================= 终端输出总结 =================
    print("="*60)
    print("📊 Speaker Similarity (SS) Summary")
    print("="*60)
    if intra_ss_scores:
        print(f"1. 平均局部连贯度 (Intra-SS):   {avg_intra:.4f}  ({len(intra_ss_scores)} 对音频) - 衡量相邻句音色平滑度")
    if global_ss_scores:
        print(f"2. 平均全局防漂移 (Global-SS):  {avg_global:.4f}  ({len(global_ss_scores)} 对音频) - 衡量全篇与首句相似度")
    if gt_ss_scores:
        print(f"3. 平均音色还原度 (GT-SS):      {avg_gt:.4f}  ({len(gt_ss_scores)} 对音频) - 衡量与原声还原度")
    print("="*60)
    print(f"✅ 详细配对得分已保存至: {os.path.abspath(OUTPUT_CSV)}")

if __name__ == "__main__":
    main()