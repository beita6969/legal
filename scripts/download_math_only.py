import os
import json
from datasets import load_dataset
from pathlib import Path
import sys

# 设置本地代理
os.environ["http_proxy"] = "http://127.0.0.1:10808"
os.environ["https_proxy"] = "http://127.0.0.1:10808"

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

def save_raw_dataset(data, name):
    """单独保存每个原始数据集"""
    if not data:
        print(f"❌ {name} 数据为空，未保存")
        return
    path = os.path.join(RAW_DIR, f"{name}.jsonl")
    print(f"💾 保存 {name} ({len(data)} 条) 到 {path}")
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def load_and_save_math():
    """尝试下载 MATH 数据集 (使用镜像源)"""
    print("📚 正在尝试下载 MATH 数据集...")
    
    # 尝试: 使用公开的 MATH 镜像
    # 官方的 'hendrycks/competition_math' 有时会有访问限制
    # 这里我们尝试使用 'HuggingFaceH4/mathematical_reasoning' 中包含的 math 子集
    # 或者直接使用 'AI-MO/NuminaMath-CoT' 这种高质量的衍生集
    
    # 最可靠的替代: 'xiyuez/im-math' 或 'metamath/MetaMathQA'
    # 但为了保持原汁原味，我们尝试: 'lighteval/MATH' (通常是开放的)
    # 如果还是不行，我们使用 'qwe11000/math' 这种个人备份
    
    alternatives = [
        "lighteval/MATH",
        "qwe11000/math",  # 个人备份，通常无权限限制
        "HuggingFaceH4/math_eval"
    ]
    
    for dataset_id in alternatives:
        try:
            print(f"🔄 尝试 ID: '{dataset_id}' ...")
            ds = load_dataset(dataset_id, split="train", trust_remote_code=True)
            print(f"✅ 成功加载 '{dataset_id}'")
            
            processed = []
            for item in ds:
                # 兼容不同数据集的字段名
                problem = item.get("problem") or item.get("question")
                solution = item.get("solution") or item.get("answer")
                
                if problem and solution:
                    processed.append({
                        "problem": problem,
                        "problem_type": "math",
                        "source": "math",
                        "ground_truth": solution
                    })
            
            if processed:
                save_raw_dataset(processed, "math")
                print("🎉 MATH 数据集下载并保存完成！")
                return
            
        except Exception as e:
            print(f"⚠️  {dataset_id} 失败: {e}")
            
    print("❌ 所有尝试均失败。建议手动下载 data/raw/math.jsonl")

if __name__ == "__main__":
    load_and_save_math()

