import pandas as pd
import json
import os
from pathlib import Path

# 配置路径
RAW_DIR = "11/integrated_aflow_roll/data/raw"
MATH_PARQUET_PATH = os.path.join(RAW_DIR, "competition_math/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet")
OUTPUT_JSONL_PATH = os.path.join(RAW_DIR, "math.jsonl")

def convert_parquet_to_jsonl():
    print(f"🔄 正在将 Parquet 转换为 JSONL...")
    print(f"📂 输入: {MATH_PARQUET_PATH}")
    
    if not os.path.exists(MATH_PARQUET_PATH):
        print(f"❌ 错误: 未找到文件 {MATH_PARQUET_PATH}")
        # 尝试模糊查找
        parent = os.path.dirname(MATH_PARQUET_PATH)
        files = list(Path(parent).glob("*.parquet"))
        if files:
            print(f"ℹ️  找到替代文件: {files[0]}")
            df = pd.read_parquet(files[0])
        else:
            return
    else:
        df = pd.read_parquet(MATH_PARQUET_PATH)
    
    print(f"📊 加载了 {len(df)} 条数据")
    print(f"   列名: {list(df.columns)}")
    
    # 转换为标准格式
    processed_data = []
    for _, row in df.iterrows():
        processed_data.append({
            "problem": row.get("problem") or row.get("question"),
            "problem_type": "math",
            "source": "math",
            "ground_truth": row.get("solution") or row.get("answer")
        })
        
    # 保存
    print(f"💾 保存到: {OUTPUT_JSONL_PATH}")
    with open(OUTPUT_JSONL_PATH, 'w') as f:
        for item in processed_data:
            f.write(json.dumps(item) + "\n")
            
    print("✅ 转换完成！")

if __name__ == "__main__":
    convert_parquet_to_jsonl()


