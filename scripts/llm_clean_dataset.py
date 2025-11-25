#!/usr/bin/env python3
"""
LLM 数据清洗器
使用本地强大的 LLM (GPT OSS 120B @ port 8002) 对数据集进行深度扫描。
目标：剔除 "问题-答案" 不匹配、指代不明或答案错误的样本。
"""

import json
import os
import asyncio
from tqdm import tqdm
from openai import OpenAI

# 配置
INPUT_FILE = "11/integrated_aflow_roll/data/ready_to_train/train.jsonl"
OUTPUT_FILE = "11/integrated_aflow_roll/data/ready_to_train/train_llm_cleaned.jsonl"
BAD_CASE_FILE = "11/integrated_aflow_roll/data/ready_to_train/dropped_samples.jsonl"

LLM_CONFIG = {
    "base_url": "http://localhost:8002/v1",
    "api_key": "sk-dummy",
    "model": "/home/yijia/lhy/openai/gpt-oss-120b"
}

client = OpenAI(base_url=LLM_CONFIG["base_url"], api_key=LLM_CONFIG["api_key"])

def check_sample(item):
    """使用 LLM 判断样本质量"""
    problem = item.get("problem", "")
    ground_truth = item.get("ground_truth", "")
    p_type = item.get("problem_type", "qa")
    
    # Code 类型通常比较可靠（且上下文太长），跳过深度检查，只做基础检查
    if p_type == "code":
        if not problem or not ground_truth:
            return False, "Empty code problem or solution"
        return True, ""

    prompt = f"""Task: Verify if the following Question-Answer pair is valid, self-contained, and correct.

Question: {problem}
Ground Truth Answer: {ground_truth}

Verification Criteria:
1. Is the question meaningful and self-contained (not dependent on missing context)?
2. Is the Ground Truth answer logically correct for the question?
3. Does the Ground Truth make sense (e.g., rejecting "length and width" as a definition for "commercial paper")?

Respond with JSON only:
{{
    "valid": true/false,
    "reason": "short explanation"
}}
"""
    
    try:
        response = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=[
                {"role": "system", "content": "You are a strict data quality auditor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=100,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        return result.get("valid", False), result.get("reason", "Unknown")
        
    except Exception as e:
        print(f"⚠️  LLM 调用失败: {e}")
        return True, "LLM Check Failed (Assume Valid)"

def main():
    print("🧹 开始 LLM 数据深度清洗...")
    print(f"  输入: {INPUT_FILE}")
    print(f"  模型: {LLM_CONFIG['model']}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 文件不存在: {INPUT_FILE}")
        return

    # 读取数据
    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()
        
    total = len(lines)
    print(f"  总样本数: {total}")
    
    valid_data = []
    dropped_data = []
    
    # 并发太高可能压垮本地服务，这里使用简单的顺序处理或小批次
    # 为了进度可见，用 tqdm
    for line in tqdm(lines):
        item = json.loads(line)
        is_valid, reason = check_sample(item)
        
        if is_valid:
            valid_data.append(item)
        else:
            item["drop_reason"] = reason
            dropped_data.append(item)
            
    # 保存结果
    print(f"\n📊 清洗结果:")
    print(f"  ✅ 保留: {len(valid_data)} ({len(valid_data)/total*100:.1f}%)")
    print(f"  🗑️  剔除: {len(dropped_data)} ({len(dropped_data)/total*100:.1f}%)")
    
    with open(OUTPUT_FILE, 'w') as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")
            
    with open(BAD_CASE_FILE, 'w') as f:
        for item in dropped_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"\n💾 已保存清洗后的数据: {OUTPUT_FILE}")
    print(f"💾 已保存剔除样本(供检查): {BAD_CASE_FILE}")

if __name__ == "__main__":
    main()


