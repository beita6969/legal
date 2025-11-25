#!/usr/bin/env python3
"""
AFlow + ROLL 数据清洗与格式化工具 (Final Polish)

目标:
1. 统一数据格式，适配训练代码 (problem, problem_type, ground_truth, 等)
2. 为 LLM-as-a-Judge 准备辅助字段 (例如: 提取简短答案, 规范化类型)
3. 修复潜在的数据质量问题 (空值, 格式错误)
4. 生成可直接用于训练的最终 JSONL 文件

输出格式:
{
    "problem": "...",
    "problem_type": "math" | "qa" | "code",
    "ground_truth": "...",
    "source": "...",
    "difficulty": "easy" | "hard",
    "meta": {
        "short_answer": "...",  # 用于快速评估 (Regex/Exact Match)
        "test_cases": "...",    # 用于 Code 执行测试
        "context": "..."        # QA 可能会用到的上下文
    }
}
"""

import json
import os
import re
from tqdm import tqdm

# 配置
INPUT_DIR = "11/integrated_aflow_roll/data/final_mix"
OUTPUT_DIR = "11/integrated_aflow_roll/data/ready_to_train"
OS_ENV_PROXY = True # 是否使用代理 (本脚本不需要，但在服务器上可能需要)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_short_answer(ground_truth, problem_type):
    """
    尝试从 ground_truth 中提取简短答案，用于自动化评估
    """
    if not ground_truth:
        return ""
    
    gt_str = str(ground_truth)
    
    if problem_type == "math":
        # 尝试提取 \boxed{} 中的内容 (MATH 数据集常用)
        boxed = re.findall(r"\\boxed\{(.*?)\}", gt_str)
        if boxed:
            return boxed[-1] # 通常最后一个 boxed 是最终答案
        
        # GSM8K 通常是 #### 后面跟数字
        hash_split = gt_str.split("####")
        if len(hash_split) > 1:
            return hash_split[-1].strip()
            
        return gt_str # 如果都提取不到，返回原值
        
    elif problem_type == "qa":
        # QA 通常 ground_truth 比较短，或者直接就是答案
        return gt_str
        
    elif problem_type == "code":
        # Code 的 "答案" 通常是完整代码，很难提取 "简短答案"
        # 这里我们可能不需要 short_answer，因为 Code 有 test cases
        return ""
        
    return gt_str

def clean_item(item):
    """
    清洗单条数据
    """
    # 1. 基础字段检查
    if "problem" not in item or not item["problem"]:
        return None # 丢弃无问题的数据
        
    if "ground_truth" not in item or not item["ground_truth"]:
        # Code 数据集可能用 'canonical_solution' 或 'code'
        # 但之前的脚本应该已经统一为 ground_truth
        # 如果还是空的，尝试挽救
        return None

    # 2. 类型规范化
    p_type = item.get("problem_type", "unknown").lower()
    if p_type not in ["math", "qa", "code"]:
        p_type = "qa" # 默认为 QA
        
    # 3. 元数据提取
    meta = item.get("meta", {})
    
    # 提取简短答案 (辅助 Judge)
    short_ans = extract_short_answer(item["ground_truth"], p_type)
    if short_ans:
        meta["short_answer"] = short_ans
        
    # 处理 Code 特有的 Test Cases
    if p_type == "code":
        # 之前的脚本可能把 test 放在了顶层字段
        test_cases = item.get("test", "") or item.get("test_list", "")
        if test_cases:
            if isinstance(test_cases, list):
                test_cases = "\n".join(test_cases)
            meta["test_cases"] = test_cases
            
        # 确保 entry_point 存在
        entry_point = item.get("entry_point", "")
        if entry_point:
            meta["entry_point"] = entry_point

    # 4. 构建最终对象
    cleaned_item = {
        "problem": item["problem"].strip(),
        "problem_type": p_type,
        "ground_truth": str(item["ground_truth"]).strip(),
        "source": item.get("source", "unknown"),
        "difficulty": item.get("difficulty", "unknown"),
        "meta": meta
    }
    
    return cleaned_item

def process_file(input_filename, output_filename):
    input_path = os.path.join(INPUT_DIR, input_filename)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    print(f"🔄 处理: {input_filename} -> {output_filename}")
    
    if not os.path.exists(input_path):
        print(f"❌ 错误: 文件不存在 {input_path}")
        return
    
    valid_count = 0
    total_count = 0
    
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in tqdm(fin):
            total_count += 1
            try:
                item = json.loads(line)
                cleaned = clean_item(item)
                if cleaned:
                    fout.write(json.dumps(cleaned) + "\n")
                    valid_count += 1
            except json.JSONDecodeError:
                continue
                
    print(f"✅ 完成: {valid_count}/{total_count} 条数据有效")

def main():
    ensure_dir(OUTPUT_DIR)
    
    # 处理训练集
    process_file("train_2k.jsonl", "train.jsonl")
    
    # 处理测试集
    process_file("test_100.jsonl", "test.jsonl")
    
    print(f"\n🎉 所有数据处理完成！保存目录: {OUTPUT_DIR}")
    print(f"💡 建议更新 config/training.yaml:")
    print(f'   train_dataset: "{os.path.join(OUTPUT_DIR, "train.jsonl")}"')
    print(f'   test_dataset: "{os.path.join(OUTPUT_DIR, "test.jsonl")}"')

if __name__ == "__main__":
    main()


