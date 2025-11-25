#!/usr/bin/env python3
"""
高级数据过滤器 (Advanced Data Filter)
目的：清洗 SQuAD 和 HotpotQA 等数据集，剔除那些严重依赖上下文、指代不明或答案过于简单的低质量样本。

过滤规则：
1. [QA] 剔除答案长度 < 2 的样本 (通常是无意义的词或数字，容易产生歧义)
2. [QA] 剔除问题长度 < 5 个单词的样本 (问题太短通常指代不明)
3. [QA] 剔除包含 "this", "that", "these", "those", "the following" 等指代词且无明确名词的问题
4. [QA] 剔除答案与问题重叠度过高的样本 (可能是无效问答)
5. [Code] 剔除无测试用例的样本
"""

import json
import os
import re
from tqdm import tqdm

# 配置
RAW_DIR = "11/integrated_aflow_roll/data/raw"
FILTERED_DIR = "11/integrated_aflow_roll/data/raw_filtered"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def is_bad_qa(problem, answer):
    """判断是否为低质量 QA"""
    problem = problem.strip()
    answer = str(answer).strip()
    
    # 规则 1: 答案极短 (除非是年份)
    if len(answer) < 2 and not answer.isdigit():
        return True, "Answer too short"
        
    # 规则 2: 问题极短
    if len(problem.split()) < 5:
        return True, "Problem too short"
        
    # 规则 3: 明显的指代不明 (Context-dependent)
    # 检查是否以指代词开头，或者包含指向上下文的短语
    context_indicators = [
        r"^what is this", r"^who is he", r"^who is she", r"^what does it",
        r"in the passage", r"according to the text", r"mentioned above",
        r"of the following", r"described here"
    ]
    for pattern in context_indicators:
        if re.search(pattern, problem, re.IGNORECASE):
            return True, "Context dependent phrase"
            
    # 规则 4: 答案即问题 (重复)
    if answer.lower() in problem.lower() and len(answer) > len(problem) * 0.8:
        return True, "Answer is just the problem"

    return False, ""

def filter_file(filename, type_check="qa"):
    input_path = os.path.join(RAW_DIR, filename)
    output_path = os.path.join(FILTERED_DIR, filename)
    
    print(f"\n🔍 正在过滤: {filename}")
    
    if not os.path.exists(input_path):
        print(f"⚠️  文件不存在: {input_path}")
        return

    total = 0
    kept = 0
    dropped = 0
    drop_reasons = {}

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in tqdm(fin):
            total += 1
            try:
                item = json.loads(line)
                problem = item.get("problem", "")
                answer = item.get("ground_truth", "")
                
                should_drop = False
                reason = ""
                
                if type_check == "qa":
                    should_drop, reason = is_bad_qa(problem, answer)
                elif type_check == "code":
                    # Code 检查测试用例
                    if not item.get("test") and not item.get("test_list"): # 兼容不同字段
                         should_drop = True
                         reason = "No test cases"
                
                if should_drop:
                    dropped += 1
                    drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                else:
                    kept += 1
                    fout.write(json.dumps(item) + "\n")
                    
            except json.JSONDecodeError:
                continue

    print(f"  ✅ 保留: {kept} ({kept/total*100:.1f}%)")
    print(f"  🗑️  剔除: {dropped} ({dropped/total*100:.1f}%)")
    if dropped > 0:
        print(f"  📉 剔除原因: {json.dumps(drop_reasons, indent=2)}")

def main():
    ensure_dir(FILTERED_DIR)
    
    # 过滤 QA 数据集 (重点)
    filter_file("squad_v2.jsonl", "qa")
    filter_file("hotpotqa.jsonl", "qa")
    
    # 过滤 Code 数据集
    filter_file("mbpp.jsonl", "code")
    filter_file("humaneval.jsonl", "code")
    
    # Math 数据集通常质量较高，但也复制过去保持一致性
    # GSM8K 和 MATH 主要是 self-contained 的，但也过一遍基本检查
    filter_file("gsm8k.jsonl", "qa") # 用 QA 规则简单检查长度
    filter_file("math.jsonl", "qa")

if __name__ == "__main__":
    main()


