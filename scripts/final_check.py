#!/usr/bin/env python3
"""
数据集终极检查脚本
目的:
1. 严格验证 Train/Test 是否重叠 (基于 Content Hash)
2. 抽样检查数据质量 (Ground Truth 是否合理)
3. 验证数据字段完整性
"""

import json
import hashlib
import random
from typing import Set, Dict

TRAIN_PATH = "11/integrated_aflow_roll/data/ready_to_train/train.jsonl"
TEST_PATH = "11/integrated_aflow_roll/data/ready_to_train/test.jsonl"

def get_content_hash(item: Dict) -> str:
    """计算核心内容的哈希 (忽略 meta 等辅助字段)"""
    content = f"{item['problem'].strip()}|{item['problem_type']}|{str(item['ground_truth']).strip()}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def check_overlap():
    print("🔍 正在检查数据集重叠...")
    
    train_hashes = set()
    train_problems = set()
    
    # 加载训练集
    with open(TRAIN_PATH, 'r') as f:
        for line in f:
            item = json.loads(line)
            h = get_content_hash(item)
            train_hashes.add(h)
            train_problems.add(item['problem'].strip())
            
    print(f"  ✅ 训练集加载完毕: {len(train_hashes)} 个唯一指纹")
    
    # 检查测试集
    overlap_count = 0
    overlap_problems = 0
    total_test = 0
    
    with open(TEST_PATH, 'r') as f:
        for line in f:
            total_test += 1
            item = json.loads(line)
            h = get_content_hash(item)
            
            if h in train_hashes:
                overlap_count += 1
            
            if item['problem'].strip() in train_problems:
                overlap_problems += 1
                
    if overlap_count > 0 or overlap_problems > 0:
        print(f"❌ 警告: 发现重叠!")
        print(f"  完全重复: {overlap_count} 条")
        print(f"  问题重复: {overlap_problems} 条")
    else:
        print(f"✅ 验证通过: 训练集与测试集无任何重叠 (测试集共 {total_test} 条)")

def spot_check(num_samples=5):
    print(f"\n🔍 抽样检查 ({num_samples} 条)...")
    
    with open(TRAIN_PATH, 'r') as f:
        lines = f.readlines()
        
    samples = random.sample(lines, num_samples)
    
    for i, line in enumerate(samples):
        item = json.loads(line)
        print(f"\n[{i+1}] 类型: {item['problem_type']} | 来源: {item.get('source', 'N/A')}")
        print(f"  Q: {item['problem'][:100]}...")
        print(f"  A: {str(item['ground_truth'])[:100]}...")
        
        # 简单启发式检查
        if item['problem_type'] == 'qa' and len(str(item['ground_truth'])) < 2:
            print("  ⚠️  警告: QA 答案过短")
        if item['problem_type'] == 'code' and 'def ' not in str(item['ground_truth']):
            print("  ⚠️  警告: Code 答案似乎不是函数定义")

if __name__ == "__main__":
    check_overlap()
    spot_check()


