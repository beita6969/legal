#!/usr/bin/env python3
"""
AFlow + ROLL 训练数据集混合器 (Final Mix)

目标:
1. 生成 2000 条训练数据，100 条测试数据。
2. 比例控制: Math (40%) : QA (30%) : Code (30%)
3. 难度分级: Easy / Medium / Hard
4. 数据去重: 训练集与测试集严格互斥

数据源:
- Math: GSM8K (Easy), MATH (Hard)
- QA: SQuAD 2.0 (Easy/Medium), HotpotQA (Medium/Hard)
- Code: MBPP (Easy/Medium), HumanEval (Medium/Hard)
"""

import json
import random
import os
from pathlib import Path
from tqdm import tqdm

# 配置
RAW_DIR = "11/integrated_aflow_roll/data/raw_filtered"  # 使用过滤后的数据源
OUTPUT_DIR = "11/integrated_aflow_roll/data/final_mix"
TRAIN_SIZE = 2000
TEST_SIZE = 100

# 比例
RATIOS = {
    "math": 0.4,
    "qa": 0.3,
    "code": 0.3
}

# 数据集映射与难度预估
# 注意: HumanEval 只有 164 条，MBPP 只有 974 条 (原始样本)，可能需要重复采样或全部利用
DATASETS = {
    "math": {
        "easy": ["gsm8k.jsonl"],
        "hard": ["math.jsonl"]
    },
    "qa": {
        "easy": ["squad_v2.jsonl"],
        "hard": ["hotpotqa.jsonl"]
    },
    "code": {
        "easy": ["mbpp.jsonl"],
        "hard": ["humaneval.jsonl"]
    }
}

def load_jsonl(filename):
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        print(f"⚠️  警告: 文件不存在 {path}")
        return []
    
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    pass
    return data

def clean_code_problem(problem_text):
    """清洗代码问题描述"""
    # HumanEval 的 problem 通常是函数头，无需清洗
    # MBPP 的 problem 是自然语言描述，有时带有多余空格
    if not problem_text:
        return ""
    return problem_text.strip()

def create_stratified_split():
    print("🚀 开始构建混合数据集...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载所有数据
    pools = {
        "math": {"easy": [], "hard": []},
        "qa": {"easy": [], "hard": []},
        "code": {"easy": [], "hard": []}
    }
    
    # 加载 Math
    pools["math"]["easy"] = load_jsonl("gsm8k.jsonl")
    pools["math"]["hard"] = load_jsonl("math.jsonl")
    
    # 加载 QA
    pools["qa"]["easy"] = load_jsonl("squad_v2.jsonl")
    pools["qa"]["hard"] = load_jsonl("hotpotqa.jsonl")
    
    # 加载 Code
    pools["code"]["easy"] = load_jsonl("mbpp.jsonl")
    pools["code"]["hard"] = load_jsonl("humaneval.jsonl")
    
    # 打印统计
    print("\n📊 原始数据统计:")
    for domain in pools:
        print(f"  {domain.upper()}: Easy={len(pools[domain]['easy'])}, Hard={len(pools[domain]['hard'])}")

    # 2. 计算目标数量
    train_counts = {k: int(TRAIN_SIZE * v) for k, v in RATIOS.items()}
    test_counts = {k: int(TEST_SIZE * v) for k, v in RATIOS.items()}
    
    # 修正总数误差
    train_counts["math"] += TRAIN_SIZE - sum(train_counts.values())
    test_counts["math"] += TEST_SIZE - sum(test_counts.values())
    
    print("\n🎯 目标采样数量 (Train / Test):")
    for k in RATIOS:
        print(f"  {k.upper()}: {train_counts[k]} / {test_counts[k]}")

    # 3. 采样逻辑
    final_train = []
    final_test = []
    
    for domain in ["math", "qa", "code"]:
        # 混合 Easy 和 Hard
        # 策略: 50% Easy, 50% Hard (如果够的话)
        # 对于 Code，HumanEval (Hard) 只有 164 条，必须全取或重复
        
        all_items = []
        
        # 给每个 item 打上难度标签（可选，用于分析）
        for item in pools[domain]["easy"]:
            item["difficulty"] = "easy"
            item["domain"] = domain
            all_items.append(item)
            
        for item in pools[domain]["hard"]:
            item["difficulty"] = "hard"
            item["domain"] = domain
            all_items.append(item)
            
        # 打乱
        random.shuffle(all_items)
        
        # 需要的总数
        needed_train = train_counts[domain]
        needed_test = test_counts[domain]
        total_needed = needed_train + needed_test
        
        # 检查是否足够
        if len(all_items) < total_needed:
            print(f"⚠️  {domain} 数据不足 ({len(all_items)} < {total_needed})，执行过采样...")
            # 过采样: 简单重复
            factor = total_needed // len(all_items) + 1
            all_items = all_items * factor
            random.shuffle(all_items)
            
        # 切分
        # 确保测试集不包含训练集数据 (在过采样前已打乱，且通常取不重复的切片即可)
        # 由于我们先 shuffle 再切片，只要源数据不重复，切片就不重复
        # 但如果发生了过采样，可能会有重复。
        # 严格做法: 先取 Test (不重复)，剩下的做 Train (可重复)
        
        # 重置为无重复列表用于 Test
        unique_items = pools[domain]["easy"] + pools[domain]["hard"]
        random.shuffle(unique_items)
        
        # 1. 抽取 Test (绝对不重复)
        if len(unique_items) < needed_test:
             print(f"❌ 严重错误: {domain} 唯一样本数少于测试集需求!")
             return
             
        domain_test = unique_items[:needed_test]
        remaining = unique_items[needed_test:]
        
        # 2. 抽取 Train (不够则重复)
        domain_train = []
        if len(remaining) >= needed_train:
            domain_train = remaining[:needed_train]
        else:
            # 需要重复采样
            while len(domain_train) < needed_train:
                k = min(needed_train - len(domain_train), len(remaining))
                domain_train.extend(remaining[:k])
                # 如果还不够，再次打乱remaining并继续
                if len(domain_train) < needed_train:
                    random.shuffle(remaining)
        
        final_train.extend(domain_train)
        final_test.extend(domain_test)
        
    # 4. 最终打乱与保存
    random.shuffle(final_train)
    random.shuffle(final_test)
    
    def save_dataset(data, name):
        path = os.path.join(OUTPUT_DIR, name)
        print(f"💾 保存 {name}: {len(data)} 条")
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
                
    save_dataset(final_train, "train_2k.jsonl")
    save_dataset(final_test, "test_100.jsonl")
    
    # 5. 质量报告
    print("\n📈 数据集分布报告:")
    for name, ds in [("Train", final_train), ("Test", final_test)]:
        stats = {"math": 0, "qa": 0, "code": 0}
        diffs = {"easy": 0, "hard": 0}
        for item in ds:
            stats[item["domain"]] += 1
            diffs[item.get("difficulty", "unknown")] += 1
        print(f"  {name}: {stats} | 难度: {diffs}")

if __name__ == "__main__":
    create_stratified_split()

