#!/usr/bin/env python3
"""
创建均匀分配的验证集
每个source数据集抽取相同数量的样本
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# 设置随机种子
random.seed(42)

# 配置
samples_per_source = 30  # 每个source抽取30个样本
total_samples = 180  # 6个source * 30 = 180

# 数据源
sources = {
    'gsm8k': 'math',
    'mbpp': 'code',
    'humaneval': 'code',
    'commonsenseqa': 'qa',
    'hotpotqa': 'qa',
    'mmlu': 'mixed'
}

# 读取训练数据
train_file = Path('data/mixed/train_mixed.jsonl')
test_file = Path('data/processed/test_mixed.jsonl')

# 按source分组
data_by_source = defaultdict(list)

# 从训练集读取
print("读取训练集...")
with open(train_file, 'r') as f:
    for line in f:
        if line.strip():
            sample = json.loads(line)
            if 'source' in sample:
                data_by_source[sample['source']].append(sample)

# 从测试集读取humaneval（因为训练集没有）
print("读取测试集(humaneval)...")
with open(test_file, 'r') as f:
    for line in f:
        if line.strip():
            sample = json.loads(line)
            if sample.get('source') == 'humaneval':
                data_by_source['humaneval'].append(sample)

# 统计
print("\n数据源统计:")
for source in sorted(sources.keys()):
    count = len(data_by_source[source])
    print(f"  {source}: {count} 样本")

# 创建均匀验证集
balanced_val = []

print(f"\n创建均匀验证集（每个source {samples_per_source}个样本）:")
for source in sorted(sources.keys()):
    available = data_by_source[source]
    if len(available) < samples_per_source:
        print(f"  ⚠️  {source}: 只有 {len(available)} 个样本，全部使用")
        selected = available
    else:
        selected = random.sample(available, samples_per_source)
        print(f"  ✅ {source}: 随机选择 {samples_per_source} / {len(available)} 个样本")

    balanced_val.extend(selected)

# 打乱顺序
random.shuffle(balanced_val)

# 保存
output_file = Path('data/balanced_val.jsonl')
with open(output_file, 'w') as f:
    for sample in balanced_val:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"\n✅ 保存到: {output_file}")
print(f"   总样本数: {len(balanced_val)}")

# 验证分布
type_counts = defaultdict(int)
source_counts = defaultdict(int)

for sample in balanced_val:
    type_counts[sample['problem_type']] += 1
    source_counts[sample.get('source', 'unknown')] += 1

print("\n验证集分布:")
print("\nProblem Type分布:")
for ptype, count in sorted(type_counts.items()):
    print(f"  {ptype}: {count} ({count/len(balanced_val)*100:.1f}%)")

print("\nSource分布（均匀）:")
for source, count in sorted(source_counts.items()):
    print(f"  {source}: {count} ({count/len(balanced_val)*100:.1f}%)")
