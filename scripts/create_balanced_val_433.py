#!/usr/bin/env python3
"""
创建按比例4:3:3分配的验证集（math:qa:code = 4:3:3）
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# 设置随机种子
random.seed(42)

# 配置：按比例分配
total_samples = 200  # 总验证集大小
ratios = {
    'math': 0.4,  # 40% -> 80个样本
    'qa': 0.3,    # 30% -> 60个样本
    'code': 0.3   # 30% -> 60个样本
}

# 计算每个类型的样本数
type_samples = {
    'math': int(total_samples * ratios['math']),  # 80
    'qa': int(total_samples * ratios['qa']),      # 60
    'code': int(total_samples * ratios['code'])   # 60
}

# 数据源映射到问题类型
source_to_type = {
    'gsm8k': 'math',
    'mbpp': 'code',
    'humaneval': 'code',
    'commonsenseqa': 'qa',
    'hotpotqa': 'qa',
    'mmlu': 'mixed'  # 特殊处理
}

# 每个类型内的source分配（均匀）
type_to_sources = {
    'math': ['gsm8k'],  # math只有gsm8k
    'code': ['mbpp', 'humaneval'],  # code有mbpp和humaneval，各30
    'qa': ['commonsenseqa', 'hotpotqa']  # qa有两个，各30
}

# 读取数据
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
for source in sorted(data_by_source.keys()):
    count = len(data_by_source[source])
    print(f"  {source}: {count} 样本")

# 创建按比例验证集
balanced_val = []

print(f"\n创建按比例验证集（math:qa:code = 4:3:3）:")
print(f"目标分配: math={type_samples['math']}, qa={type_samples['qa']}, code={type_samples['code']}")

# 1. Math类型（80个样本）
print("\nMath类型:")
math_samples = []
if len(data_by_source['gsm8k']) >= type_samples['math']:
    math_samples = random.sample(data_by_source['gsm8k'], type_samples['math'])
    print(f"  gsm8k: 选择 {type_samples['math']} / {len(data_by_source['gsm8k'])} 个样本")
else:
    math_samples = data_by_source['gsm8k']
    print(f"  ⚠️ gsm8k: 只有 {len(data_by_source['gsm8k'])} 个样本，全部使用")
balanced_val.extend(math_samples)

# 2. Code类型（60个样本，mbpp和humaneval各30）
print("\nCode类型:")
code_samples = []
for source in ['mbpp', 'humaneval']:
    samples_needed = type_samples['code'] // 2  # 30个
    if len(data_by_source[source]) >= samples_needed:
        selected = random.sample(data_by_source[source], samples_needed)
        print(f"  {source}: 选择 {samples_needed} / {len(data_by_source[source])} 个样本")
    else:
        selected = data_by_source[source]
        print(f"  ⚠️ {source}: 只有 {len(data_by_source[source])} 个样本，全部使用")
    code_samples.extend(selected)
balanced_val.extend(code_samples)

# 3. QA类型（60个样本，commonsenseqa和hotpotqa各30）
print("\nQA类型:")
qa_samples = []
for source in ['commonsenseqa', 'hotpotqa']:
    samples_needed = type_samples['qa'] // 2  # 30个
    if len(data_by_source[source]) >= samples_needed:
        selected = random.sample(data_by_source[source], samples_needed)
        print(f"  {source}: 选择 {samples_needed} / {len(data_by_source[source])} 个样本")
    else:
        selected = data_by_source[source]
        print(f"  ⚠️ {source}: 只有 {len(data_by_source[source])} 个样本，全部使用")
    qa_samples.extend(selected)
balanced_val.extend(qa_samples)

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
print("\nProblem Type分布（4:3:3）:")
for ptype, count in sorted(type_counts.items()):
    print(f"  {ptype}: {count} ({count/len(balanced_val)*100:.1f}%)")

print("\nSource分布:")
for source, count in sorted(source_counts.items()):
    print(f"  {source}: {count} ({count/len(balanced_val)*100:.1f}%)")

# 验证比例
actual_ratio = f"{type_counts['math']}:{type_counts['qa']}:{type_counts['code']}"
print(f"\n实际比例 (math:qa:code): {actual_ratio}")
expected_ratio = f"{type_samples['math']}:{type_samples['qa']}:{type_samples['code']}"
print(f"期望比例 (math:qa:code): {expected_ratio}")
