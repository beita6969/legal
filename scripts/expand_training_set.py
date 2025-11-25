#!/usr/bin/env python3
"""
扩展训练集：添加HumanEval和MBPP，扩展code到30%
"""

import json
import random
from pathlib import Path
from collections import defaultdict
import copy

random.seed(42)

# 路径
data_dir = Path('/home/yijia/.claude/11/integrated_aflow_roll/data')
mixed_dir = data_dir / 'mixed'
humaneval_dir = data_dir / 'humaneval'

def load_jsonl(file_path):
    """加载JSONL文件"""
    samples = []
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    return samples

def save_jsonl(samples, file_path):
    """保存JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

def standardize_humaneval(sample):
    """标准化HumanEval样本格式"""
    standardized = {
        'problem': sample.get('prompt', ''),
        'problem_type': 'code',
        'source': 'humaneval',
        'ground_truth': sample.get('canonical_solution', ''),
    }

    # 保留额外字段
    if 'entry_point' in sample:
        standardized['entry_point'] = sample['entry_point']
    if 'test' in sample:
        standardized['test'] = sample['test']
    if 'task_id' in sample:
        standardized['task_id'] = sample['task_id']

    return standardized

def main():
    print("="*60)
    print("📊 扩展训练集：Code样本到30%")
    print("="*60)

    # 1. 加载现有训练集
    print("\n1. 加载现有训练集...")
    train_samples = load_jsonl(mixed_dir / 'train_mixed.jsonl')
    print(f"   原始训练集: {len(train_samples)} 样本")

    # 统计现有分布
    type_counts = defaultdict(int)
    source_counts = defaultdict(int)
    existing_code_samples = []
    non_code_samples = []

    for sample in train_samples:
        ptype = sample.get('problem_type', 'unknown')
        source = sample.get('source', 'unknown')
        type_counts[ptype] += 1
        source_counts[source] += 1

        if ptype == 'code':
            existing_code_samples.append(sample)
        else:
            non_code_samples.append(sample)

    print(f"\n   现有分布:")
    for ptype, count in sorted(type_counts.items()):
        print(f"     {ptype}: {count} ({count/len(train_samples)*100:.1f}%)")

    # 2. 加载HumanEval数据
    print("\n2. 加载HumanEval数据...")
    humaneval_samples = []
    humaneval_files = [
        ('humaneval_full.jsonl', 164),
        ('humaneval_validate.jsonl', 132),
    ]

    for filename, expected_count in humaneval_files:
        file_path = humaneval_dir / filename
        samples = load_jsonl(file_path)
        print(f"   {filename}: {len(samples)} 样本")

        for sample in samples:
            humaneval_samples.append(standardize_humaneval(sample))

    print(f"   HumanEval总计: {len(humaneval_samples)} 样本")

    # 3. 收集所有code样本
    print("\n3. 整合Code样本...")
    all_code_samples = existing_code_samples + humaneval_samples
    print(f"   现有MBPP: {len([s for s in existing_code_samples if s.get('source') == 'mbpp'])} 样本")
    print(f"   新增HumanEval: {len(humaneval_samples)} 样本")
    print(f"   Code样本总计: {len(all_code_samples)} 样本")

    # 去重
    unique_code_samples = []
    seen = set()
    for sample in all_code_samples:
        # 使用problem作为去重键
        key = sample.get('problem', '')[:100]  # 前100字符
        if key and key not in seen:
            unique_code_samples.append(sample)
            seen.add(key)

    print(f"   去重后: {len(unique_code_samples)} 样本")

    # 4. 计算目标数量（30% code）
    print("\n4. 计算目标分布...")
    # 保持non-code样本不变，扩展code到30%
    # 如果code占30%，non-code占70%
    # total = non_code / 0.7
    non_code_count = len(non_code_samples)
    target_total = int(non_code_count / 0.7)
    target_code_count = target_total - non_code_count

    print(f"   Non-code样本: {non_code_count}")
    print(f"   目标总数: {target_total}")
    print(f"   目标Code数: {target_code_count} (30%)")

    # 5. 扩展code样本
    print("\n5. 扩展Code样本...")
    expanded_code_samples = []

    if len(unique_code_samples) >= target_code_count:
        # 如果样本充足，随机选择
        expanded_code_samples = random.sample(unique_code_samples, target_code_count)
        print(f"   随机选择 {target_code_count} 个样本")
    else:
        # 需要复制
        duplication_factor = (target_code_count // len(unique_code_samples)) + 1
        print(f"   需要复制 {duplication_factor} 倍")

        for i in range(duplication_factor):
            for sample in unique_code_samples:
                sample_copy = copy.deepcopy(sample)
                sample_copy['duplication_id'] = i
                expanded_code_samples.append(sample_copy)

        # 随机打乱并截取到目标数量
        random.shuffle(expanded_code_samples)
        expanded_code_samples = expanded_code_samples[:target_code_count]
        print(f"   扩展后: {len(expanded_code_samples)} 样本")

    # 6. 合并数据集
    print("\n6. 创建新训练集...")
    new_train_samples = non_code_samples + expanded_code_samples
    random.shuffle(new_train_samples)

    # 7. 统计最终分布
    final_type_counts = defaultdict(int)
    final_source_counts = defaultdict(int)

    for sample in new_train_samples:
        final_type_counts[sample.get('problem_type', 'unknown')] += 1
        final_source_counts[sample.get('source', 'unknown')] += 1

    print(f"\n   最终分布:")
    print(f"   总样本数: {len(new_train_samples)}")
    print(f"\n   按类型:")
    for ptype, count in sorted(final_type_counts.items()):
        percentage = count / len(new_train_samples) * 100
        print(f"     {ptype}: {count:,} ({percentage:.1f}%)")

    print(f"\n   按数据源:")
    for source, count in sorted(final_source_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(new_train_samples) * 100
        print(f"     {source}: {count:,} ({percentage:.1f}%)")

    # 8. 保存新训练集
    output_file = mixed_dir / 'train_mixed_balanced.jsonl'
    save_jsonl(new_train_samples, output_file)
    print(f"\n✅ 新训练集已保存到: {output_file}")
    print(f"   样本数: {len(new_train_samples)}")

    # 9. 创建统计文件
    stats = {
        'total_samples': len(new_train_samples),
        'type_distribution': dict(final_type_counts),
        'source_distribution': dict(final_source_counts),
        'original_code_samples': len(unique_code_samples),
        'expanded_code_samples': len(expanded_code_samples),
    }

    stats_file = mixed_dir / 'train_stats_balanced.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n📊 统计信息已保存到: {stats_file}")
    print("="*60)
    print("✅ 完成！")
    print("="*60)

if __name__ == '__main__':
    main()
