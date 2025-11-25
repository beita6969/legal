#!/usr/bin/env python3
"""
验证最终数据集分布
"""

import json
from pathlib import Path
from collections import defaultdict

# 数据目录
data_dir = Path('/home/yijia/.claude/11/integrated_aflow_roll/data')
mixed_dir = data_dir / 'mixed'

def analyze_dataset(filename):
    """分析数据集"""
    file_path = mixed_dir / filename

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return None

    print(f"\n📊 分析 {filename}")
    print("="*60)

    # 统计
    total = 0
    type_counts = defaultdict(int)
    source_counts = defaultdict(int)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                total += 1

                # 类型统计
                ptype = sample.get('problem_type', 'unknown')
                type_counts[ptype] += 1

                # 数据源统计
                source = sample.get('source', 'unknown')
                source_counts[source] += 1

    print(f"总样本数: {total:,}")

    # 按类型分布
    print("\n按类型分布:")
    for ptype in sorted(type_counts.keys()):
        count = type_counts[ptype]
        percentage = count / total * 100 if total > 0 else 0
        print(f"  {ptype:10s}: {count:6,} ({percentage:5.1f}%)")

    # 按数据源分布
    print("\n按数据源分布:")
    for source in sorted(source_counts.keys(), key=lambda x: source_counts[x], reverse=True):
        count = source_counts[source]
        percentage = count / total * 100 if total > 0 else 0
        print(f"  {source:15s}: {count:6,} ({percentage:5.1f}%)")

    return {
        'total': total,
        'type_counts': dict(type_counts),
        'source_counts': dict(source_counts)
    }

def main():
    print("="*60)
    print("🔍 验证最终数据集分布")
    print("="*60)

    # 分析所有相关数据集
    datasets = [
        'train_mixed_with_math.jsonl',  # 最新的包含MATH的数据集
        'train_mixed_balanced.jsonl',   # 之前的平衡数据集
        'train_mixed.jsonl',            # 原始训练集
    ]

    results = {}

    for dataset_name in datasets:
        result = analyze_dataset(dataset_name)
        if result:
            results[dataset_name] = result

    # 比较结果
    if 'train_mixed_with_math.jsonl' in results and 'train_mixed_balanced.jsonl' in results:
        print("\n" + "="*60)
        print("📈 对比分析: train_mixed_with_math vs train_mixed_balanced")
        print("="*60)

        math_data = results['train_mixed_with_math.jsonl']
        balanced_data = results['train_mixed_balanced.jsonl']

        print(f"\n样本数变化: {balanced_data['total']:,} → {math_data['total']:,} (+{math_data['total'] - balanced_data['total']:,})")

        print("\n类型分布变化:")
        all_types = set(math_data['type_counts'].keys()) | set(balanced_data['type_counts'].keys())
        for ptype in sorted(all_types):
            old_count = balanced_data['type_counts'].get(ptype, 0)
            new_count = math_data['type_counts'].get(ptype, 0)
            change = new_count - old_count
            old_pct = old_count / balanced_data['total'] * 100 if balanced_data['total'] > 0 else 0
            new_pct = new_count / math_data['total'] * 100 if math_data['total'] > 0 else 0

            change_str = f"+{change:,}" if change > 0 else f"{change:,}"
            print(f"  {ptype:10s}: {old_pct:5.1f}% → {new_pct:5.1f}% ({change_str} 样本)")

        print("\n数据源变化:")
        # 只显示MATH相关的变化
        if 'MATH' in math_data['source_counts']:
            math_count = math_data['source_counts']['MATH']
            print(f"  MATH: 0 → {math_count:,} (+{math_count:,} 样本)")

    # 推荐使用哪个数据集
    print("\n" + "="*60)
    print("✅ 推荐使用: train_mixed_with_math.jsonl")
    print("   原因: 包含了MATH数据集的高质量数学题目")
    print("="*60)

if __name__ == '__main__':
    main()
