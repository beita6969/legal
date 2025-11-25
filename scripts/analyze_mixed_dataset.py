#!/usr/bin/env python3
"""
分析mixed数据集的详细分布
"""

import json
from collections import defaultdict
from pathlib import Path

data_dir = Path('/home/yijia/.claude/11/integrated_aflow_roll/data/mixed')

def analyze_dataset(file_path, dataset_name):
    """分析单个数据集"""
    type_counts = defaultdict(int)
    source_counts = defaultdict(int)
    source_type_counts = defaultdict(lambda: defaultdict(int))

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                problem_type = sample.get('problem_type', 'unknown')
                source = sample.get('source', 'unknown')

                type_counts[problem_type] += 1
                source_counts[source] += 1
                source_type_counts[source][problem_type] += 1

    total = sum(type_counts.values())

    print(f"\n{'='*60}")
    print(f"📊 {dataset_name} 数据集分析")
    print(f"{'='*60}")
    print(f"\n总样本数: {total:,}\n")

    print("按问题类型分布:")
    print("-" * 40)
    for ptype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total * 100
        print(f"  {ptype:10} : {count:6,} ({percentage:6.2f}%)")

    print("\n按数据源分布:")
    print("-" * 40)
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total * 100
        print(f"  {source:15} : {count:6,} ({percentage:6.2f}%)")

    print("\n详细分布 (数据源 × 问题类型):")
    print("-" * 40)
    for source in sorted(source_counts.keys()):
        print(f"\n  {source}:")
        for ptype, count in sorted(source_type_counts[source].items()):
            percentage = count / total * 100
            print(f"    {ptype:10} : {count:6,} ({percentage:6.2f}%)")

    return {
        'total': total,
        'type_counts': dict(type_counts),
        'source_counts': dict(source_counts),
        'source_type_counts': {s: dict(t) for s, t in source_type_counts.items()}
    }

# 主程序
print("="*60)
print("🎯 Mixed 数据集完整分析报告")
print("="*60)

# 分析训练集
train_stats = analyze_dataset(data_dir / 'train_mixed.jsonl', '训练集 (train_mixed.jsonl)')

# 分析测试集（原验证集）
test_stats = analyze_dataset(data_dir / 'test_mixed.jsonl', '测试集 (test_mixed.jsonl，原val_mixed.jsonl)')

# 总结
print("\n" + "="*60)
print("📈 数据集总结")
print("="*60)

total_samples = train_stats['total'] + test_stats['total']
print(f"\n总样本数: {total_samples:,}")
print(f"  - 训练集: {train_stats['total']:,} ({train_stats['total']/total_samples*100:.1f}%)")
print(f"  - 测试集: {test_stats['total']:,} ({test_stats['total']/total_samples*100:.1f}%)")

# 计算比例
print("\n问题类型比例对比:")
print("-" * 40)
print(f"{'类型':10} {'训练集':>15} {'测试集':>15}")
print("-" * 40)

for ptype in ['math', 'code', 'qa', 'mixed']:
    train_count = train_stats['type_counts'].get(ptype, 0)
    test_count = test_stats['type_counts'].get(ptype, 0)
    train_pct = train_count / train_stats['total'] * 100 if train_stats['total'] > 0 else 0
    test_pct = test_count / test_stats['total'] * 100 if test_stats['total'] > 0 else 0
    print(f"{ptype:10} {train_count:6,} ({train_pct:5.1f}%) {test_count:6,} ({test_pct:5.1f}%)")

print("\n" + "="*60)
print("✅ 分析完成！")
print("\n💡 使用建议:")
print("  1. 训练集: data/mixed/train_mixed.jsonl")
print("  2. 测试集: data/mixed/test_mixed.jsonl（已从val_mixed.jsonl复制）")
print("  3. 训练集偏重QA和Mixed类型，Code样本较少")
print("  4. 测试集中Code样本比例较高，可以很好地评估Code能力")
print("="*60)
