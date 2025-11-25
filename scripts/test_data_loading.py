#!/usr/bin/env python3
"""
测试数据管理器是否正确加载新的数据集
"""

import sys
import os
sys.path.append('/home/yijia/.claude/11/integrated_aflow_roll')

from src.data_manager import DataManager
from collections import defaultdict

def test_data_loading():
    """测试数据集加载"""
    print("="*60)
    print("🔍 测试数据管理器")
    print("="*60)

    # 创建数据管理器
    data_manager = DataManager(data_dir="data")

    # 加载训练数据
    print("\n📂 加载训练数据...")
    train_data = data_manager.load_data("train")

    # 统计
    total_samples = 0
    type_counts = defaultdict(int)
    source_counts = defaultdict(int)

    for problem_type, samples in train_data.items():
        type_counts[problem_type] = len(samples)
        total_samples += len(samples)

        # 统计source
        for sample in samples:
            source = sample.get('source', 'unknown')
            source_counts[source] += 1

    print(f"\n✅ 总样本数: {total_samples:,}")

    print("\n📊 按类型分布:")
    for ptype, count in sorted(type_counts.items()):
        percentage = count / total_samples * 100 if total_samples > 0 else 0
        print(f"  {ptype:10s}: {count:6,} ({percentage:5.1f}%)")

    print("\n📊 按数据源分布:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_samples * 100 if total_samples > 0 else 0
        print(f"  {source:15s}: {count:6,} ({percentage:5.1f}%)")

    # 检查是否包含MATH数据
    if 'MATH' in source_counts:
        print(f"\n✅ 成功加载MATH数据集: {source_counts['MATH']:,} 样本")
    else:
        print("\n⚠️  警告: 未找到MATH数据集")

    # 加载测试数据
    print("\n" + "="*60)
    print("📂 加载测试数据...")
    test_data = data_manager.load_data("test")

    test_total = 0
    test_source_counts = defaultdict(int)

    for problem_type, samples in test_data.items():
        test_total += len(samples)
        for sample in samples:
            source = sample.get('source', 'unknown')
            test_source_counts[source] += 1

    print(f"\n✅ 测试集总样本数: {test_total:,}")
    print("\n📊 测试集数据源分布:")
    for source, count in sorted(test_source_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / test_total * 100 if test_total > 0 else 0
        print(f"  {source:15s}: {count:6,} ({percentage:5.1f}%)")

    print("\n" + "="*60)
    print("✅ 测试完成")
    print("="*60)

if __name__ == '__main__':
    test_data_loading()
