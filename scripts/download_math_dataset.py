#!/usr/bin/env python3
"""
下载并处理MATH数据集
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
import random

random.seed(42)

# 创建目录
data_dir = Path('/home/yijia/.claude/11/integrated_aflow_roll/data')
math_dir = data_dir / 'math_dataset'
math_dir.mkdir(exist_ok=True)

print("="*60)
print("📥 下载MATH数据集")
print("="*60)

try:
    # 尝试从HuggingFace下载MATH数据集
    print("\n正在从HuggingFace下载MATH数据集...")

    # 尝试不同的数据集名称
    dataset_names = [
        'lighteval/MATH',
        'hendrycks/competition_math',
        'competition_math'
    ]

    dataset = None
    for name in dataset_names:
        try:
            print(f"\n尝试加载: {name}")
            dataset = load_dataset(name, split='train')
            print(f"✅ 成功加载数据集: {name}")
            break
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            continue

    if dataset is None:
        print("\n尝试使用其他方法下载...")
        dataset = load_dataset('lighteval/MATH', split='train', trust_remote_code=True)

    print(f"\n✅ 下载完成，共 {len(dataset)} 个样本")

    # 分析数据结构
    print("\n分析数据结构...")
    sample = dataset[0]
    print("\n样本字段:")
    for key in sample.keys():
        value = sample[key]
        if isinstance(value, str):
            print(f"  {key}: {value[:100]}..." if len(value) > 100 else f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    # 转换为JSONL格式
    print("\n转换为JSONL格式...")

    train_samples = []
    test_samples = []

    # 按难度级别分类
    difficulty_stats = {}

    for i, item in enumerate(dataset):
        # 标准化格式
        sample = {
            'problem': item.get('problem', ''),
            'solution': item.get('solution', ''),
            'answer': item.get('answer', ''),
            'subject': item.get('type', 'unknown'),
            'level': item.get('level', 'unknown'),
            'problem_type': 'math',
            'source': 'MATH',
            'ground_truth': item.get('solution', item.get('answer', ''))
        }

        # 统计难度级别
        level = sample['level']
        if level not in difficulty_stats:
            difficulty_stats[level] = 0
        difficulty_stats[level] += 1

        # 90%训练，10%测试
        if i % 10 == 0:
            test_samples.append(sample)
        else:
            train_samples.append(sample)

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_samples)} 样本")
    print(f"  测试集: {len(test_samples)} 样本")

    print("\n难度级别分布:")
    for level, count in sorted(difficulty_stats.items()):
        print(f"  Level {level}: {count} 样本")

    # 保存数据集
    train_file = math_dir / 'train.jsonl'
    test_file = math_dir / 'test.jsonl'

    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    with open(test_file, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n✅ 数据已保存:")
    print(f"  训练集: {train_file}")
    print(f"  测试集: {test_file}")

    # 保存统计信息
    stats = {
        'total_samples': len(dataset),
        'train_samples': len(train_samples),
        'test_samples': len(test_samples),
        'difficulty_levels': difficulty_stats,
        'subjects': list(set([s.get('subject', 'unknown') for s in train_samples]))
    }

    stats_file = math_dir / 'dataset_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n📊 统计信息已保存: {stats_file}")

except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("\n尝试直接下载MATH数据集...")

    # 备用方案：使用wget下载
    import subprocess

    urls = [
        'https://github.com/hendrycks/math/raw/main/MATH.tar',
        'https://people.eecs.berkeley.edu/~hendrycks/MATH.tar'
    ]

    for url in urls:
        try:
            print(f"\n尝试从 {url} 下载...")
            result = subprocess.run(
                ["wget", "-q", "-O", "/tmp/MATH.tar", url],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print("✅ 下载成功")
                print("正在解压...")

                subprocess.run(
                    ["tar", "-xf", "/tmp/MATH.tar", "-C", str(math_dir)],
                    check=True
                )

                print("✅ 解压完成")
                break
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            continue

print("\n" + "="*60)
print("✅ 处理完成")
print("="*60)
