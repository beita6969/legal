#!/usr/bin/env python3
"""
使用GitHub镜像下载并处理MATH数据集
"""

import json
import os
import subprocess
from pathlib import Path
import tarfile
import random
from collections import defaultdict

random.seed(42)

# 创建目录
data_dir = Path('/home/yijia/.claude/11/integrated_aflow_roll/data')
math_dir = data_dir / 'math_dataset'
math_dir.mkdir(exist_ok=True)

print("="*60)
print("📥 下载MATH数据集 (使用GitHub镜像)")
print("="*60)

# GitHub镜像列表
github_mirrors = [
    'https://ghproxy.com/',
    'https://mirror.ghproxy.com/',
    'https://gh.api.99988866.xyz/',
    'https://github.moeyy.xyz/',
]

# 原始GitHub URL
original_url = 'https://github.com/hendrycks/math/archive/refs/heads/main.zip'

download_success = False

# 尝试不同的镜像
for mirror in github_mirrors:
    mirror_url = mirror + original_url
    print(f"\n尝试从镜像下载: {mirror_url[:50]}...")

    try:
        result = subprocess.run(
            ["wget", "-q", "--timeout=30", "-O", "/tmp/math_dataset.zip", mirror_url],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print("✅ 下载成功")
            download_success = True
            break
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        continue

# 如果镜像失败，尝试直接下载
if not download_success:
    print("\n尝试直接从GitHub下载...")
    try:
        result = subprocess.run(
            ["wget", "-q", "--timeout=60", "-O", "/tmp/math_dataset.zip", original_url],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            print("✅ 下载成功")
            download_success = True
    except Exception as e:
        print(f"❌ 直接下载失败: {e}")

if download_success:
    print("\n正在解压...")
    subprocess.run(
        ["unzip", "-q", "-o", "/tmp/math_dataset.zip", "-d", "/tmp/"],
        check=True
    )

    # 移动到目标目录
    subprocess.run(
        ["mv", "/tmp/math-main", str(math_dir / "raw")],
        check=False
    )

    print("✅ 解压完成")

    # 处理数据集
    print("\n处理MATH数据集...")

    raw_dir = math_dir / "raw" / "MATH"
    if not raw_dir.exists():
        raw_dir = math_dir / "raw"

    # 获取所有主题
    subjects = []
    if (raw_dir / "train").exists():
        subjects = [d.name for d in (raw_dir / "train").iterdir() if d.is_dir()]
    elif (raw_dir / "MATH" / "train").exists():
        raw_dir = raw_dir / "MATH"
        subjects = [d.name for d in (raw_dir / "train").iterdir() if d.is_dir()]

    print(f"\n找到 {len(subjects)} 个数学主题:")
    for subject in subjects[:5]:
        print(f"  - {subject}")
    if len(subjects) > 5:
        print(f"  ... 以及其他 {len(subjects)-5} 个主题")

    # 收集所有样本
    all_samples = []
    difficulty_stats = defaultdict(int)
    subject_stats = defaultdict(int)

    for split in ['train', 'test']:
        split_dir = raw_dir / split
        if not split_dir.exists():
            print(f"⚠️ {split} 目录不存在")
            continue

        print(f"\n处理 {split} 数据...")

        for subject_dir in split_dir.iterdir():
            if not subject_dir.is_dir():
                continue

            subject = subject_dir.name
            subject_stats[subject] += 1

            # 遍历所有题目
            for problem_file in subject_dir.glob("*.json"):
                try:
                    with open(problem_file, 'r', encoding='utf-8') as f:
                        problem_data = json.load(f)

                    # 标准化格式
                    sample = {
                        'problem': problem_data.get('problem', ''),
                        'solution': problem_data.get('solution', ''),
                        'answer': problem_data.get('answer', ''),
                        'subject': subject,
                        'level': problem_data.get('level', 'Level 3'),  # 默认Level 3
                        'problem_type': 'math',
                        'source': 'MATH',
                        'ground_truth': problem_data.get('solution', ''),
                        'split': split
                    }

                    # 提取难度级别
                    level = sample['level']
                    if isinstance(level, str) and 'Level' in level:
                        level_num = level.replace('Level', '').strip()
                        difficulty_stats[f"Level {level_num}"] += 1
                    else:
                        difficulty_stats[str(level)] += 1

                    all_samples.append(sample)

                except Exception as e:
                    print(f"  ⚠️ 处理 {problem_file.name} 失败: {e}")
                    continue

        print(f"  处理了 {len([s for s in all_samples if s['split'] == split])} 个样本")

    print(f"\n总共收集了 {len(all_samples)} 个样本")

    # 难度分布
    print("\n难度级别分布:")
    for level, count in sorted(difficulty_stats.items()):
        print(f"  {level}: {count} 样本")

    # 主题分布
    print("\n主题分布:")
    for subject, count in sorted(subject_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {subject}: {count} 样本")

    # 打乱并划分数据集
    random.shuffle(all_samples)

    # 按比例选择样本（选择中等难度的）
    # 优先选择 Level 3-5 的题目
    selected_samples = []
    for level in ['Level 3', 'Level 4', 'Level 5']:
        level_samples = [s for s in all_samples if level in s.get('level', '')]
        selected_samples.extend(level_samples)

    # 如果不够，添加其他级别
    if len(selected_samples) < 3000:
        other_samples = [s for s in all_samples if s not in selected_samples]
        random.shuffle(other_samples)
        selected_samples.extend(other_samples[:3000 - len(selected_samples)])

    # 限制为3000个样本（避免数据集过大）
    selected_samples = selected_samples[:3000]
    random.shuffle(selected_samples)

    # 90%训练，10%测试
    split_point = int(len(selected_samples) * 0.9)
    train_samples = selected_samples[:split_point]
    test_samples = selected_samples[split_point:]

    print(f"\n最终数据集:")
    print(f"  训练集: {len(train_samples)} 样本")
    print(f"  测试集: {len(test_samples)} 样本")

    # 保存数据集
    train_file = math_dir / 'train.jsonl'
    test_file = math_dir / 'test.jsonl'

    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            # 移除split字段
            sample.pop('split', None)
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    with open(test_file, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            sample.pop('split', None)
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n✅ 数据已保存:")
    print(f"  训练集: {train_file}")
    print(f"  测试集: {test_file}")

    # 保存统计信息
    stats = {
        'total_samples': len(all_samples),
        'selected_samples': len(selected_samples),
        'train_samples': len(train_samples),
        'test_samples': len(test_samples),
        'difficulty_distribution': dict(difficulty_stats),
        'subject_distribution': dict(subject_stats)
    }

    stats_file = math_dir / 'dataset_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n📊 统计信息已保存: {stats_file}")

else:
    print("\n❌ 所有下载尝试都失败了")

print("\n" + "="*60)
print("✅ 处理完成")
print("="*60)
