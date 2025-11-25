#!/usr/bin/env python3
"""
数据集详细检查脚本
"""
import json
import random
from collections import Counter, defaultdict

def check_dataset(file_path: str, name: str, num_samples: int = 10):
    print("="*70)
    print(f"{name}检查: {file_path}")
    print("="*70)

    with open(file_path) as f:
        samples = [json.loads(l) for l in f]

    print(f"\n总样本数: {len(samples)}")

    # 1. 类型分布
    types = Counter(s.get('problem_type') for s in samples)
    print("\n类型分布:")
    for t, c in types.most_common():
        print(f"  {t}: {c} ({c/len(samples)*100:.1f}%)")

    # 2. 字段完整性详细检查
    print("\n字段完整性检查:")
    issues = defaultdict(list)

    for i, s in enumerate(samples):
        # 基础字段
        if not s.get('problem'):
            issues['problem'].append(i)
        if not s.get('problem_type'):
            issues['problem_type'].append(i)
        if not s.get('ground_truth'):
            issues['ground_truth'].append(i)

        # Code特定字段
        if s.get('problem_type') == 'code':
            if not s.get('entry_point'):
                issues['code_entry_point'].append(i)
            if not s.get('test'):
                issues['code_test'].append(i)

    if issues:
        print("  发现问题:")
        for field, indices in issues.items():
            print(f"    {field}: {len(indices)}个样本缺失")
            if len(indices) <= 5:
                print(f"      样本索引: {indices}")
    else:
        print("  ✅ 所有字段完整")

    # 3. 随机抽样详细检查
    print(f"\n随机抽样检查 ({num_samples}个样本):")
    random.seed(42)
    sample_indices = random.sample(range(len(samples)), min(num_samples, len(samples)))

    for i, idx in enumerate(sample_indices[:3], 1):  # 只显示前3个
        s = samples[idx]
        ptype = s.get('problem_type', 'unknown')
        print(f"\n  样本{i} (索引{idx}, 类型:{ptype}):")
        print(f"    problem长度: {len(s.get('problem', ''))}")
        print(f"    ground_truth长度: {len(s.get('ground_truth', ''))}")

        if ptype == 'code':
            print(f"    entry_point: {s.get('entry_point', 'N/A')}")
            print(f"    test长度: {len(s.get('test', ''))}")
            print(f"    task_id: {s.get('task_id', 'N/A')}")
            # 检查problem格式
            problem = s.get('problem', '')
            if problem and 'def ' in problem:
                print(f"    ✅ problem包含函数定义")
            else:
                print(f"    ⚠️ problem可能缺少函数定义")

        elif ptype == 'math':
            if 'category' in s:
                print(f"    category: {s.get('category', 'N/A')}")
            if 'difficulty' in s:
                print(f"    difficulty: {s.get('difficulty', 'N/A')}")
            # 检查答案格式
            gt = s.get('ground_truth', '')
            if '\\boxed' in gt or 'boxed' in gt:
                print(f"    ✅ ground_truth包含boxed格式")

        elif ptype == 'qa':
            if 'type' in s:
                print(f"    type: {s.get('type', 'N/A')}")
            if 'context' in s:
                print(f"    context长度: {len(s.get('context', ''))}")
            if 'passage' in s:
                print(f"    passage长度: {len(s.get('passage', ''))}")

    # 4. Code样本特殊检查
    code_samples = [s for s in samples if s.get('problem_type') == 'code']
    if code_samples:
        print(f"\nCode样本详细检查 ({len(code_samples)}个):")

        # 检查entry_point分布
        entry_points = Counter(s.get('entry_point') for s in code_samples)
        print(f"  唯一entry_point数: {len(entry_points)}")
        print(f"  前5个最常见:")
        for ep, count in entry_points.most_common(5):
            print(f"    {ep}: {count}次")

        # 检查task_id分布
        task_ids = Counter(s.get('task_id', '')[:13] for s in code_samples if s.get('task_id'))
        print(f"\n  Task ID前缀分布:")
        for tid, count in task_ids.most_common():
            print(f"    {tid}: {count}个")

    # 5. 重复样本检查
    print("\n重复样本检查:")
    problem_hashes = [hash(s.get('problem', '')) for s in samples]
    unique_problems = len(set(problem_hashes))
    duplicate_count = len(samples) - unique_problems
    print(f"  唯一问题: {unique_problems}")
    print(f"  重复问题: {duplicate_count}")
    if duplicate_count > 0:
        print(f"  重复率: {duplicate_count/len(samples)*100:.1f}%")

    return len(issues) == 0

if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# 数据集详细检查")
    print("#"*70 + "\n")

    train_ok = check_dataset(
        "data/train/balanced_10k_dataset.jsonl",
        "训练集",
        num_samples=10
    )

    print("\n\n")

    test_ok = check_dataset(
        "data/test/balanced_test_dataset.jsonl",
        "测试集",
        num_samples=10
    )

    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print(f"\n训练集: {'✅ 通过' if train_ok else '❌ 有问题'}")
    print(f"测试集: {'✅ 通过' if test_ok else '❌ 有问题'}")

    if train_ok and test_ok:
        print("\n✅ 所有检查通过，数据集可以安全替换")
    else:
        print("\n⚠️ 发现问题，需要修复后再替换")
