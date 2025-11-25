#!/usr/bin/env python3
"""
完整数据集验证脚本
"""
import json
from collections import Counter
from pathlib import Path

def check_dataset(file_path: str, dataset_name: str):
    """检查单个数据集"""
    print("="*70)
    print(f"{dataset_name}检查: {file_path}")
    print("="*70)

    if not Path(file_path).exists():
        print(f"❌ 文件不存在")
        return None

    # 加载数据
    try:
        with open(file_path) as f:
            samples = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None

    print(f"\n✅ 总样本数: {len(samples)}")

    # 类型分布
    types = Counter(s.get('problem_type', 'unknown') for s in samples)
    print("\n📊 类型分布:")
    for ptype, count in types.most_common():
        print(f"  {ptype:10s}: {count:7d} ({count/len(samples)*100:5.2f}%)")

    # 字段完整性检查
    print("\n🔍 字段完整性检查:")
    missing_counts = {
        'problem': 0,
        'problem_type': 0,
        'ground_truth': 0
    }

    code_missing = {
        'entry_point': 0,
        'test': 0
    }

    code_count = 0
    for s in samples:
        if 'problem' not in s or not s['problem']:
            missing_counts['problem'] += 1
        if 'problem_type' not in s:
            missing_counts['problem_type'] += 1
        if 'ground_truth' not in s:
            missing_counts['ground_truth'] += 1

        if s.get('problem_type') == 'code':
            code_count += 1
            if 'entry_point' not in s or not s['entry_point']:
                code_missing['entry_point'] += 1
            if 'test' not in s or not s['test']:
                code_missing['test'] += 1

    all_good = True
    for field, count in missing_counts.items():
        if count > 0:
            print(f"  ⚠️  {field}: {count}个样本缺失")
            all_good = False

    if code_count > 0:
        print(f"\n🔍 Code样本字段检查 ({code_count}个):")
        for field, count in code_missing.items():
            if count > 0:
                print(f"  ⚠️  {field}: {count}个样本缺失")
                all_good = False
            else:
                print(f"  ✅ {field}: 完整")

    if all_good:
        print("  ✅ 所有必需字段完整")

    # Code样本示例
    code_samples = [s for s in samples if s.get('problem_type') == 'code']
    if len(code_samples) > 0:
        print(f"\n📝 Code样本示例 (前2个):")
        for i in range(min(2, len(code_samples))):
            s = code_samples[i]
            print(f"\n  样本{i+1}:")
            print(f"    problem长度: {len(s.get('problem', ''))}")
            print(f"    entry_point: {s.get('entry_point', 'N/A')[:50]}")
            print(f"    test长度: {len(s.get('test', ''))}")
            print(f"    task_id: {s.get('task_id', 'N/A')}")

    return {
        'total': len(samples),
        'types': dict(types),
        'code_count': code_count
    }

def main():
    print("\n" + "#"*70)
    print("# 数据集完整性验证")
    print("#"*70 + "\n")

    # 检查训练集
    train_original = check_dataset(
        "data/train/mixed_dataset.jsonl",
        "原始训练集"
    )

    print("\n")

    train_augmented = check_dataset(
        "data/train/mixed_dataset_augmented.jsonl",
        "增强训练集"
    )

    print("\n")

    # 检查测试集
    test = check_dataset(
        "data/test/mixed_dataset.jsonl",
        "测试集"
    )

    # 对比总结
    print("\n" + "="*70)
    print("📊 数据集对比总结")
    print("="*70)

    if train_original and train_augmented:
        print(f"\n原始训练集: {train_original['total']:,} samples")
        print(f"  Code: {train_original['code_count']} ({train_original['code_count']/train_original['total']*100:.2f}%)")

        print(f"\n增强训练集: {train_augmented['total']:,} samples")
        print(f"  Code: {train_augmented['code_count']} ({train_augmented['code_count']/train_augmented['total']*100:.2f}%)")
        print(f"  提升: {train_augmented['code_count']/train_original['code_count']:.1f}x")

    if test:
        print(f"\n测试集: {test['total']:,} samples")
        print(f"  Code: {test['code_count']} ({test['code_count']/test['total']*100:.2f}%)")
        print(f"  ⚠️  Code测试样本过少!")

    print("\n" + "="*70)
    print("💡 建议")
    print("="*70)

    if train_original and train_original['code_count'] < 1000:
        print("\n⚠️  当前使用的是原始训练集 (Code只有0.09%)")
        print("建议切换到增强训练集:")
        print("  ./switch_to_augmented_data.sh")
        print("  或")
        print("  cp data/train/mixed_dataset_augmented.jsonl data/train/mixed_dataset.jsonl")

    if test and test['code_count'] < 100:
        print("\n⚠️  测试集Code样本过少，评估不准确")
        print("建议增强测试集或添加HumanEval测试样本")

if __name__ == "__main__":
    main()
