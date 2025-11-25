#!/usr/bin/env python3
"""
评估每个小数据集的准确率
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

def load_test_data(test_file: str) -> List[Dict]:
    """加载测试数据"""
    samples = []
    with open(test_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

def infer_dataset(sample: Dict) -> str:
    """推断样本来源数据集"""
    # 根据sample的字段判断来源
    problem_type = sample.get('problem_type', '')

    # Math类型
    if problem_type == 'math':
        # 检查是否有category字段 (MATH dataset)
        if 'category' in sample or 'difficulty' in sample:
            return 'MATH'
        # 检查solution长度判断GSM8K (通常较短)
        if 'solution' in sample:
            return 'MATH'  # 默认归类为MATH
        return 'GSM8K'

    # Code类型
    elif problem_type == 'code':
        problem = sample.get('problem', '')
        # HumanEval: Python函数定义开头
        if problem.strip().startswith('def '):
            return 'HumanEval'
        # MBPP: 通常有更详细的描述
        return 'MBPP'

    # QA类型
    elif problem_type == 'qa':
        # 检查字段判断来源
        if 'type' in sample and 'context' in sample:
            # HotpotQA格式
            return 'HotpotQA'
        elif 'passage' in sample:
            # CommonsenseQA或其他passage-based
            return 'CommonsenseQA'
        elif 'question' in sample:
            # MMLU格式
            return 'MMLU'
        return 'Other-QA'

    return 'Unknown'

def group_by_dataset(samples: List[Dict]) -> Dict[str, List[Dict]]:
    """按数据集分组"""
    grouped = defaultdict(list)
    for sample in samples:
        dataset = infer_dataset(sample)
        grouped[dataset].append(sample)
    return grouped

def print_statistics(grouped: Dict[str, List[Dict]]):
    """打印统计信息"""
    print("\n" + "="*70)
    print("测试集数据分布")
    print("="*70)

    total = sum(len(samples) for samples in grouped.values())
    print(f"\n总样本数: {total}\n")

    # 按类型分组统计
    math_count = 0
    code_count = 0
    qa_count = 0

    for dataset, samples in sorted(grouped.items(), key=lambda x: -len(x[1])):
        count = len(samples)
        percentage = (count / total) * 100
        print(f"{dataset:20s}: {count:6d} ({percentage:5.1f}%)")

        # 统计类型
        if samples:
            sample_type = samples[0].get('problem_type', '')
            if sample_type == 'math':
                math_count += count
            elif sample_type == 'code':
                code_count += count
            elif sample_type == 'qa':
                qa_count += count

    print("\n" + "-"*70)
    print(f"Math总计: {math_count} ({math_count/total*100:.1f}%)")
    print(f"Code总计: {code_count} ({code_count/total*100:.1f}%)")
    print(f"QA总计:   {qa_count} ({qa_count/total*100:.1f}%)")
    print("="*70)

def main():
    test_file = "data/test/mixed_dataset.jsonl"

    if not Path(test_file).exists():
        print(f"❌ 测试文件不存在: {test_file}")
        sys.exit(1)

    print(f"📊 加载测试数据: {test_file}")
    samples = load_test_data(test_file)
    print(f"✅ 加载完成: {len(samples)} 个样本")

    print("\n🔍 分析数据集分布...")
    grouped = group_by_dataset(samples)

    print_statistics(grouped)

    # 保存分组结果
    output_file = "data/test/dataset_breakdown.json"
    breakdown = {
        dataset: len(samples)
        for dataset, samples in grouped.items()
    }

    with open(output_file, 'w') as f:
        json.dump(breakdown, f, indent=2)

    print(f"\n💾 数据集分布已保存: {output_file}")

if __name__ == "__main__":
    main()
