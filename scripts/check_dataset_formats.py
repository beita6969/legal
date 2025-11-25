#!/usr/bin/env python3
"""
检查数据集格式的完整性和正确性
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# 路径设置
data_dir = Path('/home/yijia/.claude/11/integrated_aflow_roll/data')
mixed_dir = data_dir / 'mixed'

def check_required_fields(sample, problem_type, idx, filename):
    """检查必需字段"""
    required_fields = ['problem', 'problem_type', 'ground_truth']
    missing_fields = []

    for field in required_fields:
        if field not in sample or sample[field] is None or sample[field] == "":
            missing_fields.append(field)

    if missing_fields:
        print(f"  ⚠️  样本 {idx} 缺少字段: {missing_fields}")
        return False

    # 检查problem_type是否正确
    valid_types = ['math', 'code', 'qa', 'mixed']
    if sample['problem_type'] not in valid_types:
        print(f"  ⚠️  样本 {idx} 的problem_type无效: {sample['problem_type']}")
        return False

    # 针对code类型的特殊检查
    if problem_type == 'code':
        code_fields = ['entry_point', 'test']
        missing_code_fields = []
        for field in code_fields:
            if field not in sample:
                missing_code_fields.append(field)
        if missing_code_fields:
            print(f"  ⚠️  Code样本 {idx} 缺少字段: {missing_code_fields}")

    return True

def analyze_sample_format(sample, problem_type):
    """分析单个样本的格式"""
    analysis = {
        'problem_length': len(sample.get('problem', '')),
        'solution_length': len(sample.get('solution', '')),
        'ground_truth_length': len(sample.get('ground_truth', '')),
        'has_source': 'source' in sample,
        'source': sample.get('source', 'unknown'),
        'fields': list(sample.keys())
    }

    # 针对不同类型的特殊分析
    if problem_type == 'math':
        analysis['has_answer'] = 'answer' in sample
        analysis['has_level'] = 'level' in sample
        analysis['has_subject'] = 'subject' in sample
    elif problem_type == 'code':
        analysis['has_entry_point'] = 'entry_point' in sample
        analysis['has_test'] = 'test' in sample
        analysis['has_prompt'] = 'prompt' in sample

    return analysis

def check_dataset_file(filename):
    """检查单个数据集文件"""
    file_path = mixed_dir / filename

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    print(f"\n📊 检查数据集: {filename}")
    print("="*60)

    # 统计信息
    total_samples = 0
    valid_samples = 0
    type_counts = defaultdict(int)
    source_counts = defaultdict(int)
    format_issues = []
    sample_analyses = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue

            total_samples += 1

            try:
                sample = json.loads(line)
                problem_type = sample.get('problem_type', 'unknown')
                type_counts[problem_type] += 1

                source = sample.get('source', 'unknown')
                source_counts[source] += 1

                # 检查必需字段
                if check_required_fields(sample, problem_type, line_idx + 1, filename):
                    valid_samples += 1
                else:
                    format_issues.append((line_idx + 1, "缺少必需字段"))

                # 分析格式
                if total_samples <= 3:  # 分析前3个样本
                    analysis = analyze_sample_format(sample, problem_type)
                    sample_analyses[problem_type].append((line_idx + 1, analysis))

            except json.JSONDecodeError as e:
                format_issues.append((line_idx + 1, f"JSON解析错误: {e}"))

    # 输出统计
    print(f"\n📈 基本统计:")
    print(f"  总样本数: {total_samples}")
    print(f"  有效样本数: {valid_samples} ({valid_samples/total_samples*100:.1f}%)")

    print(f"\n📊 类型分布:")
    for ptype, count in sorted(type_counts.items()):
        print(f"  {ptype}: {count} ({count/total_samples*100:.1f}%)")

    print(f"\n📊 数据源分布:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {source}: {count} ({count/total_samples*100:.1f}%)")

    # 格式问题
    if format_issues:
        print(f"\n⚠️  格式问题 (前10个):")
        for idx, issue in format_issues[:10]:
            print(f"  行 {idx}: {issue}")
    else:
        print(f"\n✅ 所有样本格式正确")

    return sample_analyses, source_counts

def display_sample_examples(filename, num_examples=2):
    """显示样本示例"""
    file_path = mixed_dir / filename

    if not file_path.exists():
        return

    print(f"\n📝 样本示例 ({filename}):")
    print("="*60)

    # 按类型收集样本
    samples_by_type = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                problem_type = sample.get('problem_type', 'unknown')
                samples_by_type[problem_type].append(sample)

    # 显示每种类型的示例
    for ptype in ['math', 'code', 'qa', 'mixed']:
        if ptype in samples_by_type and samples_by_type[ptype]:
            print(f"\n【{ptype.upper()} 样本示例】")
            # 随机选择一个样本
            sample = random.choice(samples_by_type[ptype][:10])

            print(f"\n问题 (前200字符):")
            print(f"  {sample['problem'][:200]}..." if len(sample['problem']) > 200 else f"  {sample['problem']}")

            print(f"\nGround Truth (前200字符):")
            gt = sample.get('ground_truth', '')
            print(f"  {gt[:200]}..." if len(gt) > 200 else f"  {gt}")

            print(f"\n字段列表: {list(sample.keys())}")
            print(f"来源: {sample.get('source', 'unknown')}")

            # 特殊字段
            if ptype == 'math' and 'MATH' in sample.get('source', ''):
                print(f"\nMATH特有字段:")
                print(f"  - subject: {sample.get('subject', 'N/A')}")
                print(f"  - level: {sample.get('level', 'N/A')}")
                print(f"  - answer: {sample.get('answer', 'N/A')[:100]}..." if len(sample.get('answer', '')) > 100 else f"  - answer: {sample.get('answer', 'N/A')}")

def main():
    print("="*60)
    print("🔍 数据集格式检查")
    print("="*60)

    # 检查的数据集文件
    datasets_to_check = [
        'train_mixed_with_math.jsonl',
        'test_mixed.jsonl'
    ]

    all_analyses = {}

    for dataset in datasets_to_check:
        analyses, sources = check_dataset_file(dataset)
        all_analyses[dataset] = analyses

        # 显示样本示例
        display_sample_examples(dataset, num_examples=1)

    # 特别检查MATH样本
    print("\n" + "="*60)
    print("🔬 MATH数据集样本深度检查")
    print("="*60)

    math_file = mixed_dir / 'train_mixed_with_math.jsonl'
    math_samples = []

    with open(math_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                if sample.get('source') == 'MATH':
                    math_samples.append(sample)

    print(f"\n找到 {len(math_samples)} 个MATH样本")

    if math_samples:
        # 检查前5个MATH样本
        print("\n详细检查前3个MATH样本:")
        for i, sample in enumerate(math_samples[:3]):
            print(f"\n--- MATH样本 {i+1} ---")
            print(f"问题类型: {sample.get('problem_type')}")
            print(f"学科: {sample.get('subject')}")
            print(f"难度: {sample.get('level')}")
            print(f"\n问题 (前300字符):")
            print(f"{sample['problem'][:300]}..." if len(sample['problem']) > 300 else sample['problem'])
            print(f"\n解答 (前300字符):")
            solution = sample.get('solution', sample.get('ground_truth', ''))
            print(f"{solution[:300]}..." if len(solution) > 300 else solution)
            print(f"\n答案: {sample.get('answer', 'N/A')}")
            print(f"\n所有字段: {list(sample.keys())}")

    print("\n" + "="*60)
    print("✅ 检查完成")
    print("="*60)

if __name__ == '__main__':
    random.seed(42)  # 保证示例的一致性
    main()
