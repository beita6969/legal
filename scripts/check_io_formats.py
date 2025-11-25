#!/usr/bin/env python3
"""
详细检查每种数据类型的输入输出格式
"""

import json
import random
from pathlib import Path

# 路径设置
data_dir = Path('/home/yijia/.claude/11/integrated_aflow_roll/data')
mixed_dir = data_dir / 'mixed'

def check_sample_io_format():
    """检查每种类型的输入输出格式"""
    file_path = mixed_dir / 'train_mixed_with_math.jsonl'

    # 收集各类型样本
    samples_by_type = {
        'math': [],
        'code': [],
        'qa': [],
        'mixed': []
    }

    samples_by_source = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                problem_type = sample.get('problem_type')
                source = sample.get('source', 'unknown')

                if problem_type in samples_by_type:
                    samples_by_type[problem_type].append(sample)

                if source not in samples_by_source:
                    samples_by_source[source] = []
                samples_by_source[source].append(sample)

    print("="*80)
    print("📊 数据集输入输出格式详细检查")
    print("="*80)

    # 1. MATH类型检查
    print("\n" + "="*80)
    print("1️⃣ MATH 类型样本检查")
    print("="*80)

    math_samples = samples_by_type['math']
    print(f"\n总数: {len(math_samples)} 个样本")

    # 按source分类
    math_by_source = {}
    for sample in math_samples:
        source = sample.get('source', 'unknown')
        if source not in math_by_source:
            math_by_source[source] = []
        math_by_source[source].append(sample)

    for source, samples in math_by_source.items():
        print(f"\n📌 {source} ({len(samples)} 样本):")
        # 随机选择一个样本
        sample = random.choice(samples[:10])

        print(f"\n【示例】")
        print(f"\n输入 (problem):")
        print(f"  类型: {type(sample['problem']).__name__}")
        print(f"  长度: {len(sample['problem'])} 字符")
        print(f"  内容预览: {sample['problem'][:150]}...")

        print(f"\n输出 (ground_truth):")
        gt = sample.get('ground_truth', '')
        print(f"  类型: {type(gt).__name__}")
        print(f"  长度: {len(gt)} 字符")
        print(f"  内容预览: {gt[:150]}..." if len(gt) > 150 else f"  内容: {gt}")

        # MATH特有字段
        if source == 'MATH':
            print(f"\nMATH特有字段:")
            print(f"  subject: {sample.get('subject')}")
            print(f"  level: {sample.get('level')}")
            print(f"  answer: {sample.get('answer')}")
            print(f"  solution长度: {len(sample.get('solution', ''))} 字符")

    # 2. CODE类型检查
    print("\n" + "="*80)
    print("2️⃣ CODE 类型样本检查")
    print("="*80)

    code_samples = samples_by_type['code']
    print(f"\n总数: {len(code_samples)} 个样本")

    # 按source分类
    code_by_source = {}
    for sample in code_samples:
        source = sample.get('source', 'unknown')
        if source not in code_by_source:
            code_by_source[source] = []
        code_by_source[source].append(sample)

    for source, samples in code_by_source.items():
        print(f"\n📌 {source} ({len(samples)} 样本):")
        # 随机选择一个样本
        sample = random.choice(samples[:10])

        print(f"\n【示例】")
        print(f"\n输入 (problem):")
        print(f"  类型: {type(sample['problem']).__name__}")
        print(f"  长度: {len(sample['problem'])} 字符")
        # 检查是否包含函数签名
        has_def = 'def ' in sample['problem']
        has_docstring = '"""' in sample['problem'] or "'''" in sample['problem']
        print(f"  包含函数定义: {has_def}")
        print(f"  包含文档字符串: {has_docstring}")
        print(f"  内容预览: {sample['problem'][:200]}...")

        print(f"\n输出 (ground_truth):")
        gt = sample.get('ground_truth', '')
        print(f"  类型: {type(gt).__name__}")
        print(f"  长度: {len(gt)} 字符")
        print(f"  内容预览: {gt[:200]}..." if len(gt) > 200 else f"  内容: {gt}")

        # Code特有字段
        if 'entry_point' in sample:
            print(f"\nCode特有字段:")
            print(f"  entry_point: {sample.get('entry_point')}")
            print(f"  test长度: {len(sample.get('test', ''))} 字符")
            if 'task_id' in sample:
                print(f"  task_id: {sample.get('task_id')}")

    # 3. QA类型检查
    print("\n" + "="*80)
    print("3️⃣ QA 类型样本检查")
    print("="*80)

    qa_samples = samples_by_type['qa']
    print(f"\n总数: {len(qa_samples)} 个样本")

    # 按source分类
    qa_by_source = {}
    for sample in qa_samples:
        source = sample.get('source', 'unknown')
        if source not in qa_by_source:
            qa_by_source[source] = []
        qa_by_source[source].append(sample)

    for source, samples in qa_by_source.items():
        print(f"\n📌 {source} ({len(samples)} 样本):")
        # 随机选择一个样本
        sample = random.choice(samples[:10])

        print(f"\n【示例】")
        print(f"\n输入 (problem):")
        print(f"  类型: {type(sample['problem']).__name__}")
        print(f"  长度: {len(sample['problem'])} 字符")
        has_choices = 'Choices:' in sample['problem'] or 'choices' in sample
        print(f"  包含选项: {has_choices}")
        print(f"  内容预览: {sample['problem'][:200]}...")

        print(f"\n输出 (ground_truth):")
        gt = sample.get('ground_truth', '')
        print(f"  类型: {type(gt).__name__}")
        print(f"  长度: {len(gt)} 字符")
        print(f"  内容: {gt}")

        # QA特有字段
        if 'choices' in sample:
            print(f"\nQA特有字段:")
            print(f"  choices: {sample.get('choices')}")

    # 4. 格式一致性检查
    print("\n" + "="*80)
    print("4️⃣ 格式一致性检查")
    print("="*80)

    print("\n✅ Problem字段格式:")
    for ptype, samples in samples_by_type.items():
        if samples:
            all_str = all(isinstance(s['problem'], str) for s in samples[:100])
            print(f"  {ptype}: {'全部为字符串' if all_str else '❌ 存在非字符串'}")

    print("\n✅ Ground Truth字段格式:")
    for ptype, samples in samples_by_type.items():
        if samples:
            all_str = all(isinstance(s['ground_truth'], str) for s in samples[:100])
            empty_count = sum(1 for s in samples[:100] if s['ground_truth'] == '')
            print(f"  {ptype}: {'全部为字符串' if all_str else '❌ 存在非字符串'} (空值: {empty_count})")

    # 5. 数据源特定格式
    print("\n" + "="*80)
    print("5️⃣ 数据源特定格式检查")
    print("="*80)

    for source in ['MATH', 'humaneval', 'mbpp', 'gsm8k', 'commonsenseqa', 'hotpotqa', 'mmlu']:
        if source in samples_by_source and samples_by_source[source]:
            samples = samples_by_source[source]
            print(f"\n📌 {source} ({len(samples)} 样本):")

            # 统计字段出现率
            field_counts = {}
            for sample in samples[:100]:
                for field in sample.keys():
                    if field not in field_counts:
                        field_counts[field] = 0
                    field_counts[field] += 1

            print("  字段出现率:")
            for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
                if count == len(samples[:100]):
                    print(f"    {field}: 100% ✅")
                elif count > 0:
                    print(f"    {field}: {count/len(samples[:100])*100:.1f}%")

    print("\n" + "="*80)
    print("✅ 检查完成")
    print("="*80)

if __name__ == '__main__':
    random.seed(42)
    check_sample_io_format()
