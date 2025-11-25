#!/usr/bin/env python3
"""
创建混合数据集（训练+测试，不包含验证集）
按照 math:qa:code = 4:3:3 的比例
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# 设置随机种子
random.seed(42)

# 配置
ratios = {
    'math': 0.4,  # 40%
    'qa': 0.3,    # 30%
    'code': 0.3   # 30%
}

data_sources = {
    'math': [
        ('data/gsm8k/train.jsonl', 'gsm8k'),
        ('data/raw/gsm8k/train.jsonl', 'gsm8k'),
        ('data/raw/gsm8k/test.jsonl', 'gsm8k'),
    ],
    'code': [
        ('data/mbpp/train.jsonl', 'mbpp'),
        ('data/humaneval/humaneval_full.jsonl', 'humaneval'),
        ('data/humaneval/humaneval_validate.jsonl', 'humaneval'),
        ('data/humaneval/humaneval_test.jsonl', 'humaneval'),
        ('data/processed/test_mixed.jsonl', 'humaneval'),  # 包含100个humaneval样本
    ],
    'qa': [
        ('data/raw/commonsenseqa/train.jsonl', 'commonsenseqa'),
        ('data/raw/commonsenseqa/dev.jsonl', 'commonsenseqa'),
        ('data/raw/commonsenseqa/test.jsonl', 'commonsenseqa'),
        ('data/hotpotqa/train.jsonl', 'hotpotqa'),
        ('data/drop/train.jsonl', 'drop'),
    ],
    'mixed': [
        ('data/raw/mmlu/auxiliary_train.jsonl', 'mmlu'),
        ('data/raw/mmlu/test.jsonl', 'mmlu'),
        ('data/raw/mmlu/validation.jsonl', 'mmlu'),
    ]
}

def load_dataset(file_path, source_name, problem_type):
    """加载数据集并标准化格式"""
    samples = []
    path = Path(file_path)

    if not path.exists():
        print(f"  ⚠️  文件不存在: {file_path}")
        return samples

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)

                # 标准化字段
                if 'source' not in sample:
                    sample['source'] = source_name
                if 'problem_type' not in sample:
                    sample['problem_type'] = problem_type

                # 处理HumanEval特殊格式
                if source_name == 'humaneval':
                    if 'problem' not in sample and 'prompt' in sample:
                        sample['problem'] = sample['prompt']
                    if 'ground_truth' not in sample and 'canonical_solution' in sample:
                        sample['ground_truth'] = sample['canonical_solution']

                # 处理GSM8K格式
                elif source_name == 'gsm8k':
                    if 'problem' not in sample and 'question' in sample:
                        sample['problem'] = sample['question']
                    if 'ground_truth' not in sample and 'answer' in sample:
                        sample['ground_truth'] = sample['answer']

                # 处理CommonsenseQA格式
                elif source_name == 'commonsenseqa':
                    if 'problem' not in sample and 'question' in sample:
                        # 构建问题文本，包含选项
                        question_text = sample['question'].get('stem', '')
                        if 'choices' in sample['question']:
                            choices_text = '\n'.join([f"{c['label']}: {c['text']}"
                                                      for c in sample['question']['choices']])
                            sample['problem'] = f"{question_text}\n{choices_text}"
                        else:
                            sample['problem'] = question_text
                    if 'ground_truth' not in sample and 'answerKey' in sample:
                        sample['ground_truth'] = sample['answerKey']

                # 确保必要字段存在
                if 'problem' in sample and 'ground_truth' in sample:
                    samples.append(sample)

            except json.JSONDecodeError:
                print(f"  ⚠️  JSON解析错误: {file_path} 第{line_num}行")
            except Exception as e:
                print(f"  ⚠️  处理错误: {file_path} 第{line_num}行: {e}")

    return samples

def main():
    print("="*60)
    print("创建混合数据集（训练+测试）")
    print("="*60)

    # 收集所有数据
    all_data = defaultdict(list)
    source_stats = defaultdict(lambda: defaultdict(int))

    # 加载所有数据集
    for problem_type, sources in data_sources.items():
        print(f"\n加载 {problem_type} 类型数据:")
        for file_path, source_name in sources:
            samples = load_dataset(file_path, source_name, problem_type)
            if samples:
                all_data[problem_type].extend(samples)
                source_stats[problem_type][source_name] += len(samples)
                print(f"  ✅ {source_name}: {len(samples)} 样本")

    # 统计信息
    print("\n" + "="*60)
    print("数据统计:")
    for problem_type, sources in source_stats.items():
        total = sum(sources.values())
        print(f"\n{problem_type} 类型 (总计: {total}):")
        for source, count in sorted(sources.items()):
            print(f"  - {source}: {count}")

    # 计算目标样本数，确保比例为 math:qa:code = 4:3:3
    # 先统计每种类型的可用样本数
    available_counts = {}
    for problem_type in ['math', 'code', 'qa']:
        available_counts[problem_type] = len(all_data.get(problem_type, []))

    print("\n可用样本数:")
    for ptype, count in available_counts.items():
        print(f"  {ptype}: {count}")

    # 设定目标比例和总样本数
    # 由于code样本很少，我们将进行大量复制
    target_samples = {
        'math': 8000,  # 40%
        'qa': 6000,    # 30%
        'code': 6000   # 30% - 需要大量复制
    }

    print("\n目标样本数:")
    for ptype, count in target_samples.items():
        print(f"  {ptype}: {count} (需要{count/available_counts.get(ptype, 1):.1f}倍)")

    # 创建训练集和测试集
    train_ratio = 0.9  # 90% for training
    test_ratio = 0.1   # 10% for testing

    train_data = []
    test_data = []

    print("\n" + "="*60)
    print("创建数据集:")

    # 对每种类型的数据进行处理
    for problem_type in ['math', 'qa', 'code']:
        if problem_type in all_data:
            samples = all_data[problem_type]
            random.shuffle(samples)

            # 处理样本数量
            if len(samples) < target_samples[problem_type]:
                print(f"\n⚠️  {problem_type} 样本不足 ({len(samples)} < {target_samples[problem_type]})，进行复制...")
                original_count = len(samples)
                # 计算需要复制的倍数
                multiplication_factor = (target_samples[problem_type] + original_count - 1) // original_count
                # 复制样本
                expanded_samples = []
                for i in range(multiplication_factor):
                    # 为每个复制添加标记以区分
                    for sample in samples:
                        sample_copy = sample.copy()
                        sample_copy['duplication_id'] = i
                        expanded_samples.append(sample_copy)
                random.shuffle(expanded_samples)
                samples = expanded_samples[:target_samples[problem_type]]
                print(f"  原始: {original_count}, 复制{multiplication_factor}倍后: {len(samples)}")
            else:
                # 如果样本充足，只取需要的数量
                samples = samples[:target_samples[problem_type]]

            # 分割为训练集和测试集
            split_point = int(len(samples) * train_ratio)
            train_samples = samples[:split_point]
            test_samples = samples[split_point:]

            train_data.extend(train_samples)
            test_data.extend(test_samples)

            print(f"\n{problem_type}:")
            print(f"  训练集: {len(train_samples)} 样本")
            print(f"  测试集: {len(test_samples)} 样本")

    # 添加mixed类型（如果需要，可以按比例分配到其他类型）
    if 'mixed' in all_data:
        samples = all_data['mixed']
        random.shuffle(samples)

        # 将mixed类型按比例分配到train和test
        split_point = int(len(samples) * train_ratio)
        train_samples = samples[:split_point]
        test_samples = samples[split_point:]

        train_data.extend(train_samples)
        test_data.extend(test_samples)

        print(f"\nmixed:")
        print(f"  训练集: {len(train_samples)} 样本")
        print(f"  测试集: {len(test_samples)} 样本")

    # 打乱数据
    random.shuffle(train_data)
    random.shuffle(test_data)

    # 保存数据集
    print("\n" + "="*60)
    print("保存数据集:")

    # 创建输出目录
    output_dir = Path('data/final_mixed')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存训练集
    train_file = output_dir / 'train.jsonl'
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  ✅ 训练集: {train_file} ({len(train_data)} 样本)")

    # 保存测试集
    test_file = output_dir / 'test.jsonl'
    with open(test_file, 'w', encoding='utf-8') as f:
        for sample in test_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  ✅ 测试集: {test_file} ({len(test_data)} 样本)")

    # 分析最终分布
    print("\n" + "="*60)
    print("最终数据分布:")

    for dataset_name, dataset in [('训练集', train_data), ('测试集', test_data)]:
        type_counts = defaultdict(int)
        source_counts = defaultdict(int)

        for sample in dataset:
            type_counts[sample.get('problem_type', 'unknown')] += 1
            source_counts[sample.get('source', 'unknown')] += 1

        print(f"\n{dataset_name} (共 {len(dataset)} 样本):")
        print("  按类型:")
        for ptype, count in sorted(type_counts.items()):
            percentage = count / len(dataset) * 100
            print(f"    {ptype}: {count} ({percentage:.1f}%)")

        print("  按来源:")
        for source, count in sorted(source_counts.items()):
            percentage = count / len(dataset) * 100
            print(f"    {source}: {count} ({percentage:.1f}%)")

    print("\n" + "="*60)
    print("✅ 混合数据集创建完成！")
    print("="*60)

if __name__ == '__main__':
    main()
