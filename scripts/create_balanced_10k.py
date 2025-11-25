#!/usr/bin/env python3
"""
创建10k平衡数据集
Math: 40% (4000)
Code: 30% (3000)
QA:   30% (3000)
"""
import json
import random
from typing import List, Dict
from collections import Counter

def load_jsonl(file_path: str) -> List[Dict]:
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

def save_jsonl(samples: List[Dict], file_path: str):
    with open(file_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

def identify_sub_dataset(sample: Dict) -> str:
    """识别样本来源的子数据集"""
    ptype = sample.get('problem_type', 'unknown')

    if ptype == 'math':
        if 'category' in sample or 'difficulty' in sample:
            return 'MATH'
        return 'GSM8K'

    elif ptype == 'code':
        return 'HumanEval'

    elif ptype == 'qa':
        if 'type' in sample and 'context' in sample:
            return 'HotpotQA'
        elif 'passage' in sample:
            if 'answer_type' in sample:
                return 'DROP'
            return 'CommonsenseQA'
        return 'Other-QA'

    return 'Unknown'

def create_balanced_10k_dataset(
    input_file: str,
    output_file: str,
    target_total: int = 10000,
    math_ratio: float = 0.40,
    code_ratio: float = 0.30,
    qa_ratio: float = 0.30
):
    print("="*70)
    print("创建10k平衡数据集")
    print("="*70)

    # 加载数据
    print(f"\n📂 加载数据: {input_file}")
    all_samples = load_jsonl(input_file)
    print(f"✅ 加载完成: {len(all_samples):,} 个样本")

    # 分类
    math_samples = [s for s in all_samples if s.get('problem_type') == 'math']
    code_samples = [s for s in all_samples if s.get('problem_type') == 'code']
    qa_samples = [s for s in all_samples if s.get('problem_type') == 'qa']

    print(f"\n📊 可用样本:")
    print(f"  Math: {len(math_samples):,}")
    print(f"  Code: {len(code_samples):,}")
    print(f"  QA:   {len(qa_samples):,}")

    # 计算目标数量
    target_math = int(target_total * math_ratio)
    target_code = int(target_total * code_ratio)
    target_qa = int(target_total * qa_ratio)

    # 调整以确保总数为10000
    diff = target_total - (target_math + target_code + target_qa)
    target_math += diff

    print(f"\n🎯 目标分布:")
    print(f"  Math: {target_math} ({target_math/target_total*100:.1f}%)")
    print(f"  Code: {target_code} ({target_code/target_total*100:.1f}%)")
    print(f"  QA:   {target_qa} ({target_qa/target_total*100:.1f}%)")
    print(f"  总计: {target_total}")

    # 采样
    print(f"\n🎲 随机采样...")
    random.seed(42)

    selected_math = random.sample(math_samples, min(target_math, len(math_samples)))
    selected_code = random.sample(code_samples, min(target_code, len(code_samples)))
    selected_qa = random.sample(qa_samples, min(target_qa, len(qa_samples)))

    # 合并
    final_samples = selected_math + selected_code + selected_qa
    random.shuffle(final_samples)

    print(f"✅ 采样完成: {len(final_samples)} 个样本")

    # 分析子数据集组成
    print(f"\n📊 子数据集分布:")
    sub_datasets = Counter(identify_sub_dataset(s) for s in final_samples)

    for dataset, count in sub_datasets.most_common():
        print(f"  {dataset:20s}: {count:5d} ({count/len(final_samples)*100:5.1f}%)")

    # 验证类型分布
    types = Counter(s.get('problem_type') for s in final_samples)
    print(f"\n✅ 最终类型分布:")
    for ptype, count in types.most_common():
        print(f"  {ptype}: {count} ({count/len(final_samples)*100:.1f}%)")

    # 保存
    print(f"\n💾 保存: {output_file}")
    save_jsonl(final_samples, output_file)

    # 保存统计
    stats = {
        'total': len(final_samples),
        'target_total': target_total,
        'math_count': len(selected_math),
        'code_count': len(selected_code),
        'qa_count': len(selected_qa),
        'math_ratio': len(selected_math) / len(final_samples),
        'code_ratio': len(selected_code) / len(final_samples),
        'qa_ratio': len(selected_qa) / len(final_samples),
        'sub_datasets': dict(sub_datasets)
    }

    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"📊 统计信息: {stats_file}")

    print("\n" + "="*70)
    print("✅ 10k平衡数据集创建完成!")
    print("="*70)

if __name__ == "__main__":
    create_balanced_10k_dataset(
        input_file="data/train/mixed_dataset_augmented_v2.jsonl",
        output_file="data/train/balanced_10k_dataset.jsonl",
        target_total=10000,
        math_ratio=0.40,
        code_ratio=0.30,
        qa_ratio=0.30
    )
