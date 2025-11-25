#!/usr/bin/env python3
"""
增强测试数据集
"""
import json
import random
from typing import List, Dict

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

def augment_test_data(
    input_file: str,
    output_file: str,
    humaneval_test_file: str
):
    print("="*70)
    print("测试集增强")
    print("="*70)

    # 加载原始测试集
    print(f"\n📂 加载测试集: {input_file}")
    test_samples = load_jsonl(input_file)
    print(f"✅ 原始测试集: {len(test_samples)} 个样本")

    # 统计
    from collections import Counter
    types = Counter(s.get('problem_type') for s in test_samples)
    print(f"\n原始分布:")
    for ptype, count in types.most_common():
        print(f"  {ptype}: {count} ({count/len(test_samples)*100:.2f}%)")

    # 加载HumanEval测试样本
    print(f"\n📥 加载HumanEval测试: {humaneval_test_file}")
    humaneval_raw = load_jsonl(humaneval_test_file)

    humaneval_test = []
    for hr in humaneval_raw:
        sample = {
            'problem': hr.get('prompt', ''),
            'problem_type': 'code',
            'ground_truth': hr.get('canonical_solution', ''),
            'entry_point': hr.get('entry_point', ''),
            'test': hr.get('test', ''),
            'task_id': hr.get('task_id', '')
        }
        humaneval_test.append(sample)

    print(f"✅ HumanEval测试样本: {len(humaneval_test)}")

    # 合并
    final_samples = test_samples + humaneval_test

    # 统计
    final_types = Counter(s.get('problem_type') for s in final_samples)
    print(f"\n📊 增强后分布:")
    for ptype, count in final_types.most_common():
        print(f"  {ptype}: {count} ({count/len(final_samples)*100:.2f}%)")
    print(f"  总计: {len(final_samples)}")

    # 保存
    print(f"\n💾 保存增强测试集: {output_file}")
    save_jsonl(final_samples, output_file)
    print(f"✅ 保存完成!")

    print("\n" + "="*70)
    print("✅ 测试集增强完成!")
    print("="*70)

if __name__ == "__main__":
    augment_test_data(
        input_file="data/test/mixed_dataset.jsonl",
        output_file="data/test/mixed_dataset_augmented.jsonl",
        humaneval_test_file="data/humaneval/humaneval_test.jsonl"
    )
