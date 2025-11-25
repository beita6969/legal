#!/usr/bin/env python3
"""
修复测试集 - 移除不完整样本，只使用HumanEval
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

def is_valid_code_sample(sample: Dict) -> bool:
    """检查Code样本是否完整"""
    if sample.get('problem_type') != 'code':
        return True

    required = ['problem', 'entry_point', 'test', 'ground_truth']
    for field in required:
        if field not in sample or not sample[field]:
            return False
    return True

def identify_sub_dataset(sample: Dict) -> str:
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
        elif 'passage' in sample and 'answer_type' in sample:
            return 'DROP'
        elif 'passage' in sample:
            return 'CommonsenseQA'
        return 'Other-QA'
    return 'Unknown'

def fix_and_create_test_set(
    augmented_test_file: str,
    output_file: str,
    target_total: int = 2000
):
    print("="*70)
    print("修复并创建测试集")
    print("="*70)

    # 加载增强测试集
    print(f"\n📂 加载: {augmented_test_file}")
    all_samples = load_jsonl(augmented_test_file)
    print(f"✅ 加载完成: {len(all_samples):,}")

    # 过滤掉不完整的Code样本
    print(f"\n🔍 过滤不完整样本...")
    valid_samples = [s for s in all_samples if is_valid_code_sample(s)]
    removed = len(all_samples) - len(valid_samples)
    print(f"✅ 有效样本: {len(valid_samples)}")
    print(f"⚠️  移除不完整Code样本: {removed}")

    # 分类
    math_samples = [s for s in valid_samples if s.get('problem_type') == 'math']
    code_samples = [s for s in valid_samples if s.get('problem_type') == 'code']
    qa_samples = [s for s in valid_samples if s.get('problem_type') == 'qa']

    print(f"\n📊 有效样本:")
    print(f"  Math: {len(math_samples)}")
    print(f"  Code: {len(code_samples)}")
    print(f"  QA:   {len(qa_samples)}")

    # 计算目标数量 (40:30:30)
    target_math = int(target_total * 0.40)
    target_code = int(target_total * 0.30)
    target_qa = int(target_total * 0.30)

    # 调整确保总数精确
    diff = target_total - (target_math + target_code + target_qa)
    target_math += diff

    print(f"\n🎯 目标分布:")
    print(f"  Math: {target_math} (40%)")
    print(f"  Code: {target_code} (30%)")
    print(f"  QA:   {target_qa} (30%)")

    # 处理Code样本不足
    if len(code_samples) < target_code:
        print(f"\n⚠️  Code样本不足: {len(code_samples)} < {target_code}")
        print(f"   将重复使用")
        repetitions = target_code // len(code_samples)
        remainder = target_code % len(code_samples)
        code_samples_extended = code_samples * repetitions
        if remainder > 0:
            code_samples_extended += random.sample(code_samples, remainder)
        code_samples = code_samples_extended
        print(f"   重复后: {len(code_samples)}")

    # 采样
    random.seed(42)
    selected_math = random.sample(math_samples, min(target_math, len(math_samples)))
    selected_code = code_samples[:target_code]
    selected_qa = random.sample(qa_samples, min(target_qa, len(qa_samples)))

    # 合并和打乱
    final_samples = selected_math + selected_code + selected_qa
    random.shuffle(final_samples)

    print(f"\n✅ 最终样本: {len(final_samples)}")

    # 验证Code完整性
    code_in_final = [s for s in final_samples if s.get('problem_type') == 'code']
    invalid_code = [s for s in code_in_final if not is_valid_code_sample(s)]
    if invalid_code:
        print(f"\n❌ 警告: {len(invalid_code)}个不完整Code样本")
        return
    else:
        print(f"\n✅ 所有Code样本完整 ({len(code_in_final)}个)")

    # 统计
    types = Counter(s.get('problem_type') for s in final_samples)
    print(f"\n📊 最终类型分布:")
    for t, c in types.most_common():
        print(f"  {t}: {c} ({c/len(final_samples)*100:.1f}%)")

    sub_datasets = Counter(identify_sub_dataset(s) for s in final_samples)
    print(f"\n📊 子数据集分布:")
    for ds, c in sub_datasets.most_common():
        print(f"  {ds:20s}: {c:4d} ({c/len(final_samples)*100:.1f}%)")

    # 保存
    print(f"\n💾 保存: {output_file}")
    save_jsonl(final_samples, output_file)

    stats = {
        'total': len(final_samples),
        'math_count': len(selected_math),
        'code_count': len(selected_code),
        'qa_count': len(selected_qa),
        'math_ratio': len(selected_math) / len(final_samples),
        'code_ratio': len(selected_code) / len(final_samples),
        'qa_ratio': len(selected_qa) / len(final_samples),
        'sub_datasets': dict(sub_datasets),
        'removed_invalid': removed
    }

    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"📊 统计: {stats_file}")
    print("\n✅ 完成!")

if __name__ == "__main__":
    fix_and_create_test_set(
        augmented_test_file="data/test/mixed_dataset_augmented.jsonl",
        output_file="data/test/balanced_test_dataset_v2.jsonl",
        target_total=2000
    )
