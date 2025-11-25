#!/usr/bin/env python3
"""
修复和重新增强训练数据
- 只使用HumanEval（格式完整）
- 移除格式不完整的MBPP样本
- 创建干净的增强数据集
"""
import json
import random
from pathlib import Path
from typing import List, Dict

def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

def save_jsonl(samples: List[Dict], file_path: str):
    """保存JSONL文件"""
    with open(file_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

def is_valid_code_sample(sample: Dict) -> bool:
    """检查Code样本是否格式完整"""
    if sample.get('problem_type') != 'code':
        return True  # 非Code样本都有效

    # Code样本必须有这些字段
    required_fields = ['problem', 'entry_point', 'test', 'ground_truth']
    for field in required_fields:
        if field not in sample or not sample[field]:
            return False

    return True

def augment_training_data_clean(
    input_file: str,
    output_file: str,
    humaneval_file: str,
    target_code_ratio: float = 0.10
):
    """
    创建干净的增强训练数据
    """
    print("="*70)
    print("训练数据修复和增强")
    print("="*70)

    # 1. 加载原始数据
    print(f"\n📂 加载原始训练集: {input_file}")
    samples = load_jsonl(input_file)
    print(f"✅ 加载完成: {len(samples)} 个样本")

    # 2. 过滤掉格式不完整的Code样本
    print(f"\n🔍 过滤格式不完整的样本...")
    valid_samples = []
    invalid_code_count = 0

    for s in samples:
        if is_valid_code_sample(s):
            valid_samples.append(s)
        else:
            invalid_code_count += 1

    print(f"✅ 有效样本: {len(valid_samples)}")
    print(f"⚠️  移除无效Code样本: {invalid_code_count}")

    # 3. 分类
    math_samples = [s for s in valid_samples if s.get('problem_type') == 'math']
    qa_samples = [s for s in valid_samples if s.get('problem_type') == 'qa']
    code_samples = [s for s in valid_samples if s.get('problem_type') == 'code']

    print(f"\n📊 有效样本分布:")
    print(f"  Math: {len(math_samples)}")
    print(f"  QA:   {len(qa_samples)}")
    print(f"  Code: {len(code_samples)}")

    # 4. 加载HumanEval
    print(f"\n📥 加载HumanEval: {humaneval_file}")
    humaneval_raw = load_jsonl(humaneval_file)

    humaneval_samples = []
    for hr in humaneval_raw:
        sample = {
            'problem': hr.get('prompt', ''),
            'problem_type': 'code',
            'ground_truth': hr.get('canonical_solution', ''),
            'entry_point': hr.get('entry_point', ''),
            'test': hr.get('test', ''),
            'task_id': hr.get('task_id', '')
        }
        if is_valid_code_sample(sample):
            humaneval_samples.append(sample)

    print(f"✅ HumanEval有效样本: {len(humaneval_samples)}")
    code_samples.extend(humaneval_samples)

    # 5. 计算需要的Code样本数
    total_non_code = len(math_samples) + len(qa_samples)
    target_code_count = int(total_non_code * target_code_ratio / (1 - target_code_ratio))
    current_code_count = len(code_samples)

    print(f"\n🎯 Code样本目标: {target_code_count}")
    print(f"   当前Code样本: {current_code_count}")
    print(f"   需要增加: {max(0, target_code_count - current_code_count)}")

    # 6. 重复Code样本
    if current_code_count > 0 and current_code_count < target_code_count:
        repetitions = target_code_count // current_code_count
        remainder = target_code_count % current_code_count

        print(f"\n🔄 重复策略:")
        print(f"   完整重复: {repetitions} 次")
        print(f"   额外样本: {remainder} 个")

        augmented_code_samples = code_samples * repetitions
        if remainder > 0:
            extra_samples = random.sample(code_samples, remainder)
            augmented_code_samples.extend(extra_samples)

        print(f"✅ 增强后Code样本: {len(augmented_code_samples)}")
    else:
        augmented_code_samples = code_samples
        print(f"✅ Code样本数量合适")

    # 7. 合并和打乱
    final_samples = math_samples + qa_samples + augmented_code_samples
    random.shuffle(final_samples)

    # 8. 统计
    final_math = sum(1 for s in final_samples if s.get('problem_type') == 'math')
    final_qa = sum(1 for s in final_samples if s.get('problem_type') == 'qa')
    final_code = sum(1 for s in final_samples if s.get('problem_type') == 'code')

    print(f"\n📊 最终分布:")
    print(f"  Math: {final_math} ({final_math/len(final_samples)*100:.2f}%)")
    print(f"  QA:   {final_qa} ({final_qa/len(final_samples)*100:.2f}%)")
    print(f"  Code: {final_code} ({final_code/len(final_samples)*100:.2f}%)")
    print(f"  总计: {len(final_samples)}")

    # 9. 验证所有Code样本格式完整
    print(f"\n🔍 验证Code样本完整性...")
    all_valid = all(is_valid_code_sample(s) for s in final_samples if s.get('problem_type') == 'code')
    if all_valid:
        print(f"✅ 所有Code样本格式完整")
    else:
        print(f"❌ 仍有格式不完整的Code样本")
        return

    # 10. 保存
    print(f"\n💾 保存增强训练集: {output_file}")
    save_jsonl(final_samples, output_file)
    print(f"✅ 保存完成!")

    # 11. 保存统计
    stats = {
        'original_total': len(samples),
        'invalid_code_removed': invalid_code_count,
        'valid_total': len(valid_samples),
        'humaneval_added': len(humaneval_samples),
        'final_total': len(final_samples),
        'final_math': final_math,
        'final_qa': final_qa,
        'final_code': final_code,
        'target_code_ratio': target_code_ratio,
        'actual_code_ratio': final_code / len(final_samples)
    }

    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"📊 统计信息: {stats_file}")

    print("\n" + "="*70)
    print("✅ 数据修复和增强完成!")
    print("="*70)

if __name__ == "__main__":
    random.seed(42)

    augment_training_data_clean(
        input_file="data/train/mixed_dataset.jsonl",
        output_file="data/train/mixed_dataset_augmented_v2.jsonl",
        humaneval_file="data/humaneval/humaneval_full.jsonl",
        target_code_ratio=0.10
    )
