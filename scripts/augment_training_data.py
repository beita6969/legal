#!/usr/bin/env python3
"""
增强训练数据：提升Code样本比例

当前: Code 0.09% (128/147432)
目标: Code 10%
方法: 重复Code样本
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

def augment_training_data(
    input_file: str,
    output_file: str,
    target_code_ratio: float = 0.10,
    add_humaneval: bool = True
):
    """
    增强训练数据

    Args:
        input_file: 原始训练集
        output_file: 增强后的训练集
        target_code_ratio: 目标Code样本比例 (默认10%)
        add_humaneval: 是否添加HumanEval数据
    """
    print("="*70)
    print("训练数据增强")
    print("="*70)

    # 1. 加载原始数据
    print(f"\n📂 加载原始训练集: {input_file}")
    samples = load_jsonl(input_file)
    print(f"✅ 加载完成: {len(samples)} 个样本")

    # 2. 分类统计
    math_samples = [s for s in samples if s.get('problem_type') == 'math']
    qa_samples = [s for s in samples if s.get('problem_type') == 'qa']
    code_samples = [s for s in samples if s.get('problem_type') == 'code']

    print(f"\n📊 原始分布:")
    print(f"  Math: {len(math_samples)} ({len(math_samples)/len(samples)*100:.2f}%)")
    print(f"  QA:   {len(qa_samples)} ({len(qa_samples)/len(samples)*100:.2f}%)")
    print(f"  Code: {len(code_samples)} ({len(code_samples)/len(samples)*100:.2f}%)")

    # 3. 添加HumanEval数据（如果存在）
    humaneval_samples = []
    if add_humaneval:
        humaneval_file = "data/humaneval/humaneval_full.jsonl"
        if Path(humaneval_file).exists():
            print(f"\n📥 加载HumanEval数据: {humaneval_file}")
            humaneval_raw = load_jsonl(humaneval_file)

            # 转换HumanEval格式为统一格式
            for hr in humaneval_raw:
                sample = {
                    'problem': hr.get('prompt', ''),
                    'problem_type': 'code',
                    'ground_truth': hr.get('canonical_solution', ''),
                    'entry_point': hr.get('entry_point', ''),
                    'test': hr.get('test', ''),
                    'task_id': hr.get('task_id', '')
                }
                humaneval_samples.append(sample)

            print(f"✅ HumanEval: {len(humaneval_samples)} 个样本")
            code_samples.extend(humaneval_samples)

    # 4. 计算需要的Code样本数
    total_non_code = len(math_samples) + len(qa_samples)
    # target_code_ratio = code / (code + non_code)
    # code = target_code_ratio * (code + non_code)
    # code = target_code_ratio * total
    # total = code / target_code_ratio
    # non_code = total - code = code / target_code_ratio - code = code * (1 - target_code_ratio) / target_code_ratio

    target_code_count = int(total_non_code * target_code_ratio / (1 - target_code_ratio))
    current_code_count = len(code_samples)

    print(f"\n🎯 目标Code样本数: {target_code_count}")
    print(f"   当前Code样本数: {current_code_count}")
    print(f"   需要增加: {target_code_count - current_code_count}")

    # 5. 重复Code样本达到目标
    if current_code_count < target_code_count:
        repetitions = target_code_count // current_code_count
        remainder = target_code_count % current_code_count

        print(f"\n🔄 重复策略:")
        print(f"   完整重复: {repetitions} 次")
        print(f"   额外样本: {remainder} 个")

        augmented_code_samples = code_samples * repetitions
        if remainder > 0:
            # 随机选择额外样本
            extra_samples = random.sample(code_samples, remainder)
            augmented_code_samples.extend(extra_samples)

        print(f"✅ 增强后Code样本: {len(augmented_code_samples)}")
    else:
        augmented_code_samples = code_samples
        print(f"✅ Code样本已足够，无需重复")

    # 6. 合并所有样本
    final_samples = math_samples + qa_samples + augmented_code_samples

    # 7. 打乱顺序
    print(f"\n🔀 打乱样本顺序...")
    random.shuffle(final_samples)

    # 8. 统计最终分布
    final_math = sum(1 for s in final_samples if s.get('problem_type') == 'math')
    final_qa = sum(1 for s in final_samples if s.get('problem_type') == 'qa')
    final_code = sum(1 for s in final_samples if s.get('problem_type') == 'code')

    print(f"\n📊 最终分布:")
    print(f"  Math: {final_math} ({final_math/len(final_samples)*100:.2f}%)")
    print(f"  QA:   {final_qa} ({final_qa/len(final_samples)*100:.2f}%)")
    print(f"  Code: {final_code} ({final_code/len(final_samples)*100:.2f}%)")
    print(f"  总计: {len(final_samples)}")

    # 9. 保存
    print(f"\n💾 保存增强训练集: {output_file}")
    save_jsonl(final_samples, output_file)
    print(f"✅ 保存完成!")

    # 10. 保存统计信息
    stats = {
        'original_total': len(samples),
        'original_math': len(math_samples),
        'original_qa': len(qa_samples),
        'original_code': len(code_samples) - len(humaneval_samples),
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
    print(f"📊 统计信息已保存: {stats_file}")

    print("\n" + "="*70)
    print("✅ 数据增强完成!")
    print("="*70)

if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    random.seed(42)

    augment_training_data(
        input_file="data/train/mixed_dataset.jsonl",
        output_file="data/train/mixed_dataset_augmented.jsonl",
        target_code_ratio=0.10,  # 目标10% Code样本
        add_humaneval=True  # 添加HumanEval数据
    )
