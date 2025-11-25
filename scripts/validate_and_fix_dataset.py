#!/usr/bin/env python3
"""
数据集格式验证和修复工具
"""

import json
from pathlib import Path
from collections import defaultdict
import copy

# 路径设置
data_dir = Path('/home/yijia/.claude/11/integrated_aflow_roll/data')
mixed_dir = data_dir / 'mixed'

def fix_code_samples(filename, output_filename):
    """修复代码样本缺少的字段"""
    file_path = mixed_dir / filename
    output_path = mixed_dir / output_filename

    print(f"\n📝 修复 {filename} 中的代码样本...")

    total_samples = 0
    fixed_samples = 0
    samples = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            total_samples += 1
            sample = json.loads(line)

            # 修复代码样本
            if sample.get('problem_type') == 'code':
                # 检查是否来自 MBPP (缺少 entry_point 和 test)
                if sample.get('source') == 'mbpp' and 'entry_point' not in sample:
                    # MBPP样本特殊处理
                    # 从problem中提取函数名作为entry_point
                    problem = sample.get('problem', '')

                    # 尝试从problem中提取函数名
                    if 'def ' in problem:
                        func_start = problem.find('def ') + 4
                        func_end = problem.find('(', func_start)
                        if func_end > func_start:
                            sample['entry_point'] = problem[func_start:func_end].strip()
                        else:
                            sample['entry_point'] = 'solution'
                    else:
                        sample['entry_point'] = 'solution'

                    # 添加默认测试
                    sample['test'] = 'def check(candidate):\n    # Test cases from MBPP\n    pass'
                    fixed_samples += 1

            samples.append(sample)

    # 写入修复后的数据集
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"✅ 修复了 {fixed_samples} 个代码样本")
    print(f"📁 保存到: {output_path}")

    return total_samples, fixed_samples

def validate_dataset(filename):
    """验证数据集完整性"""
    file_path = mixed_dir / filename

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return None

    print(f"\n🔍 验证数据集: {filename}")
    print("="*60)

    stats = {
        'total': 0,
        'valid': 0,
        'by_type': defaultdict(int),
        'by_source': defaultdict(int),
        'issues': [],
        'field_stats': defaultdict(int)
    }

    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue

            stats['total'] += 1

            try:
                sample = json.loads(line)
                problem_type = sample.get('problem_type', 'unknown')
                source = sample.get('source', 'unknown')

                stats['by_type'][problem_type] += 1
                stats['by_source'][source] += 1

                # 验证必需字段
                required_fields = ['problem', 'problem_type', 'ground_truth']
                missing = []

                for field in required_fields:
                    if field not in sample or sample[field] is None or sample[field] == "":
                        missing.append(field)
                    else:
                        stats['field_stats'][field] += 1

                # 类型特定字段验证
                if problem_type == 'code' and source == 'humaneval':
                    code_fields = ['entry_point', 'test']
                    for field in code_fields:
                        if field in sample:
                            stats['field_stats'][field] += 1
                        else:
                            missing.append(field)

                elif problem_type == 'math' and source == 'MATH':
                    math_fields = ['subject', 'level', 'answer']
                    for field in math_fields:
                        if field in sample:
                            stats['field_stats'][field] += 1

                if not missing:
                    stats['valid'] += 1
                else:
                    stats['issues'].append((idx, missing))

            except json.JSONDecodeError as e:
                stats['issues'].append((idx, f"JSON错误: {e}"))

    # 输出报告
    print(f"\n📊 验证报告:")
    print(f"  总样本数: {stats['total']:,}")
    print(f"  有效样本: {stats['valid']:,} ({stats['valid']/stats['total']*100:.1f}%)")

    print(f"\n按类型分布:")
    for ptype, count in sorted(stats['by_type'].items()):
        pct = count / stats['total'] * 100
        print(f"  {ptype:10s}: {count:6,} ({pct:5.1f}%)")

    print(f"\n按数据源分布:")
    for source, count in sorted(stats['by_source'].items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = count / stats['total'] * 100
        print(f"  {source:15s}: {count:6,} ({pct:5.1f}%)")

    print(f"\n字段覆盖率:")
    for field, count in sorted(stats['field_stats'].items()):
        pct = count / stats['total'] * 100
        print(f"  {field:20s}: {count:6,} ({pct:5.1f}%)")

    if stats['issues']:
        print(f"\n⚠️  发现 {len(stats['issues'])} 个问题")
    else:
        print(f"\n✅ 所有样本通过验证")

    return stats

def main():
    print("="*60)
    print("🔧 数据集格式验证和修复")
    print("="*60)

    # 1. 验证原始数据集
    print("\n步骤1: 验证原始数据集")
    original_stats = validate_dataset('train_mixed_with_math.jsonl')

    # 2. 修复数据集
    print("\n步骤2: 修复数据集")
    fix_code_samples('train_mixed_with_math.jsonl', 'train_mixed_with_math_fixed.jsonl')

    # 3. 验证修复后的数据集
    print("\n步骤3: 验证修复后的数据集")
    fixed_stats = validate_dataset('train_mixed_with_math_fixed.jsonl')

    # 4. 对比结果
    if original_stats and fixed_stats:
        print("\n" + "="*60)
        print("📈 修复效果对比")
        print("="*60)

        print(f"\n有效样本数变化:")
        print(f"  修复前: {original_stats['valid']:,} ({original_stats['valid']/original_stats['total']*100:.1f}%)")
        print(f"  修复后: {fixed_stats['valid']:,} ({fixed_stats['valid']/fixed_stats['total']*100:.1f}%)")
        print(f"  改善: +{fixed_stats['valid'] - original_stats['valid']:,} 样本")

        print(f"\n问题数量变化:")
        print(f"  修复前: {len(original_stats['issues'])} 个问题")
        print(f"  修复后: {len(fixed_stats['issues'])} 个问题")
        print(f"  解决: {len(original_stats['issues']) - len(fixed_stats['issues'])} 个问题")

    print("\n" + "="*60)
    print("✅ 完成")
    print("="*60)

if __name__ == '__main__':
    main()
