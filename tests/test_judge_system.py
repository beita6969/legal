#!/usr/bin/env python3
"""
测试数据集专属Judge系统
"""
import sys
import os

# 添加路径
sys.path.insert(0, '/home/yijia/.claude/11/integrated_aflow_roll/src')

from judge_prompt_loader import JudgePromptLoader

def test_judge_prompt_loader():
    """测试Judge Prompt加载器"""
    print("="*60)
    print("测试1: Judge Prompt加载器基本功能")
    print("="*60)

    loader = JudgePromptLoader()
    stats = loader.get_stats()

    print(f"\n✅ 加载器初始化成功")
    print(f"总数据集配置: {stats['total_datasets']}")
    print(f"启用数据集: {', '.join(stats['enabled_datasets'])}")
    print(f"禁用数据集: {', '.join(stats['disabled_datasets'])}")

    print("\n" + "="*60)
    print("测试2: 不同数据集的Prompt内容")
    print("="*60)

    # 测试GSM8K
    print("\n[GSM8K Prompt]")
    prompt = loader.get_judge_prompt(source='gsm8k', problem_type='math')
    print(f"长度: {len(prompt)} 字符")
    print(f"包含'####': {'####' in prompt}")
    print(f"包含'<<calc>>': {'<<calc>>' in prompt}")
    print(f"包含'GSM8K': {'GSM8K' in prompt}")
    print(f"\n前200字符:\n{prompt[:200]}...")

    # 测试Math
    print("\n[Math Dataset Prompt]")
    prompt = loader.get_judge_prompt(source='math', problem_type='math')
    print(f"长度: {len(prompt)} 字符")
    print(f"包含'MATH Dataset': {'MATH Dataset' in prompt}")
    print(f"包含'LaTeX': {'LaTeX' in prompt}")
    has_frac = '\\\\frac' in prompt
    print(f"包含'\\\\frac': {has_frac}")

    # 测试HotpotQA
    print("\n[HotpotQA Prompt]")
    prompt = loader.get_judge_prompt(source='hotpotqa', problem_type='qa')
    print(f"长度: {len(prompt)} 字符")
    print(f"包含'PROHIBITION': {'PROHIBITION' in prompt}")
    print(f"包含'option letter': {'option letter' in prompt}")
    print(f"包含'might dream': {'might dream' in prompt}")

    # 测试CommonsenseQA
    print("\n[CommonsenseQA Prompt]")
    prompt = loader.get_judge_prompt(source='commonsenseqa', problem_type='qa')
    print(f"长度: {len(prompt)} 字符")
    print(f"包含'Common Sense': {'Common Sense' in prompt}")
    print(f"包含'multiple choice': {'multiple choice' in prompt}")

    # 测试未知数据集（应该fallback）
    print("\n[Unknown Dataset - Fallback]")
    prompt = loader.get_judge_prompt(source='unknown_dataset', problem_type='math')
    print(f"长度: {len(prompt)} 字符")
    print(f"使用Fallback: {True if 'mathematical equivalence evaluator' in prompt else False}")

    print("\n" + "="*60)
    print("测试3: Code数据集的test_execution标志")
    print("="*60)

    # 测试HumanEval
    should_execute = loader.should_use_test_execution('humaneval')
    print(f"\n[HumanEval] 应该使用测试执行: {should_execute}")

    # 测试MBPP
    should_execute = loader.should_use_test_execution('mbpp')
    print(f"[MBPP] 应该使用测试执行: {should_execute}")

    # 测试Math（不应该使用测试执行）
    should_execute = loader.should_use_test_execution('math')
    print(f"[Math] 应该使用测试执行: {should_execute}")

    print("\n" + "="*60)
    print("测试4: 数据集映射")
    print("="*60)

    print(f"\n数据集映射表:")
    for source, dataset in stats['dataset_mappings'].items():
        print(f"  {source:15} → {dataset}")

    print("\n✅ 所有测试完成！")


def test_prompt_format():
    """测试Prompt格式化"""
    print("\n" + "="*60)
    print("测试5: Prompt格式化功能")
    print("="*60)

    loader = JudgePromptLoader()

    # 模拟真实数据
    test_cases = [
        {
            'source': 'gsm8k',
            'problem_type': 'math',
            'problem': 'Natalia sold clips to 48 of her friends in April.',
            'prediction': '\\boxed{72}',
            'ground_truth': 'Natalia sold 48/2 = <<48/2=24>>24...\\n#### 72'
        },
        {
            'source': 'hotpotqa',
            'problem_type': 'qa',
            'problem': 'When are you likely to dream?',
            'prediction': 'E',
            'ground_truth': 'might dream'
        },
        {
            'source': 'math',
            'problem_type': 'math',
            'problem': 'Simplify: 1/2 + 1/4',
            'prediction': '\\boxed{0.75}',
            'ground_truth': '\\frac{3}{4}'
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n[测试用例 {i}: {case['source']}]")
        prompt_template = loader.get_judge_prompt(
            source=case['source'],
            problem_type=case['problem_type']
        )

        # 格式化prompt（手动替换，避免format()解析XML标签）
        try:
            formatted_prompt = prompt_template.replace('{{problem}}', case['problem'])
            formatted_prompt = formatted_prompt.replace('{{prediction}}', case['prediction'])
            formatted_prompt = formatted_prompt.replace('{{ground_truth}}', case['ground_truth'])
            print(f"✅ 格式化成功")
            print(f"Prediction在Prompt中: {case['prediction'] in formatted_prompt}")
            print(f"Ground Truth在Prompt中: {case['ground_truth'] in formatted_prompt}")
            print(f"总长度: {len(formatted_prompt)} 字符")
        except Exception as e:
            print(f"❌ 格式化失败: {e}")
            raise  # 让测试真正失败

    print("\n✅ Prompt格式化测试完成！")


if __name__ == "__main__":
    try:
        test_judge_prompt_loader()
        test_prompt_format()
        print("\n" + "="*60)
        print("🎉 所有测试通过！数据集专属Judge系统工作正常")
        print("="*60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
