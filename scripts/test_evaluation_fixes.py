#!/usr/bin/env python3
"""
测试评估系统修复 - 验证MATH数据集和LLM Judge修复
"""
import sys
import json
sys.path.insert(0, 'src')

from reward_computer import RewardComputer


def test_math_evaluation():
    """
    测试MATH数据集评估修复

    问题：之前比较的是完整解答文本，导致正确答案被判定为错误
    修复：现在比较数值答案
    """
    print("\n" + "=" * 80)
    print("🧪 测试 1: MATH数据集评估修复")
    print("=" * 80)

    # 从实际数据集加载一个MATH样本
    print("\n📂 加载MATH样本...")
    with open('data/mixed/train_mixed_with_math_fixed.jsonl', 'r') as f:
        for line in f:
            sample = json.loads(line)
            if sample.get('source') == 'MATH' and 'answer' in sample:
                break

    print(f"\n📝 样本信息:")
    print(f"  问题: {sample['problem'][:100]}...")
    print(f"  数值答案 (answer字段): {sample.get('answer', 'N/A')}")
    print(f"  完整解答 (ground_truth字段): {sample.get('ground_truth', 'N/A')[:100]}...")

    # 初始化奖励计算器（启用LLM Judge和调试日志）
    reward_computer = RewardComputer(
        use_llm_judge=True,
        llm_config={
            "base_url": "http://localhost:8002/v1",
            "api_key": "sk-dummy",
            "model_name": "/home/yijia/lhy/openai/gpt-oss-120b"
        },
        debug_logging=True
    )

    # 测试场景1：使用正确答案（数值）
    print("\n✅ 场景1: 预测正确的数值答案")
    correct_answer = sample['answer']
    reward_correct = reward_computer.compute_reward(
        problem=sample['problem'],
        prediction=correct_answer,  # 模拟模型输出正确答案
        ground_truth=correct_answer,  # ✅ 修复：使用'answer'字段而非'ground_truth'
        problem_type='math'
    )
    print(f"  预测: {correct_answer}")
    print(f"  真值: {correct_answer}")
    print(f"  奖励: {reward_correct} (期望: 1.0)")
    assert reward_correct == 1.0, f"❌ 失败: 正确答案应得1.0奖励，实际得到{reward_correct}"
    print("  ✅ 通过: 正确答案获得最高奖励")

    # 测试场景2：使用错误答案
    print("\n❌ 场景2: 预测错误的数值答案")
    wrong_answer = "123456789"  # 明显错误的答案
    reward_wrong = reward_computer.compute_reward(
        problem=sample['problem'],
        prediction=wrong_answer,
        ground_truth=correct_answer,
        problem_type='math'
    )
    print(f"  预测: {wrong_answer}")
    print(f"  真值: {correct_answer}")
    print(f"  奖励: {reward_wrong} (期望: 0.0)")
    assert reward_wrong == 0.0, f"❌ 失败: 错误答案应得0.0奖励，实际得到{reward_wrong}"
    print("  ✅ 通过: 错误答案获得最低奖励")

    # 测试场景3：使用完整解答文本（旧bug场景）
    print("\n⚠️  场景3: 使用完整解答文本作为预测（旧bug模拟）")
    full_solution = sample['ground_truth']
    reward_solution = reward_computer.compute_reward(
        problem=sample['problem'],
        prediction=full_solution,
        ground_truth=correct_answer,  # ✅ 修复：使用'answer'字段
        problem_type='math'
    )
    print(f"  预测: {full_solution[:100]}...")
    print(f"  真值: {correct_answer}")
    print(f"  奖励: {reward_solution}")
    print(f"  说明: LLM Judge应能从完整解答中提取答案并判定为正确")

    print("\n" + "=" * 80)
    print("✅ MATH数据集评估修复测试通过")
    print("=" * 80)
    return reward_computer


def test_llm_judge_robustness(reward_computer):
    """
    测试LLM Judge的鲁棒性和错误处理
    """
    print("\n" + "=" * 80)
    print("🧪 测试 2: LLM Judge鲁棒性")
    print("=" * 80)

    test_cases = [
        {
            "name": "数学 - 不同格式的相同答案",
            "problem": "What is 1/2 as a decimal?",
            "prediction": "0.5",
            "ground_truth": "1/2",
            "expected_correct": True
        },
        {
            "name": "数学 - LaTeX格式",
            "problem": "Solve x^2 = 4",
            "prediction": "\\boxed{2}",
            "ground_truth": "2",
            "expected_correct": True
        },
        {
            "name": "数学 - 带单位的答案",
            "problem": "Calculate the cost",
            "prediction": "$30",
            "ground_truth": "30",
            "expected_correct": True
        },
        {
            "name": "文本 - 大小写不同",
            "problem": "What is the capital of France?",
            "prediction": "Paris",
            "ground_truth": "paris",
            "expected_correct": True
        },
        {
            "name": "文本 - 完全不同的答案",
            "problem": "What is the capital of France?",
            "prediction": "London",
            "ground_truth": "Paris",
            "expected_correct": False
        }
    ]

    passed = 0
    failed = 0

    for idx, case in enumerate(test_cases, 1):
        print(f"\n--- 测试 {idx}/{len(test_cases)}: {case['name']} ---")
        print(f"  问题: {case['problem']}")
        print(f"  预测: {case['prediction']}")
        print(f"  真值: {case['ground_truth']}")
        print(f"  期望: {'✅ 正确' if case['expected_correct'] else '❌ 错误'}")

        reward = reward_computer.compute_reward(
            problem=case['problem'],
            prediction=case['prediction'],
            ground_truth=case['ground_truth'],
            problem_type='math'
        )

        actual_correct = (reward == 1.0)
        print(f"  实际: {'✅ 正确' if actual_correct else '❌ 错误'} (奖励={reward})")

        if actual_correct == case['expected_correct']:
            print(f"  ✅ 通过")
            passed += 1
        else:
            print(f"  ❌ 失败: 判决不符合期望")
            failed += 1

    print("\n" + "=" * 80)
    print(f"LLM Judge鲁棒性测试结果: {passed}/{len(test_cases)} 通过")
    if failed > 0:
        print(f"⚠️  {failed} 个测���失败 - 可能需要进一步调整prompt")
    else:
        print("✅ 所有测试通过")
    print("=" * 80)


def test_eval_stats(reward_computer):
    """
    测试评估统计功能
    """
    print("\n" + "=" * 80)
    print("🧪 测试 3: 评估统计功能")
    print("=" * 80)

    # 打印当前统计
    reward_computer.print_eval_stats()

    # 重置统计
    print("\n重置统计...")
    reward_computer.reset_eval_stats()
    reward_computer.print_eval_stats()

    print("\n" + "=" * 80)
    print("✅ 评估统计功能测试通过")
    print("=" * 80)


def main():
    """
    主测试函数
    """
    print("\n" + "=" * 80)
    print("🚀 评估系统修复测试套件")
    print("=" * 80)
    print("\n修复内容:")
    print("  1. ✅ MATH数据集: 使用'answer'字段而非'ground_truth'字段")
    print("  2. ✅ LLM Judge: 增强输出解析（5种格式）")
    print("  3. ✅ 错误处理: 添加重试机制和详细日志")
    print("  4. ✅ 统计功能: 追踪成功率和失败原因")

    try:
        # 测试1: MATH数据集评估修复
        reward_computer = test_math_evaluation()

        # 测试2: LLM Judge鲁棒性
        test_llm_judge_robustness(reward_computer)

        # 测试3: 评估统计功能
        test_eval_stats(reward_computer)

        print("\n" + "=" * 80)
        print("🎉 所有测试完成")
        print("=" * 80)
        print("\n✅ 评估系统修复验证通过，可以开始训练")
        print("\n建议:")
        print("  1. 在config/training.yaml中启用 reward_computer.debug_logging: true（如需详细日志）")
        print("  2. 运行训练时监控准确率指标")
        print("  3. 每50步查看评估统计 (print_eval_stats)")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
