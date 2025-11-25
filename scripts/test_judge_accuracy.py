#!/usr/bin/env python3
"""
LLM Judge 精度压力测试脚本
目的：通过构造一系列边界案例（陷阱），测试 LLM Judge 的判决标准是否合理。
"""
import sys
import os
import asyncio
from typing import List, Dict

# 添加 src 路径
sys.path.insert(0, '11/integrated_aflow_roll/src')

from reward_computer import RewardComputer

async def run_tests():
    print("⚖️  LLM Judge 精度压力测试")
    print("=" * 60)
    
    # 初始化，启用 debug logging 以便看到详细输出
    judge = RewardComputer(use_llm_judge=True, debug_logging=True)
    
    if not judge.llm_judge_client:
        print("❌ LLM Judge 初始化失败，无法进行测试。请检查 vLLM 服务。")
        return

    test_cases = [
        # --- 组 1: 格式与提取 (应为 True) ---
        {
            "problem": "What is 10 + 20?",
            "prediction": "The calculation is simple. 10+20=30. <answer>30</answer>",
            "ground_truth": "30",
            "type": "math",
            "expected": True,
            "desc": "标准 XML 标签提取"
        },
        {
            "problem": "Solve for x.",
            "prediction": "After solving, we get \\boxed{5}.",
            "ground_truth": "5",
            "type": "math",
            "expected": True,
            "desc": "LaTeX Boxed 格式"
        },
        {
            "problem": "Who won?",
            "prediction": "I think the winner is France.",
            "ground_truth": "France",
            "type": "qa",
            "expected": True,
            "desc": "自然语言包含"
        },

        # --- 组 2: 数值与精度 (应为 True) ---
        {
            "problem": "Calculate ratio.",
            "prediction": "0.5",
            "ground_truth": "1/2",
            "type": "math",
            "expected": True,
            "desc": "分数与小数等价"
        },
        {
            "problem": "Calculate cost.",
            "prediction": "$42.00",
            "ground_truth": "42",
            "type": "math",
            "expected": True,
            "desc": "货币符号与精度"
        },

        # --- 组 3: 错误陷阱 (应为 False) ---
        {
            "problem": "Calculate pi.",
            "prediction": "3.14",
            "ground_truth": "3.14159",
            "type": "math",
            "expected": False,
            "desc": "精度不足 (近似值)"
        },
        {
            "problem": "What is the unit?",
            "prediction": "10 kg",
            "ground_truth": "10 m",
            "type": "math",
            "expected": False,
            "desc": "单位错误"
        },
        {
            "problem": "True or False?",
            "prediction": "True",
            "ground_truth": "False",
            "type": "qa",
            "expected": False,
            "desc": "布尔值对立"
        },
        {
            "problem": "Who is the president?",
            "prediction": "Donald Trump",
            "ground_truth": "Joe Biden",
            "type": "qa",
            "expected": False,
            "desc": "实体错误"
        },
        
        # --- 组 4: 复杂语义 (挑战项) ---
        {
            "problem": "Explain the process.",
            "prediction": "First do A, then B.",
            "ground_truth": "Do B after A.",
            "type": "qa",
            "expected": True,
            "desc": "语义等价 (顺序描述)"
        }
    ]

    results = {"pass": 0, "fail": 0}
    
    for i, case in enumerate(test_cases):
        print(f"\n🔍 Case {i+1}: {case['desc']}")
        print(f"  Q: {case['problem']}")
        print(f"  Pred: {case['prediction']}")
        print(f"  GT:   {case['ground_truth']}")
        
        # 调用判决
        # 模拟 compute_reward 内部调用 _llm_judge_compare
        verdict = judge._llm_judge_compare(
            problem=case["problem"],
            prediction=case["prediction"],
            ground_truth=case["ground_truth"],
            problem_type=case["type"]
        )
        
        status = "✅ 通过" if verdict == case["expected"] else "❌ 失败"
        print(f"  结果: {verdict} (预期: {case['expected']}) -> {status}")
        
        if verdict == case["expected"]:
            results["pass"] += 1
        else:
            results["fail"] += 1

    print("\n" + "=" * 60)
    print(f"📊 测试总结: 通过 {results['pass']} / 总计 {len(test_cases)}")
    print(f"   通过率: {results['pass']/len(test_cases)*100:.1f}%")
    
    if results['fail'] > 0:
        print("\n⚠️  存在判决偏差，建议微调 Prompt 或 Temperature。")
    else:
        print("\n🎉 LLM Judge 表现完美，严谨度适中。")

if __name__ == "__main__":
    asyncio.run(run_tests())


