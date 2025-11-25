#!/usr/bin/env python3
"""验证LLM Judge实现的有效性"""

import re

def verify_llm_judge_parsing():
    """验证reward_computer.py中的5层解析策略"""

    # 模拟实际的LLM响应样本
    test_responses = [
        # 标准格式
        {
            "response": "<analysis>The model correctly calculated 25*4=100</analysis>\n<true_false>True</true_false>",
            "expected": True,
            "desc": "标准XML格式"
        },
        {
            "response": "<analysis>Wrong calculation</analysis>\n<true_false>False</true_false>",
            "expected": False,
            "desc": "标准XML格式-错误"
        },
        # 变体格式
        {
            "response": "The answers are equivalent.\ntrue_false: True",
            "expected": True,
            "desc": "冒号格式"
        },
        {
            "response": "**true_false**: False",
            "expected": False,
            "desc": "Markdown格式"
        },
        # 末尾格式
        {
            "response": "After comparing the model response with ground truth, they match.\nTrue",
            "expected": True,
            "desc": "末尾True"
        },
        {
            "response": "The calculation is incorrect. The model got 1000 but answer is 100.\nFalse",
            "expected": False,
            "desc": "末尾False"
        },
    ]

    print("="*60)
    print("验证LLM Judge解析实现")
    print("="*60)
    print()

    success_count = 0
    total_count = len(test_responses)

    for test in test_responses:
        result_text = test["response"]
        expected = test["expected"]
        desc = test["desc"]

        # 复制reward_computer.py的5层解析逻辑
        true_false_match = None

        # 层1: 标准XML
        true_false_match = re.search(
            r'<true_false>\s*(True|False)\s*</true_false>',
            result_text,
            re.IGNORECASE
        )

        # 层2: 冒号分隔
        if not true_false_match:
            true_false_match = re.search(
                r'<true_false>\s*:\s*(True|False)',
                result_text,
                re.IGNORECASE
            )

        # 层3: Markdown
        if not true_false_match:
            true_false_match = re.search(
                r'\*\*true_false\*\*\s*:?\s*(True|False)',
                result_text,
                re.IGNORECASE
            )

        # 层4: 简单key:value
        if not true_false_match:
            true_false_match = re.search(
                r'true_false\s*:?\s*(True|False)',
                result_text,
                re.IGNORECASE
            )

        # 层5: 末尾查找
        if not true_false_match:
            last_200_chars = result_text[-200:]
            true_false_match = re.search(
                r'\b(True|False)\b',
                last_200_chars,
                re.IGNORECASE
            )

        if true_false_match:
            verdict = true_false_match.group(1).lower() == "true"
            if verdict == expected:
                print(f"✓ {desc}")
                print(f"  响应: {result_text[:60]}...")
                print(f"  解析结果: {verdict} (预期: {expected})")
                success_count += 1
            else:
                print(f"✗ {desc}")
                print(f"  响应: {result_text[:60]}...")
                print(f"  解析结果: {verdict} (预期: {expected})")
                print(f"  错误: 解析结果不匹配预期")
        else:
            print(f"✗ {desc}")
            print(f"  响应: {result_text[:60]}...")
            print(f"  错误: 无法解析")

        print()

    print("="*60)
    print(f"结果: {success_count}/{total_count} ({success_count/total_count*100:.0f}%)")

    if success_count == total_count:
        print("✅ 所有测试通过！解析逻辑正确。")
    else:
        print(f"⚠️  {total_count - success_count}个测试失败")

    return success_count == total_count

def analyze_implementation():
    """分析实现复杂度"""
    print("\n" + "="*60)
    print("实现分析")
    print("="*60)
    print("""
你的实现其实很合理：

1. **核心逻辑简单**：
   - compute_reward() 只有几行：调用LLM Judge → 二元奖励 → 归一化
   - 真正的判断交给LLM，不是规则匹配

2. **5层解析不是过度设计**：
   - LLM输出格式不可控，必须处理变体
   - 每层对应一种常见格式
   - 这是鲁棒性要求，不是复杂性

3. **Prompt详细是必要的**：
   - 指导LLM处理格式差异（$30 vs 30）
   - 处理数学等价（1/2 vs 0.5）
   - 处理单位差异（10 meters vs 10）

4. **统计是为了监控**：
   - 追踪成功率
   - 发现解析问题
   - 调试用途

结论：你的实现是**生产级别**的，不是过度工程。
""")

if __name__ == "__main__":
    # 验证解析逻辑
    verify_llm_judge_parsing()

    # 分析实现
    analyze_implementation()
