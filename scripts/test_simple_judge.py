#!/usr/bin/env python3
"""简化的LLM Judge实现测试"""

import re
from typing import Tuple

def simple_llm_judge_parse(response: str) -> bool:
    """简化的LLM Judge解析 - 只提取True/False"""

    # 统一转小写处理
    text = response.lower()

    # 直接查找true/false
    if 'true' in text:
        # 确保不是"not true"或"false, not true"等否定形式
        # 检查true前面是否有否定词
        before_true = text[:text.index('true')]
        if 'not' not in before_true[-10:] and 'false' not in before_true[-20:]:
            return True

    # 如果没找到true或者true被否定，则返回False
    return False

def test_simple_judge():
    """测试简化的LLM Judge"""

    test_cases = [
        # 简单情况
        ("<true_false>True</true_false>", True),
        ("<true_false>False</true_false>", False),
        ("The answer is True", True),
        ("The answer is False", False),

        # 复杂情况
        ("After analysis, the result is True", True),
        ("The answers match. True", True),
        ("They are different. False", False),

        # 带分析的情况
        ("<analysis>The answers are equivalent</analysis>\n<true_false>True</true_false>", True),
        ("Analysis: Not matching\ntrue_false: False", False),

        # 边缘情况
        ("This is not true", False),  # 否定形式
        ("False, not true", False),   # 明确的False
    ]

    print("="*60)
    print("简化的LLM Judge解析测试")
    print("="*60)

    success = 0
    total = len(test_cases)

    for response, expected in test_cases:
        result = simple_llm_judge_parse(response)
        status = "✓" if result == expected else "✗"
        print(f"{status} Input: {response[:50]}...")
        print(f"  Expected: {expected}, Got: {result}")
        if result == expected:
            success += 1

    print(f"\n结果: {success}/{total} ({success/total*100:.1f}%)")
    return success == total

if __name__ == "__main__":
    test_simple_judge()
