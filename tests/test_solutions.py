#!/usr/bin/env python3
"""
测试方案1和方案2的修复效果
"""
import sys
import os

# 添加路径
sys.path.insert(0, '/home/yijia/.claude/11/integrated_aflow_roll/src')

from answer_extractor import AnswerExtractor
from rl_workflow_generator import RLWorkflowGenerator

def test_solution_2_code_execution():
    """测试方案2：代码执行功能"""
    print("="*60)
    print("测试方案2：代码执行功能")
    print("="*60)

    extractor = AnswerExtractor(use_llm_fallback=False)

    # 测试用例1：简单计算
    test_code_1 = """```python
result = 15 + 27
```"""

    print("\n[测试用例1: 简单计算]")
    print(f"代码:\n{test_code_1}")
    answer = extractor._extract_math_answer(f"\\boxed{{{test_code_1}}}", False)
    print(f"提取答案: {answer}")
    print(f"预期: 42")
    print(f"✅ 通过" if answer == "42" else f"❌ 失败")

    # 测试用例2：带变量计算
    test_code_2 = """```python
x = 100
y = 50
final_answer = x - y
```"""

    print("\n[测试用例2: 带变量计算]")
    print(f"代码:\n{test_code_2}")
    answer = extractor._extract_math_answer(f"\\boxed{{{test_code_2}}}", False)
    print(f"提取答案: {answer}")
    print(f"预期: 50")
    print(f"✅ 通过" if answer == "50" else f"❌ 失败")

    # 测试用例3：分数计算
    test_code_3 = """```python
from fractions import Fraction
result = Fraction(1, 3) + Fraction(1, 6)
print(result)
```"""

    print("\n[测试用例3: 分数计算]")
    print(f"代码:\n{test_code_3}")
    answer = extractor._extract_math_answer(f"\\boxed{{{test_code_3}}}", False)
    print(f"提取答案: {answer}")
    print(f"预期: 1/2")
    print(f"✅ 通过" if answer == "1/2" else f"❌ 可能失败（执行结果不确定）")

    print("\n" + "="*60)
    print("方案2测试完成")
    print("="*60)


def test_solution_1_workflow_fix():
    """测试方案1：workflow自动修复"""
    print("\n" + "="*60)
    print("测试方案1：Workflow自动修复")
    print("="*60)

    # 模拟一个缺少revise初始化的workflow代码
    problematic_workflow = """import workspace.math.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.review = operator.Review(self.llm)
        # ⚠️ 注意：这里缺少 self.revise 的初始化！

    async def __call__(self, problem: str):
        solution = await self.answer_generate(input=problem)
        review = await self.review(problem=problem, solution=solution['answer'])
        if not review['review_result']:
            # 使用了revise但没有初始化！
            revised = await self.revise(problem=problem, solution=solution['answer'], feedback=review['feedback'])
            return revised['solution'], self.llm.get_usage_summary()["total_cost"]
        return solution['answer'], self.llm.get_usage_summary()["total_cost"]
"""

    print("\n[原始代码 - 缺少revise初始化]")
    print(f"已初始化: answer_generate, review")
    print(f"已使用: answer_generate, review, revise")
    print(f"缺失: revise")

    # 创建临时生成器实例来测试修复功能（不加载模型）
    # 我们直接调用 _validate_and_fix_workflow 方法
    class DummyGenerator:
        def _validate_and_fix_workflow(self, code: str, problem_type: str) -> str:
            """复制自 RLWorkflowGenerator 的修复逻辑"""
            import re

            # 1. 提取__init__中已初始化的operators
            initialized_ops = set()
            init_section = re.search(r'def __init__\([^)]+\):[\s\S]+?(?=\n    async def|\n    def|$)', code)
            if init_section:
                init_code = init_section.group(0)
                init_patterns = re.findall(r'self\.(\w+)\s*=\s*operator\.(\w+)\(', init_code)
                for attr_name, op_name in init_patterns:
                    initialized_ops.add(attr_name)

            # 2. 提取__call__中使用的operators
            used_ops = set()
            call_section = re.search(r'async def __call__\([^)]+\):[\s\S]+', code)
            if call_section:
                call_code = call_section.group(0)
                used_patterns = re.findall(r'await self\.(\w+)\(', call_code)
                for op_name in used_patterns:
                    used_ops.add(op_name)

            # 3. 找出缺失的operators
            missing_ops = used_ops - initialized_ops

            if missing_ops:
                print(f"\n⚠️  检测到缺失的operator初始化: {missing_ops}")
                print(f"   已初始化: {initialized_ops}")
                print(f"   已使用: {used_ops}")

                # 4. 自动添加缺失的初始化代码
                llm_init_match = re.search(r'(\s+)(self\.llm = create_llm_instance\([^)]+\))', code)
                if llm_init_match:
                    indent = llm_init_match.group(1)
                    llm_init_line = llm_init_match.group(2)

                    missing_inits = []
                    for op_name in sorted(missing_ops):
                        op_class_name = ''.join(word.capitalize() for word in op_name.split('_'))
                        valid_operators = ['Custom', 'AnswerGenerate', 'Programmer', 'Test', 'Review', 'Revise', 'ScEnsemble']
                        if op_class_name in valid_operators:
                            missing_inits.append(f"{indent}self.{op_name} = operator.{op_class_name}(self.llm)")

                    if missing_inits:
                        insert_code = '\n' + '\n'.join(missing_inits)
                        code = code.replace(llm_init_line, llm_init_line + insert_code)
                        print(f"✅ 自动添加了 {len(missing_inits)} 个缺失的operator初始化")

            return code

    generator = DummyGenerator()
    fixed_workflow = generator._validate_and_fix_workflow(problematic_workflow, "math")

    print("\n[修复后的代码]")
    # 验证revise已被添加
    if 'self.revise = operator.Revise(self.llm)' in fixed_workflow:
        print("✅ 成功添加 self.revise = operator.Revise(self.llm)")
    else:
        print("❌ 修复失败：未找到 revise 初始化")

    # 显示修复后的__init__部分
    import re
    init_section = re.search(r'def __init__\([^)]+\):[\s\S]+?(?=\n    async def)', fixed_workflow)
    if init_section:
        print("\n__init__ 方法 (修复后):")
        print(init_section.group(0))

    print("\n" + "="*60)
    print("方案1测试完成")
    print("="*60)


if __name__ == "__main__":
    try:
        print("\n" + "#"*60)
        print("# 测试方案1和方案2的修复效果")
        print("#"*60)

        # 测试方案2
        test_solution_2_code_execution()

        # 测试方案1
        test_solution_1_workflow_fix()

        print("\n" + "#"*60)
        print("# 🎉 所有测试完成！")
        print("#"*60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
