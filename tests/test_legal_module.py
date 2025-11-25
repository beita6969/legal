#!/usr/bin/env python3
"""
法律模块测试脚本
Test script for Legal Module
"""

import sys
import asyncio
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """测试导入"""
    print("\n" + "="*60)
    print("🧪 测试法律模块导入")
    print("="*60)

    try:
        from src.legal import (
            DirectAnswer, CaseLearning, StatuteLearning,
            Debate, LegalEnsemble, LegalRevise,
            LegalRetriever,
            SUPPORTED_JURISDICTIONS, TASK_TYPES, LEGAL_DOMAINS
        )
        print("✅ 所有法律Operator导入成功")
        print(f"   支持的管辖区: {SUPPORTED_JURISDICTIONS}")
        print(f"   任务类型: {TASK_TYPES}")
        print(f"   法律领域: {LEGAL_DOMAINS}")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_data_processor():
    """测试数据处理器"""
    print("\n" + "="*60)
    print("🧪 测试数据处理器")
    print("="*60)

    try:
        from src.legal.data_processor import LegalDataProcessor

        processor = LegalDataProcessor(data_dir="data/legal")

        # 测试检测法律领域
        cn_criminal = processor._detect_legal_domain("被告人盗窃他人财物", "CN")
        cn_civil = processor._detect_legal_domain("合同纠纷买卖", "CN")
        us_contract = processor._detect_legal_domain("contract breach agreement", "US")
        us_tort = processor._detect_legal_domain("negligence damages injury", "US")

        print(f"✅ CN 刑事检测: '{cn_criminal}'")
        print(f"✅ CN 民事检测: '{cn_civil}'")
        print(f"✅ US 合同检测: '{us_contract}'")
        print(f"✅ US 侵权检测: '{us_tort}'")

        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_reward_computer():
    """测试奖励计算器"""
    print("\n" + "="*60)
    print("🧪 测试奖励计算器")
    print("="*60)

    try:
        from src.legal.reward import LegalRewardComputer

        computer = LegalRewardComputer()

        # 测试法律引用提取
        cn_text = "根据《刑法》第264条和《民法典》第1165条的规定"
        us_text = "Under 18 U.S.C. § 1341 and citing Smith v. Jones"

        cn_citations = computer._extract_legal_citations(cn_text, "CN")
        us_citations = computer._extract_legal_citations(us_text, "US")

        print(f"✅ CN法律引用提取: {cn_citations}")
        print(f"✅ US法律引用提取: {us_citations}")

        # 测试奖励映射
        levels = [0.1, 0.35, 0.55, 0.75, 0.95]
        for score in levels:
            reward = computer._map_to_reward_level(score)
            print(f"   分数 {score:.2f} -> 奖励 {reward}")

        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_retriever_init():
    """测试检索器初始化"""
    print("\n" + "="*60)
    print("🧪 测试检索器初始化")
    print("="*60)

    try:
        from src.legal.retriever import LegalRetriever, FAISS_AVAILABLE, ST_AVAILABLE

        print(f"   FAISS 可用: {FAISS_AVAILABLE}")
        print(f"   SentenceTransformers 可用: {ST_AVAILABLE}")

        retriever = LegalRetriever(data_dir="data/legal")
        stats = retriever.get_stats()

        print(f"✅ 检索器创建成功")
        print(f"   统计: {stats}")

        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_data_manager_legal():
    """测试法律模式数据管理器"""
    print("\n" + "="*60)
    print("🧪 测试法律模式数据管理器")
    print("="*60)

    try:
        from src.data_manager import DataManager

        # 法律模式
        manager = DataManager(
            data_dir="data",
            domain_ratios={"legal_cn": 0.5, "legal_us": 0.5}
        )

        print(f"✅ 法律模式: {manager.legal_mode}")
        print(f"   域比例: {manager.domain_ratios}")
        print(f"   当前索引: {manager.current_indices}")

        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_workflow_generator_legal():
    """测试法律提示词生成"""
    print("\n" + "="*60)
    print("🧪 测试法律Workflow提示词生成")
    print("="*60)

    try:
        from src.rl_workflow_generator import RLWorkflowGenerator

        # 只测试提示词构建，不加载模型
        class MockGenerator:
            def _build_legal_prompt(self, problem, jurisdiction):
                return RLWorkflowGenerator._build_legal_prompt(self, problem, jurisdiction)

        gen = MockGenerator()

        cn_prompt = gen._build_legal_prompt("被告人盗窃价值5000元财物", "CN")
        us_prompt = gen._build_legal_prompt("Contract breach damages", "US")

        print(f"✅ CN法律提示词长度: {len(cn_prompt)} 字符")
        print(f"   包含关键词: {'CaseLearning' in cn_prompt and '案例学习' in cn_prompt}")

        print(f"✅ US法律提示词长度: {len(us_prompt)} 字符")
        print(f"   包含关键词: {'CaseLearning' in us_prompt and 'Bluebook' in us_prompt}")

        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("#  法律领域模块测试  #")
    print("#"*60)

    results = {
        "导入测试": test_imports(),
        "数据处理器": test_data_processor(),
        "奖励计算器": test_reward_computer(),
        "检索器初始化": test_retriever_init(),
        "数据管理器": test_data_manager_legal(),
        "提示词生成": test_workflow_generator_legal(),
    }

    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {name}: {status}")

    print(f"\n总计: {passed}/{total} 测试通过")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
