#!/usr/bin/env python3
"""
Dataset-Specific Judge Prompt Loader
针对不同数据集加载专属的LLM Judge提示词
"""
import yaml
from pathlib import Path
from typing import Dict, Optional


class JudgePromptLoader:
    """加载和管理数据集专属的Judge Prompt"""

    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: judge_prompts.yaml的路径
        """
        if config_path is None:
            # 默认路径
            config_path = Path(__file__).parent.parent / "config" / "judge_prompts.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.global_config = self.config.get('global', {})
        self.dataset_mapping = self.config.get('dataset_mapping', {}).get('by_source', {})

    def _load_config(self) -> Dict:
        """加载YAML配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  无法加载Judge配置文件 {self.config_path}: {e}")
            return {}

    def get_judge_prompt(
        self,
        source: Optional[str] = None,
        problem_type: Optional[str] = None
    ) -> str:
        """
        获取数据集专属的Judge Prompt

        Args:
            source: 数据集来源（如'gsm8k', 'math', 'hotpotqa'）
            problem_type: 问题类型（如'math', 'code', 'qa'）- 用于fallback

        Returns:
            格式化后的Judge Prompt
        """
        # 1. 尝试根据source获取专属配置
        dataset_key = None
        if source:
            source_lower = source.lower()
            dataset_key = self.dataset_mapping.get(source_lower)

        # 2. 如果找到数据集配置
        if dataset_key and dataset_key in self.config:
            dataset_config = self.config[dataset_key]
            if dataset_config.get('enabled', True):
                prompt = dataset_config.get('judge_prompt', '')
                if prompt:
                    # 先注入output_format（替换{{output_format}}占位符）
                    output_format = self.global_config.get('output_format', '')
                    prompt = prompt.replace('{{output_format}}', output_format)
                    # 返回的prompt现在只包含{{problem}}, {{prediction}}, {{ground_truth}}
                    return prompt

        # 3. Fallback: 使用通用prompt（根据problem_type）
        return self._get_fallback_prompt(problem_type)

    def _get_fallback_prompt(self, problem_type: Optional[str]) -> str:
        """获取通用的Fallback Prompt"""
        output_format = self.global_config.get('output_format', '')

        # 根据问题类型返回基础prompt
        if problem_type == 'math':
            base_prompt = """You are a mathematical equivalence evaluator.

**Task**: Determine if the predicted answer is mathematically equivalent to the ground truth.

**Prediction**: {prediction}
**Ground Truth**: {ground_truth}

**Evaluation Steps**:
1. Extract final numerical answers from both texts
2. Normalize formats (remove units, standardize notation)
3. Compare values with tolerance 0.01
"""
        elif problem_type == 'qa':
            base_prompt = """You are an answer equivalence evaluator.

**Task**: Determine if the predicted answer is equivalent to the ground truth.

**Prediction**: {prediction}
**Ground Truth**: {ground_truth}

**Evaluation Steps**:
1. Normalize both answers (lowercase, remove articles/punctuation)
2. Check for exact match or substring containment
3. Allow common entity variations
"""
        else:
            # 通用prompt
            base_prompt = """You are a precise answer equivalence evaluator.

**Task**: Determine if the predicted answer is equivalent to the ground truth.

**Prediction**: {prediction}
**Ground Truth**: {ground_truth}

**Evaluation**: Compare the semantic meaning of both answers.
"""

        return base_prompt + "\n" + output_format

    def get_dataset_config(self, source: Optional[str]) -> Dict:
        """获取完整的数据集配置"""
        if not source:
            return {}

        source_lower = source.lower()
        dataset_key = self.dataset_mapping.get(source_lower)

        if dataset_key and dataset_key in self.config:
            return self.config[dataset_key]

        return {}

    def should_use_test_execution(self, source: Optional[str]) -> bool:
        """判断是否应该使用测试执行而非LLM Judge"""
        if not source:
            return False

        dataset_config = self.get_dataset_config(source)

        # 检查evaluation_method字段
        eval_method = dataset_config.get('evaluation_method', '')
        return eval_method == 'test_execution'

    def get_stats(self) -> Dict:
        """获取配置统计信息"""
        enabled_datasets = []
        disabled_datasets = []

        for dataset_key, config in self.config.items():
            if isinstance(config, dict) and 'enabled' in config:
                if config['enabled']:
                    enabled_datasets.append(dataset_key)
                else:
                    disabled_datasets.append(dataset_key)

        return {
            'total_datasets': len(enabled_datasets) + len(disabled_datasets),
            'enabled_datasets': enabled_datasets,
            'disabled_datasets': disabled_datasets,
            'dataset_mappings': self.dataset_mapping
        }


if __name__ == "__main__":
    # 测试代码
    loader = JudgePromptLoader()

    print("📋 Judge Prompt配置统计:")
    stats = loader.get_stats()
    print(f"  总数据集: {stats['total_datasets']}")
    print(f"  启用: {', '.join(stats['enabled_datasets'])}")
    print(f"  禁用: {', '.join(stats['disabled_datasets'])}")

    print("\n🔍 测试不同数据集的Prompt:")

    # 测试GSM8K
    prompt = loader.get_judge_prompt(source='gsm8k', problem_type='math')
    print(f"\n[GSM8K] Prompt长度: {len(prompt)} 字符")
    print(f"包含'####': {'####' in prompt}")
    print(f"包含'<<calc>>': {'<<calc>>' in prompt}")

    # 测试HotpotQA
    prompt = loader.get_judge_prompt(source='hotpotqa', problem_type='qa')
    print(f"\n[HotpotQA] Prompt长度: {len(prompt)} 字符")
    print(f"包含'PROHIBITION': {'PROHIBITION' in prompt}")
    print(f"包含'option letter': {'option letter' in prompt}")

    # 测试Code数据集
    should_execute = loader.should_use_test_execution('humaneval')
    print(f"\n[HumanEval] 是否使用测试执行: {should_execute}")
