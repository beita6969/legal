#!/usr/bin/env python3
"""
法律数据处理器 - 数据预处理和格式转换
Legal Data Processor - Data preprocessing and format conversion

支持:
- CAIL2018 格式转换
- DISC-Law-SFT 格式转换
- LegalBench 格式转换
- 训练数据生成
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class LegalDataProcessor:
    """法律数据处理器"""

    # 任务类型映射
    TASK_TYPE_MAP = {
        # 中国数据集
        'cail2018': 'case_prediction',
        'jec_qa': 'statute_qa',
        'disc_law': 'consultation',
        'refined_legal': 'case_prediction',
        # 美国数据集
        'legalbench': 'statute_qa',
        'casehold': 'case_prediction',
        'cuad': 'document_gen'
    }

    def __init__(self, data_dir: str = "data/legal"):
        self.data_dir = Path(data_dir)

    def process_cail2018(
        self,
        input_file: str,
        output_file: str = None,
        max_samples: int = 5000
    ) -> List[Dict]:
        """
        处理CAIL2018数据集

        CAIL2018格式:
        {
            "fact": "经审理查明...",
            "meta": {
                "accusation": ["盗窃"],
                "relevant_articles": [264],
                "punish_of_money": 0,
                "criminals": ["张三"],
                "term_of_imprisonment": {...}
            }
        }
        """
        samples = []
        input_path = Path(input_file)

        if not input_path.exists():
            print(f"❌ 文件不存在: {input_file}")
            return samples

        print(f"📖 处理CAIL2018数据: {input_file}")

        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break

                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                    meta = item.get('meta', {})

                    # 构建ground_truth
                    charges = meta.get('accusation', [])
                    articles = meta.get('relevant_articles', [])
                    term = meta.get('term_of_imprisonment', {})

                    # 计算刑期（月）
                    imprisonment_months = (
                        term.get('imprisonment', 0) +
                        term.get('death_penalty', 0) * 999 +
                        term.get('life_imprisonment', 0) * 999
                    )

                    sample = {
                        'id': f"cn_cail_{i}",
                        'jurisdiction': 'CN',
                        'problem': f"根据以下案情事实，分析被告人应当承担的刑事责任：\n\n{item.get('fact', '')}",
                        'problem_type': 'legal',
                        'task_type': 'case_prediction',
                        'source': 'cail2018',
                        'ground_truth': {
                            'charges': charges,
                            'articles': [f"刑法第{a}条" for a in articles],
                            'sentence': {
                                'imprisonment_months': imprisonment_months,
                                'fine': meta.get('punish_of_money', 0)
                            }
                        },
                        'legal_domain': 'criminal',
                        'difficulty': self._estimate_difficulty(item)
                    }
                    samples.append(sample)

                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON解析错误 行{i}: {e}")
                    continue

        print(f"✅ 处理完成: {len(samples)} 样本")

        if output_file:
            self._save_jsonl(samples, output_file)

        return samples

    def process_disc_law(
        self,
        input_file: str,
        output_file: str = None,
        max_samples: int = 5000
    ) -> List[Dict]:
        """
        处理DISC-Law-SFT数据集

        DISC格式:
        {
            "input": "问题",
            "output": "答案",
            "type": "类型"
        }
        """
        samples = []
        input_path = Path(input_file)

        if not input_path.exists():
            print(f"❌ 文件不存在: {input_file}")
            return samples

        print(f"📖 处理DISC-Law数据: {input_file}")

        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break

                if not line.strip():
                    continue

                try:
                    item = json.loads(line)

                    # 判断任务类型
                    disc_type = item.get('type', 'qa')
                    if '文书' in disc_type or 'document' in disc_type.lower():
                        task_type = 'document_gen'
                    elif '案例' in disc_type or 'case' in disc_type.lower():
                        task_type = 'case_prediction'
                    else:
                        task_type = 'consultation'

                    sample = {
                        'id': f"cn_disc_{i}",
                        'jurisdiction': 'CN',
                        'problem': item.get('input', ''),
                        'problem_type': 'legal',
                        'task_type': task_type,
                        'source': 'disc_law',
                        'ground_truth': {
                            'answer': item.get('output', '')
                        },
                        'legal_domain': self._detect_legal_domain(item.get('input', ''), 'CN'),
                        'difficulty': 'medium'
                    }
                    samples.append(sample)

                except json.JSONDecodeError:
                    continue

        print(f"✅ 处理完成: {len(samples)} 样本")

        if output_file:
            self._save_jsonl(samples, output_file)

        return samples

    def process_legalbench(
        self,
        input_file: str,
        output_file: str = None,
        max_samples: int = 5000
    ) -> List[Dict]:
        """
        处理LegalBench数据集 (US)

        LegalBench格式:
        {
            "text": "问题文本",
            "label": "标签/答案",
            "task": "任务名称"
        }
        """
        samples = []
        input_path = Path(input_file)

        if not input_path.exists():
            print(f"❌ 文件不存在: {input_file}")
            return samples

        print(f"📖 处理LegalBench数据: {input_file}")

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f) if input_file.endswith('.json') else [
                json.loads(line) for line in f if line.strip()
            ]

        for i, item in enumerate(data[:max_samples]):
            task_name = item.get('task', 'general')

            # 映射任务类型
            if 'contract' in task_name.lower():
                task_type = 'document_gen'
                legal_domain = 'contract'
            elif 'case' in task_name.lower() or 'holding' in task_name.lower():
                task_type = 'case_prediction'
                legal_domain = 'civil'
            else:
                task_type = 'statute_qa'
                legal_domain = self._detect_legal_domain(item.get('text', ''), 'US')

            sample = {
                'id': f"us_legalbench_{i}",
                'jurisdiction': 'US',
                'problem': item.get('text', item.get('question', '')),
                'problem_type': 'legal',
                'task_type': task_type,
                'source': 'legalbench',
                'ground_truth': {
                    'answer': str(item.get('label', item.get('answer', '')))
                },
                'legal_domain': legal_domain,
                'difficulty': 'medium',
                'original_task': task_name
            }
            samples.append(sample)

        print(f"✅ 处理完成: {len(samples)} 样本")

        if output_file:
            self._save_jsonl(samples, output_file)

        return samples

    def process_casehold(
        self,
        input_file: str,
        output_file: str = None,
        max_samples: int = 5000
    ) -> List[Dict]:
        """
        处理CaseHOLD数据集 (US)

        CaseHOLD格式:
        {
            "citing_prompt": "引用上下文",
            "holding_0": "选项0",
            "holding_1": "选项1",
            ...,
            "label": 正确选项索引
        }
        """
        samples = []
        input_path = Path(input_file)

        if not input_path.exists():
            print(f"❌ 文件不存在: {input_file}")
            return samples

        print(f"📖 处理CaseHOLD数据: {input_file}")

        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break

                if not line.strip():
                    continue

                try:
                    item = json.loads(line)

                    # 构建问题
                    prompt = item.get('citing_prompt', '')
                    holdings = [item.get(f'holding_{j}', '') for j in range(5)]
                    label = item.get('label', 0)

                    problem = f"{prompt}\n\nWhich holding is correct?\n"
                    for j, h in enumerate(holdings):
                        if h:
                            problem += f"{j}. {h}\n"

                    sample = {
                        'id': f"us_casehold_{i}",
                        'jurisdiction': 'US',
                        'problem': problem,
                        'problem_type': 'legal',
                        'task_type': 'case_prediction',
                        'source': 'casehold',
                        'ground_truth': {
                            'correct_index': label,
                            'correct_holding': holdings[label] if label < len(holdings) else ''
                        },
                        'legal_domain': 'civil',
                        'difficulty': 'hard'
                    }
                    samples.append(sample)

                except json.JSONDecodeError:
                    continue

        print(f"✅ 处理完成: {len(samples)} 样本")

        if output_file:
            self._save_jsonl(samples, output_file)

        return samples

    def create_training_dataset(
        self,
        cn_samples: List[Dict],
        us_samples: List[Dict],
        output_file: str,
        cn_ratio: float = 0.5,
        task_type_ratios: Dict[str, float] = None
    ) -> List[Dict]:
        """
        创建混合训练数据集

        Args:
            cn_samples: 中国法律样本
            us_samples: 美国法律样本
            output_file: 输出文件
            cn_ratio: 中国样本比例
            task_type_ratios: 任务类型比例
        """
        task_type_ratios = task_type_ratios or {
            'case_prediction': 0.35,
            'statute_qa': 0.25,
            'consultation': 0.25,
            'document_gen': 0.15
        }

        # 按任务类型分组
        cn_by_task = defaultdict(list)
        us_by_task = defaultdict(list)

        for s in cn_samples:
            cn_by_task[s.get('task_type', 'consultation')].append(s)
        for s in us_samples:
            us_by_task[s.get('task_type', 'consultation')].append(s)

        # 计算每种类型的样本数
        total_samples = len(cn_samples) + len(us_samples)
        cn_count = int(total_samples * cn_ratio)
        us_count = total_samples - cn_count

        final_samples = []

        # 按比例采样
        for task_type, ratio in task_type_ratios.items():
            cn_task_count = int(cn_count * ratio)
            us_task_count = int(us_count * ratio)

            cn_task_samples = cn_by_task.get(task_type, [])
            us_task_samples = us_by_task.get(task_type, [])

            if cn_task_samples:
                selected = random.sample(
                    cn_task_samples,
                    min(cn_task_count, len(cn_task_samples))
                )
                final_samples.extend(selected)

            if us_task_samples:
                selected = random.sample(
                    us_task_samples,
                    min(us_task_count, len(us_task_samples))
                )
                final_samples.extend(selected)

        # 打乱
        random.shuffle(final_samples)

        # 保存
        self._save_jsonl(final_samples, output_file)

        # 统计
        stats = self._compute_stats(final_samples)
        print(f"\n📊 训练数据集统计:")
        print(f"  总样本: {len(final_samples)}")
        print(f"  CN: {stats['jurisdiction']['CN']}, US: {stats['jurisdiction']['US']}")
        print(f"  任务类型分布: {stats['task_type']}")

        return final_samples

    def split_dataset(
        self,
        samples: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        output_dir: str = None
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """划分训练/验证/测试集"""
        random.shuffle(samples)

        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = samples[:train_end]
        val = samples[train_end:val_end]
        test = samples[val_end:]

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            self._save_jsonl(train, output_dir / "train.jsonl")
            self._save_jsonl(val, output_dir / "val.jsonl")
            self._save_jsonl(test, output_dir / "test.jsonl")

            print(f"\n💾 数据集已保存:")
            print(f"  训练集: {len(train)} 样本")
            print(f"  验证集: {len(val)} 样本")
            print(f"  测试集: {len(test)} 样本")

        return train, val, test

    def _estimate_difficulty(self, item: Dict) -> str:
        """估计样本难度"""
        meta = item.get('meta', {})
        charges = meta.get('accusation', [])
        articles = meta.get('relevant_articles', [])

        # 多罪名或多法条 -> 困难
        if len(charges) > 1 or len(articles) > 2:
            return 'hard'
        elif len(articles) > 1:
            return 'medium'
        else:
            return 'easy'

    def _detect_legal_domain(self, text: str, jurisdiction: str) -> str:
        """检测法律领域"""
        text_lower = text.lower()

        if jurisdiction == "CN":
            if any(kw in text for kw in ['刑法', '犯罪', '盗窃', '故意', '罪']):
                return 'criminal'
            elif any(kw in text for kw in ['合同', '买卖', '借款', '债务']):
                return 'civil'
            elif any(kw in text for kw in ['行政', '处罚', '许可']):
                return 'administrative'
            elif any(kw in text for kw in ['劳动', '工资', '解雇', '辞退']):
                return 'labor'
        else:  # US
            if any(kw in text_lower for kw in ['criminal', 'crime', 'murder', 'theft', 'felony']):
                return 'criminal'
            elif any(kw in text_lower for kw in ['contract', 'agreement', 'breach']):
                return 'contract'
            elif any(kw in text_lower for kw in ['tort', 'negligence', 'injury', 'damages']):
                return 'tort'
            elif any(kw in text_lower for kw in ['constitution', 'amendment', 'rights']):
                return 'constitutional'

        return 'general'

    def _compute_stats(self, samples: List[Dict]) -> Dict:
        """计算数据集统计信息"""
        stats = {
            'jurisdiction': defaultdict(int),
            'task_type': defaultdict(int),
            'legal_domain': defaultdict(int),
            'source': defaultdict(int),
            'difficulty': defaultdict(int)
        }

        for s in samples:
            stats['jurisdiction'][s.get('jurisdiction', 'unknown')] += 1
            stats['task_type'][s.get('task_type', 'unknown')] += 1
            stats['legal_domain'][s.get('legal_domain', 'unknown')] += 1
            stats['source'][s.get('source', 'unknown')] += 1
            stats['difficulty'][s.get('difficulty', 'unknown')] += 1

        return {k: dict(v) for k, v in stats.items()}

    def _save_jsonl(self, data: List[Dict], output_file: str):
        """保存为JSONL格式"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"💾 保存: {output_path} ({len(data)} 样本)")


def test_processor():
    """测试数据处理器"""
    print("\n" + "="*60)
    print("🧪 测试法律数据处理器")
    print("="*60)

    processor = LegalDataProcessor(data_dir="data/legal")

    # 创建示例数据
    cn_sample = {
        'id': 'cn_test_001',
        'jurisdiction': 'CN',
        'problem': '被告人张某盗窃他人财物价值5000元，应如何定罪量刑？',
        'problem_type': 'legal',
        'task_type': 'case_prediction',
        'source': 'test',
        'ground_truth': {'charges': ['盗窃罪'], 'articles': ['刑法第264条']},
        'legal_domain': 'criminal',
        'difficulty': 'easy'
    }

    us_sample = {
        'id': 'us_test_001',
        'jurisdiction': 'US',
        'problem': 'Did the defendant breach the contract by failing to deliver goods?',
        'problem_type': 'legal',
        'task_type': 'statute_qa',
        'source': 'test',
        'ground_truth': {'answer': 'Yes, failure to deliver constitutes breach'},
        'legal_domain': 'contract',
        'difficulty': 'medium'
    }

    print(f"\n📋 示例CN样本: {cn_sample['id']}")
    print(f"📋 示例US样本: {us_sample['id']}")


if __name__ == "__main__":
    test_processor()
