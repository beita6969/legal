#!/usr/bin/env python3
"""
法律任务奖励计算
Legal Task Reward Computation

多维度奖励:
- 法律依据准确性 (35%)
- 推理逻辑质量 (25%)
- 罪名/结论正确性 (20%)
- 答案完整性 (20%)

5档细粒度奖励: [0.0, 0.2, 0.4, 0.7, 1.0]
"""

import re
from typing import Dict, List, Optional, Any, Tuple


class LegalRewardComputer:
    """法律任务奖励计算器"""

    # 奖励权重
    WEIGHTS = {
        'legal_basis': 0.35,      # 法律依据准确性
        'reasoning': 0.25,         # 推理逻辑质量
        'conclusion': 0.20,        # 结论正确性
        'completeness': 0.20       # 答案完整性
    }

    # 5档奖励值
    REWARD_LEVELS = [0.0, 0.2, 0.4, 0.7, 1.0]

    def __init__(self, llm=None):
        """
        Args:
            llm: LLM实例用于LLM-as-Judge评估
        """
        self.llm = llm

    async def compute_reward(
        self,
        answer: str,
        ground_truth: Dict,
        task_type: str,
        jurisdiction: str = "CN",
        question: str = ""
    ) -> Tuple[float, Dict]:
        """
        计算法律任务奖励

        Args:
            answer: 模型生成的答案
            ground_truth: 标准答案/参考信息
            task_type: 任务类型
            jurisdiction: 管辖区
            question: 原始问题

        Returns:
            (reward, details): 奖励值和详细评分
        """
        # 计算各维度得分
        scores = {}

        # 1. 法律依据准确性
        scores['legal_basis'] = await self._evaluate_legal_basis(
            answer, ground_truth, jurisdiction
        )

        # 2. 推理逻辑质量
        scores['reasoning'] = await self._evaluate_reasoning(
            answer, question, jurisdiction
        )

        # 3. 结论正确性
        scores['conclusion'] = self._evaluate_conclusion(
            answer, ground_truth, task_type, jurisdiction
        )

        # 4. 答案完整性
        scores['completeness'] = self._evaluate_completeness(
            answer, task_type, jurisdiction
        )

        # 加权计算总分
        total_score = sum(
            scores[dim] * self.WEIGHTS[dim]
            for dim in scores
        )

        # 映射到5档奖励
        reward = self._map_to_reward_level(total_score)

        details = {
            'dimension_scores': scores,
            'weighted_total': total_score,
            'final_reward': reward,
            'jurisdiction': jurisdiction,
            'task_type': task_type
        }

        return reward, details

    async def _evaluate_legal_basis(
        self,
        answer: str,
        ground_truth: Dict,
        jurisdiction: str
    ) -> float:
        """
        评估法律依据准确性

        检查:
        - CN: 法条引用是否匹配 (《刑法》第264条)
        - US: 判例/法规引用是否正确 (18 U.S.C. § 1341)
        """
        gt_articles = ground_truth.get('articles', [])
        if not gt_articles:
            gt_articles = ground_truth.get('applicable_statutes', [])

        if not gt_articles:
            # 无标准答案参考，使用LLM评估
            if self.llm:
                return await self._llm_evaluate_legal_basis(answer, jurisdiction)
            return 0.5  # 默认中等分

        # 提取答案中的法律引用
        cited_articles = self._extract_legal_citations(answer, jurisdiction)

        if not cited_articles:
            return 0.0  # 无引用

        # 计算匹配度
        gt_set = set(str(a).lower() for a in gt_articles)
        cited_set = set(str(a).lower() for a in cited_articles)

        # Jaccard相似度
        intersection = len(gt_set & cited_set)
        union = len(gt_set | cited_set)

        if union == 0:
            return 0.0

        similarity = intersection / union

        # 精确匹配加分
        exact_matches = len([c for c in cited_articles if any(
            str(g).lower() in str(c).lower() or str(c).lower() in str(g).lower()
            for g in gt_articles
        )])

        bonus = min(0.2, exact_matches * 0.1)

        return min(1.0, similarity + bonus)

    async def _llm_evaluate_legal_basis(self, answer: str, jurisdiction: str) -> float:
        """使用LLM评估法律依据"""
        if not self.llm:
            return 0.5

        if jurisdiction == "CN":
            prompt = f"""请评估以下法律回答中的法律依据引用质量（0-10分）：

{answer}

评分标准：
- 10分：法条引用完全正确，格式规范（《法律名》第X条）
- 7分：引用基本正确，有小错误
- 4分：引用不完整或部分错误
- 0分：无引用或完全错误

只输出分数（0-10的整数）："""
        else:
            prompt = f"""Rate the legal citation quality in this answer (0-10):

{answer}

Criteria:
- 10: Perfect citations in Bluebook format
- 7: Mostly correct with minor errors
- 4: Incomplete or partially incorrect
- 0: No citations or completely wrong

Output only the score (integer 0-10):"""

        try:
            response = await self.llm.aask(msg=prompt)
            score = int(re.search(r'\d+', response).group())
            return min(1.0, score / 10.0)
        except:
            return 0.5

    async def _evaluate_reasoning(
        self,
        answer: str,
        question: str,
        jurisdiction: str
    ) -> float:
        """
        评估推理逻辑质量

        使用LLM-as-Judge或基于规则评估
        """
        # 基本检查
        length_score = min(1.0, len(answer) / 500)  # 至少500字符

        # 结构检查
        structure_score = 0.0
        if jurisdiction == "CN":
            structure_markers = ['首先', '其次', '因此', '综上', '根据', '依据', '本案']
        else:
            structure_markers = ['first', 'second', 'therefore', 'accordingly', 'pursuant', 'holding']

        found_markers = sum(1 for m in structure_markers if m.lower() in answer.lower())
        structure_score = min(1.0, found_markers / 3)

        # LLM评估（如果可用）
        if self.llm:
            llm_score = await self._llm_evaluate_reasoning(answer, question, jurisdiction)
            return 0.3 * length_score + 0.2 * structure_score + 0.5 * llm_score

        return 0.5 * length_score + 0.5 * structure_score

    async def _llm_evaluate_reasoning(self, answer: str, question: str, jurisdiction: str) -> float:
        """使用LLM评估推理质量"""
        if jurisdiction == "CN":
            prompt = f"""评估以下法律推理的逻辑质量（0-10分）：

问题：{question[:200]}

回答：{answer[:1000]}

评分标准：
- 逻辑连贯性
- 论证充分性
- 法律推理规范性

只输出分数："""
        else:
            prompt = f"""Rate the legal reasoning quality (0-10):

Question: {question[:200]}

Answer: {answer[:1000]}

Criteria:
- Logical coherence
- Argument sufficiency
- Legal reasoning standards

Output only the score:"""

        try:
            response = await self.llm.aask(msg=prompt)
            score = int(re.search(r'\d+', response).group())
            return min(1.0, score / 10.0)
        except:
            return 0.5

    def _evaluate_conclusion(
        self,
        answer: str,
        ground_truth: Dict,
        task_type: str,
        jurisdiction: str
    ) -> float:
        """
        评估结论正确性

        根据任务类型使用不同评估方法
        """
        if task_type == 'case_prediction':
            return self._evaluate_case_prediction(answer, ground_truth, jurisdiction)
        elif task_type == 'statute_qa':
            return self._evaluate_statute_qa(answer, ground_truth)
        elif task_type == 'document_gen':
            return self._evaluate_document_gen(answer, ground_truth)
        else:  # consultation
            return self._evaluate_consultation(answer, ground_truth)

    def _evaluate_case_prediction(
        self,
        answer: str,
        ground_truth: Dict,
        jurisdiction: str
    ) -> float:
        """
        评估案件预测结果

        CN: 罪名、法条、刑期匹配
        US: Holding匹配
        """
        score = 0.0

        if jurisdiction == "CN":
            # 罪名匹配
            gt_charges = ground_truth.get('charges', [])
            if gt_charges:
                charge_match = any(
                    charge in answer for charge in gt_charges
                )
                if charge_match:
                    score += 0.5

            # 法条匹配
            gt_articles = ground_truth.get('articles', [])
            if gt_articles:
                article_match = any(
                    str(art) in answer for art in gt_articles
                )
                if article_match:
                    score += 0.3

            # 量刑匹配（如有）
            sentence = ground_truth.get('sentence', {})
            if sentence:
                # 简化：只检查是否提及刑期
                if '年' in answer or '月' in answer or '有期徒刑' in answer:
                    score += 0.2

        else:  # US
            # Holding匹配
            gt_holding = ground_truth.get('correct_holding', '')
            gt_answer = ground_truth.get('answer', '')

            if gt_holding and gt_holding.lower() in answer.lower():
                score = 1.0
            elif gt_answer and gt_answer.lower() in answer.lower():
                score = 0.8
            else:
                # 部分匹配
                gt_text = gt_holding or gt_answer
                if gt_text:
                    gt_words = set(gt_text.lower().split())
                    answer_words = set(answer.lower().split())
                    overlap = len(gt_words & answer_words) / max(len(gt_words), 1)
                    score = overlap * 0.6

        return min(1.0, score)

    def _evaluate_statute_qa(self, answer: str, ground_truth: Dict) -> float:
        """评估法条问答结果"""
        gt_answer = ground_truth.get('answer', '')

        if not gt_answer:
            return 0.5

        # 简单文本匹配
        gt_lower = gt_answer.lower()
        answer_lower = answer.lower()

        if gt_lower in answer_lower:
            return 1.0

        # 关键词匹配
        gt_words = set(gt_lower.split())
        answer_words = set(answer_lower.split())
        overlap = len(gt_words & answer_words) / max(len(gt_words), 1)

        return min(1.0, overlap)

    def _evaluate_document_gen(self, answer: str, ground_truth: Dict) -> float:
        """评估法律文书生成结果"""
        # 检查文书基本要素
        required_elements = [
            '原告', '被告', '诉讼请求', '事实与理由',  # CN
            'plaintiff', 'defendant', 'prayer', 'facts'  # US
        ]

        found = sum(1 for e in required_elements if e.lower() in answer.lower())
        return min(1.0, found / 4)

    def _evaluate_consultation(self, answer: str, ground_truth: Dict) -> float:
        """评估法律咨询结果"""
        gt_answer = ground_truth.get('answer', '')

        if not gt_answer:
            # 无标准答案，检查答案质量
            if len(answer) > 200:
                return 0.6
            elif len(answer) > 100:
                return 0.4
            else:
                return 0.2

        # 语义相似度（简化版）
        gt_words = set(gt_answer.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(gt_words & answer_words) / max(len(gt_words), 1)

        return min(1.0, overlap * 1.2)

    def _evaluate_completeness(
        self,
        answer: str,
        task_type: str,
        jurisdiction: str
    ) -> float:
        """
        评估答案完整性

        检查是否包含必要的法律要素
        """
        completeness_checklist = {
            'case_prediction': {
                'CN': ['罪名', '法条', '量刑', '理由'],
                'US': ['charge', 'statute', 'holding', 'reasoning']
            },
            'statute_qa': {
                'CN': ['法条', '解释', '适用'],
                'US': ['statute', 'interpretation', 'application']
            },
            'document_gen': {
                'CN': ['原告', '被告', '请求', '理由', '证据'],
                'US': ['plaintiff', 'defendant', 'relief', 'facts', 'evidence']
            },
            'consultation': {
                'CN': ['建议', '依据', '风险'],
                'US': ['advice', 'authority', 'risk']
            }
        }

        checklist = completeness_checklist.get(task_type, {}).get(jurisdiction, [])

        if not checklist:
            return 0.5

        found = sum(1 for item in checklist if item.lower() in answer.lower())
        return found / len(checklist)

    def _extract_legal_citations(self, text: str, jurisdiction: str) -> List[str]:
        """提取法律引用"""
        citations = []

        if jurisdiction == "CN":
            # 匹配《xxx》第xxx条
            pattern = r'《[^》]+》[第]?\d+[条款]?'
            citations = re.findall(pattern, text)

            # 匹配 刑法第xxx条
            pattern2 = r'[刑民行诉][法典][第]\d+[条款]?'
            citations.extend(re.findall(pattern2, text))

        else:  # US
            # 匹配 X U.S.C. § XXXX
            pattern = r'\d+\s*U\.S\.C\.\s*§\s*\d+'
            citations = re.findall(pattern, text)

            # 匹配案例引用 XXX v. XXX
            pattern2 = r'[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+'
            citations.extend(re.findall(pattern2, text))

        return citations

    def _map_to_reward_level(self, score: float) -> float:
        """将分数映射到5档奖励"""
        if score >= 0.9:
            return 1.0
        elif score >= 0.7:
            return 0.7
        elif score >= 0.5:
            return 0.4
        elif score >= 0.3:
            return 0.2
        else:
            return 0.0


def test_reward_computer():
    """测试奖励计算器"""
    print("\n" + "="*60)
    print("🧪 测试法律奖励计算器")
    print("="*60)

    computer = LegalRewardComputer()

    # 测试CN案例
    cn_answer = """根据案情分析，被告人张某的行为构成盗窃罪。
    依据《刑法》第264条规定，盗窃公私财物，数额较大的，处三年以下有期徒刑。
    本案中，张某盗窃财物价值5000元，属于数额较大，建议判处有期徒刑一年。"""

    cn_gt = {
        'charges': ['盗窃罪'],
        'articles': ['刑法第264条'],
        'sentence': {'imprisonment_months': 12}
    }

    # 同步测试各维度
    legal_basis = computer._extract_legal_citations(cn_answer, "CN")
    print(f"\n📋 提取的法律引用: {legal_basis}")

    completeness = computer._evaluate_completeness(cn_answer, 'case_prediction', 'CN')
    print(f"📊 完整性评分: {completeness:.2f}")

    conclusion = computer._evaluate_case_prediction(cn_answer, cn_gt, 'CN')
    print(f"📊 结论评分: {conclusion:.2f}")


if __name__ == "__main__":
    test_reward_computer()
