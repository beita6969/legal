#!/usr/bin/env python3
"""
法律领域 Operator 实现
Legal Domain Operators - Supporting CN/US dual systems

6个核心Operator:
1. DirectAnswer - 直接生成答案
2. CaseLearning - 案例学习
3. StatuteLearning - 法条学习
4. Debate - 辩论投票
5. LegalEnsemble - 集成选择
6. LegalRevise - 法律修订
"""

import asyncio
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


class BaseLegalOperator(ABC):
    """法律Operator基类"""

    def __init__(self, llm, retriever=None):
        """
        Args:
            llm: LLM实例 (如 GPT-4o-mini)
            retriever: 法律检索器 (FAISS)
        """
        self.llm = llm
        self.retriever = retriever

    @abstractmethod
    async def __call__(self, *args, **kwargs) -> Dict:
        pass

    def _get_system_prompt(self, jurisdiction: str) -> str:
        """获取分国家的系统提示词"""
        if jurisdiction == "CN":
            return """你是一位专业的中国法律专家。
请基于中华人民共和国现行法律法规、司法解释和指导性案例进行分析。
引用法律时使用中文法律名称和条款编号，如"《刑法》第264条"。"""
        else:  # US
            return """You are a professional U.S. legal expert.
Analyze based on U.S. federal and state laws, regulations, and case precedents.
Cite laws using standard Bluebook format, e.g., "18 U.S.C. § 1341"."""


class DirectAnswer(BaseLegalOperator):
    """直接生成答案 - 最简单的Operator"""

    async def __call__(
        self,
        question: str,
        context: str = "",
        jurisdiction: str = "CN"
    ) -> Dict:
        """
        直接根据问题和上下文生成法律答案

        Args:
            question: 法律问题
            context: 上下文信息（如案例学习记录、法条等）
            jurisdiction: 管辖区 "CN" 或 "US"

        Returns:
            {"answer": str, "confidence": float, "jurisdiction": str}
        """
        system_prompt = self._get_system_prompt(jurisdiction)

        if jurisdiction == "CN":
            user_prompt = f"""请回答以下法律问题：

问题：{question}

{f'参考信息：{context}' if context else ''}

请提供：
1. 直接、准确的法律分析
2. 相关的法律依据
3. 明确的结论"""
        else:
            user_prompt = f"""Please answer the following legal question:

Question: {question}

{f'Reference: {context}' if context else ''}

Provide:
1. Direct and accurate legal analysis
2. Relevant legal authorities
3. Clear conclusion"""

        response = await self.llm.aask(
            msg=user_prompt,
            system_msgs=[system_prompt]
        )

        return {
            "answer": response,
            "confidence": 0.8,  # 可以通过后续分析调整
            "jurisdiction": jurisdiction
        }


class CaseLearning(BaseLegalOperator):
    """案例学习 - 检索并学习历史案例"""

    async def __call__(
        self,
        question: str,
        jurisdiction: str = "CN",
        top_k: int = 3,
        legal_domain: str = None
    ) -> Dict:
        """
        从案例库检索相关案例并生成学习记录

        CN: 检索指导性案例、典型案例
        US: 检索判例法（binding/persuasive precedents）

        Args:
            question: 法律问题
            jurisdiction: 管辖区
            top_k: 返回案例数量
            legal_domain: 法律领域（如criminal, civil）

        Returns:
            {
                "learning_record": str,
                "retrieved_cases": list,
                "key_insights": list,
                "precedent_type": str
            }
        """
        # 1. 检索相关案例
        retrieved_cases = []
        if self.retriever:
            retrieved_cases = await self.retriever.search_cases(
                query=question,
                jurisdiction=jurisdiction,
                top_k=top_k,
                legal_domain=legal_domain
            )

        # 2. 生成学习记录
        system_prompt = self._get_system_prompt(jurisdiction)

        if jurisdiction == "CN":
            cases_text = "\n\n".join([
                f"案例{i+1}：{c.get('case_type', '案例')}\n"
                f"案情：{c.get('facts', '')}\n"
                f"裁判要旨：{c.get('reasoning', '')}"
                for i, c in enumerate(retrieved_cases)
            ]) if retrieved_cases else "未检索到相关案例"

            user_prompt = f"""请分析以下法律问题，并从检索到的案例中提取关键启示：

问题：{question}

检索到的案例：
{cases_text}

请生成：
1. 案例学习记录（总结案例的关键法律要点）
2. 对当前问题的启示
3. 可参考的裁判思路"""
            precedent_type = "指导性案例" if any(
                c.get('case_type') == '指导性案例' for c in retrieved_cases
            ) else "普通案例"
        else:
            cases_text = "\n\n".join([
                f"Case {i+1}: {c.get('case_name', 'Unknown')}\n"
                f"Citation: {c.get('citation', '')}\n"
                f"Facts: {c.get('facts', '')}\n"
                f"Holding: {c.get('holding', '')}"
                for i, c in enumerate(retrieved_cases)
            ]) if retrieved_cases else "No relevant cases found"

            user_prompt = f"""Analyze the following legal question and extract key insights from retrieved cases:

Question: {question}

Retrieved Cases:
{cases_text}

Please generate:
1. Case learning record (summarize key legal points)
2. Insights for the current question
3. Applicable legal reasoning"""
            precedent_type = "binding" if any(
                c.get('precedent_type') == 'binding' for c in retrieved_cases
            ) else "persuasive"

        response = await self.llm.aask(
            msg=user_prompt,
            system_msgs=[system_prompt]
        )

        # 提取关键启示
        key_insights = self._extract_insights(response, jurisdiction)

        return {
            "learning_record": response,
            "retrieved_cases": retrieved_cases,
            "key_insights": key_insights,
            "precedent_type": precedent_type
        }

    def _extract_insights(self, response: str, jurisdiction: str) -> List[str]:
        """从响应中提取关键启示"""
        # 简单实现：按行分割，提取要点
        lines = response.split('\n')
        insights = []
        for line in lines:
            line = line.strip()
            if line and (line.startswith(('1.', '2.', '3.', '-', '•', '*')) or
                        '要点' in line or '启示' in line or 'insight' in line.lower()):
                insights.append(line)
        return insights[:5]  # 最多返回5条


class StatuteLearning(BaseLegalOperator):
    """法条学习 - 检索并学习相关法律条文"""

    async def __call__(
        self,
        question: str,
        jurisdiction: str = "CN",
        legal_domain: str = None,
        top_k: int = 5
    ) -> Dict:
        """
        从法条库检索相关法律条文并生成学习记录

        CN: 检索法律条文 + 司法解释
        US: 检索 US Code + CFR + State Codes

        Returns:
            {
                "learning_record": str,
                "retrieved_statutes": list,
                "applicable_rules": list
            }
        """
        # 1. 检索相关法条
        retrieved_statutes = []
        if self.retriever:
            retrieved_statutes = await self.retriever.search_statutes(
                query=question,
                jurisdiction=jurisdiction,
                top_k=top_k,
                legal_domain=legal_domain
            )

        # 2. 生成学习记录
        system_prompt = self._get_system_prompt(jurisdiction)

        if jurisdiction == "CN":
            statutes_text = "\n\n".join([
                f"《{s.get('law_name', '')}》{s.get('article_number', '')}：\n"
                f"{s.get('content', '')}\n"
                f"{'司法解释：' + s.get('interpretations', [{}])[0].get('content', '') if s.get('interpretations') else ''}"
                for s in retrieved_statutes
            ]) if retrieved_statutes else "未检索到相关法条"

            user_prompt = f"""请分析以下法律问题，并从检索到的法条中提取适用规则：

问题：{question}

相关法条：
{statutes_text}

请生成：
1. 法条学习记录（解释法条的含义和适用条件）
2. 对当前问题适用的具体规则
3. 法条之间的关联分析"""
        else:
            statutes_text = "\n\n".join([
                f"{s.get('code_name', '')} {s.get('title', '')} § {s.get('section', '')}:\n"
                f"{s.get('section_title', '')}\n"
                f"{s.get('content', '')}"
                for s in retrieved_statutes
            ]) if retrieved_statutes else "No relevant statutes found"

            user_prompt = f"""Analyze the following legal question and extract applicable rules from retrieved statutes:

Question: {question}

Relevant Statutes:
{statutes_text}

Please generate:
1. Statute learning record (explain meaning and applicability)
2. Specific rules applicable to the current question
3. Analysis of statutory relationships"""

        response = await self.llm.aask(
            msg=user_prompt,
            system_msgs=[system_prompt]
        )

        # 提取适���规则
        applicable_rules = self._extract_rules(response, jurisdiction)

        return {
            "learning_record": response,
            "retrieved_statutes": retrieved_statutes,
            "applicable_rules": applicable_rules
        }

    def _extract_rules(self, response: str, jurisdiction: str) -> List[str]:
        """从响应中提取适用规则"""
        lines = response.split('\n')
        rules = []
        for line in lines:
            line = line.strip()
            if line and ('第' in line and '条' in line or
                        '§' in line or 'U.S.C.' in line or
                        '规则' in line or 'rule' in line.lower()):
                rules.append(line)
        return rules[:5]


class Debate(BaseLegalOperator):
    """辩论投票 - 多视角辩论后投票决定"""

    async def __call__(
        self,
        question: str,
        context: str,
        jurisdiction: str = "CN",
        rounds: int = 2
    ) -> Dict:
        """
        让多个视角进行辩论，然后投票决定最终答案

        CN perspectives: ["原告视角", "被告视角", "法官视角"]
        US perspectives: ["Plaintiff", "Defendant", "Judge", "Jury"]

        Returns:
            {
                "final_answer": str,
                "debate_log": list,
                "vote_result": dict
            }
        """
        if jurisdiction == "CN":
            perspectives = ["原告视角", "被告视角", "法官视角"]
        else:
            perspectives = ["Plaintiff", "Defendant", "Judge"]

        debate_log = []
        system_prompt = self._get_system_prompt(jurisdiction)

        # 每轮辩论
        previous_arguments = ""
        for round_num in range(rounds):
            round_arguments = []

            for perspective in perspectives:
                if jurisdiction == "CN":
                    prompt = f"""你现在扮演{perspective}。

法律问题：{question}

背景信息：{context}

{f'之前的辩论记录：{previous_arguments}' if previous_arguments else ''}

请从{perspective}的立场，提出你的法律论点和依据。简洁明了，不超过200字。"""
                else:
                    prompt = f"""You are now acting as the {perspective}.

Legal Question: {question}

Background: {context}

{f'Previous debate: {previous_arguments}' if previous_arguments else ''}

Present your legal arguments from the {perspective}'s perspective. Be concise, under 200 words."""

                response = await self.llm.aask(
                    msg=prompt,
                    system_msgs=[system_prompt]
                )

                round_arguments.append({
                    "perspective": perspective,
                    "round": round_num + 1,
                    "argument": response
                })

            debate_log.extend(round_arguments)
            previous_arguments = "\n".join([
                f"{a['perspective']}: {a['argument']}"
                for a in round_arguments
            ])

        # 投票决定最终答案
        if jurisdiction == "CN":
            vote_prompt = f"""基于以下辩论记录，作为中立的法律专家，请做出最终裁决：

法律问题：{question}

辩论记录：
{chr(10).join([f"{a['perspective']}(第{a['round']}轮): {a['argument']}" for a in debate_log])}

请综合各方观点，给出：
1. 最终法律结论
2. 支持该结论的主要理由
3. 相关法律依据"""
        else:
            vote_prompt = f"""Based on the following debate, as a neutral legal expert, please make the final decision:

Legal Question: {question}

Debate Record:
{chr(10).join([f"{a['perspective']} (Round {a['round']}): {a['argument']}" for a in debate_log])}

Please provide:
1. Final legal conclusion
2. Main reasons supporting this conclusion
3. Relevant legal authorities"""

        final_answer = await self.llm.aask(
            msg=vote_prompt,
            system_msgs=[system_prompt]
        )

        return {
            "final_answer": final_answer,
            "debate_log": debate_log,
            "vote_result": {
                "rounds": rounds,
                "perspectives": perspectives,
                "total_arguments": len(debate_log)
            }
        }


class LegalEnsemble(BaseLegalOperator):
    """集成选择 - 并行生成多个答案，选择最优"""

    async def __call__(
        self,
        question: str,
        context: str,
        jurisdiction: str = "CN",
        num_candidates: int = 3
    ) -> Dict:
        """
        并行生成多个候选答案，通过评分选择最优

        Returns:
            {
                "final_answer": str,
                "all_candidates": list,
                "scores": list
            }
        """
        system_prompt = self._get_system_prompt(jurisdiction)

        # 并行生成候选答案
        tasks = []
        for i in range(num_candidates):
            if jurisdiction == "CN":
                prompt = f"""请回答以下法律问题（尝试{i+1}）：

问题：{question}

参考信息：{context}

请提供完整的法律分析和结论。"""
            else:
                prompt = f"""Answer the following legal question (attempt {i+1}):

Question: {question}

Reference: {context}

Provide complete legal analysis and conclusion."""

            tasks.append(self.llm.aask(
                msg=prompt,
                system_msgs=[system_prompt]
            ))

        candidates = await asyncio.gather(*tasks)

        # 评分选择
        if jurisdiction == "CN":
            score_prompt = f"""请评估以下{num_candidates}个法律回答的质量，选择最优答案：

问题：{question}

{chr(10).join([f'答案{i+1}：{c}' for i, c in enumerate(candidates)])}

评估标准：
1. 法律依据准确性
2. 逻辑推理质量
3. 答案完整性

请输出：
1. 最优答案编号（1-{num_candidates}）
2. 各答案评分（0-10）
3. 选择理由"""
        else:
            score_prompt = f"""Evaluate the following {num_candidates} legal answers and select the best:

Question: {question}

{chr(10).join([f'Answer {i+1}: {c}' for i, c in enumerate(candidates)])}

Criteria:
1. Legal authority accuracy
2. Reasoning quality
3. Answer completeness

Output:
1. Best answer number (1-{num_candidates})
2. Scores for each answer (0-10)
3. Selection rationale"""

        score_response = await self.llm.aask(
            msg=score_prompt,
            system_msgs=[system_prompt]
        )

        # 解析评分结果
        best_idx, scores = self._parse_scores(score_response, num_candidates)

        return {
            "final_answer": candidates[best_idx],
            "all_candidates": list(candidates),
            "scores": scores,
            "selection_rationale": score_response
        }

    def _parse_scores(self, response: str, num_candidates: int) -> tuple:
        """解析评分结果"""
        # 简单实现：尝试找到最优答案编号
        best_idx = 0
        scores = [5.0] * num_candidates

        for i in range(num_candidates):
            if f"答案{i+1}" in response and ("最优" in response or "best" in response.lower()):
                best_idx = i
                break
            if f"Answer {i+1}" in response and "best" in response.lower():
                best_idx = i
                break

        return best_idx, scores


class LegalRevise(BaseLegalOperator):
    """法律修订 - 检查并修订法律答案"""

    async def __call__(
        self,
        answer: str,
        question: str,
        jurisdiction: str = "CN",
        legal_context: str = ""
    ) -> Dict:
        """
        检查并修订法律答案

        CN: 检查法条引用格式、罪名表述、法律术语
        US: 检查判例引用格式（Bluebook）、法律术语准确性

        Returns:
            {
                "revised_answer": str,
                "revision_log": str,
                "issues_found": list
            }
        """
        system_prompt = self._get_system_prompt(jurisdiction)

        if jurisdiction == "CN":
            revise_prompt = f"""请检查并修订以下法律回答：

原问题：{question}

原答案：{answer}

{f'补充背景：{legal_context}' if legal_context else ''}

检查要点：
1. 法条引用格式是否正确（应为《法律名》第X条）
2. 罪名表述是否准确（参照刑法规定）
3. 法律术语是否专业规范
4. 逻辑推理是否完整
5. 结论是否明确

请输出：
1. 发现的问题列表
2. 修订后的完整答案"""
        else:
            revise_prompt = f"""Review and revise the following legal answer:

Original Question: {question}

Original Answer: {answer}

{f'Additional Context: {legal_context}' if legal_context else ''}

Check for:
1. Citation format (Bluebook compliance)
2. Legal terminology accuracy
3. Logical reasoning completeness
4. Clarity of conclusion
5. Proper legal authorities

Output:
1. List of issues found
2. Revised complete answer"""

        response = await self.llm.aask(
            msg=revise_prompt,
            system_msgs=[system_prompt]
        )

        # 提取修订日志和问题
        issues_found = self._extract_issues(response, jurisdiction)

        return {
            "revised_answer": response,
            "revision_log": response,
            "issues_found": issues_found
        }

    def _extract_issues(self, response: str, jurisdiction: str) -> List[str]:
        """提取发现的问题"""
        issues = []
        lines = response.split('\n')

        in_issues_section = False
        for line in lines:
            line = line.strip()
            if '问题' in line or 'issue' in line.lower() or '发现' in line:
                in_issues_section = True
                continue
            if in_issues_section and line.startswith(('-', '•', '*', '1.', '2.', '3.')):
                issues.append(line)
            if '修订' in line or 'revised' in line.lower():
                in_issues_section = False

        return issues[:10]
