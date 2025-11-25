#!/usr/bin/env python3
"""
法律检索器 - 基于FAISS的向量检索
Legal Retriever - FAISS-based vector search for CN/US legal systems

支持:
- 案例库检索 (Case retrieval)
- 法条库检索 (Statute retrieval)
- 中美分离索引 (Separate indices for CN/US)
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS not installed. Run: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Run: pip install sentence-transformers")


class LegalRetriever:
    """法律检索器 - 支持CN/US双系统"""

    # 默认嵌入模型
    DEFAULT_EMBEDDING_MODELS = {
        'CN': 'shibing624/text2vec-base-chinese',  # 中文模型
        'US': 'all-MiniLM-L6-v2'  # 英文模型
    }

    def __init__(
        self,
        data_dir: str = "data/legal",
        embedding_model_cn: str = None,
        embedding_model_us: str = None,
        device: str = "cpu"
    ):
        """
        Args:
            data_dir: 法律数据目录
            embedding_model_cn: 中文嵌入模型
            embedding_model_us: 英文嵌入模型
            device: 计算设备
        """
        self.data_dir = Path(data_dir)
        self.device = device

        # 嵌入模型
        self.embedding_models = {}
        self.embedding_model_names = {
            'CN': embedding_model_cn or self.DEFAULT_EMBEDDING_MODELS['CN'],
            'US': embedding_model_us or self.DEFAULT_EMBEDDING_MODELS['US']
        }

        # FAISS索引
        self.case_indices = {'CN': None, 'US': None}
        self.statute_indices = {'CN': None, 'US': None}

        # 原始数据
        self.cases = {'CN': [], 'US': []}
        self.statutes = {'CN': [], 'US': []}

        # 索引状态
        self.initialized = {'CN': False, 'US': False}

    def _get_embedding_model(self, jurisdiction: str) -> 'SentenceTransformer':
        """获取或加载嵌入模型"""
        if not ST_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed")

        if jurisdiction not in self.embedding_models:
            model_name = self.embedding_model_names[jurisdiction]
            print(f"📦 加载嵌入模型: {model_name}")
            self.embedding_models[jurisdiction] = SentenceTransformer(
                model_name,
                device=self.device
            )

        return self.embedding_models[jurisdiction]

    def initialize(self, jurisdictions: List[str] = None):
        """初始化检索器 - 加载数据和构建索引"""
        if not FAISS_AVAILABLE:
            print("❌ FAISS not available, retriever disabled")
            return

        jurisdictions = jurisdictions or ['CN', 'US']

        for jurisdiction in jurisdictions:
            if self.initialized[jurisdiction]:
                continue

            print(f"\n{'='*50}")
            print(f"📚 初始化 {jurisdiction} 法律检索器")
            print(f"{'='*50}")

            # 加载数据
            self._load_cases(jurisdiction)
            self._load_statutes(jurisdiction)

            # 构建索引
            self._build_case_index(jurisdiction)
            self._build_statute_index(jurisdiction)

            self.initialized[jurisdiction] = True
            print(f"✅ {jurisdiction} 检索器初始化完成")

    def _load_cases(self, jurisdiction: str):
        """加载案例数据"""
        case_dir = self.data_dir / jurisdiction.lower() / "cases"

        if not case_dir.exists():
            print(f"⚠️  案例目录不存在: {case_dir}")
            return

        for file_path in case_dir.glob("*.json*"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.suffix == '.jsonl':
                        for line in f:
                            if line.strip():
                                self.cases[jurisdiction].append(json.loads(line))
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.cases[jurisdiction].extend(data)
                        else:
                            self.cases[jurisdiction].append(data)
            except Exception as e:
                print(f"❌ 加载案例文件失败 {file_path}: {e}")

        print(f"📖 加载 {jurisdiction} 案例: {len(self.cases[jurisdiction])} 条")

    def _load_statutes(self, jurisdiction: str):
        """加载法条数据"""
        statute_dir = self.data_dir / jurisdiction.lower() / "statutes"

        if not statute_dir.exists():
            print(f"⚠️  法条目录不存在: {statute_dir}")
            return

        for file_path in statute_dir.glob("*.json*"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.suffix == '.jsonl':
                        for line in f:
                            if line.strip():
                                self.statutes[jurisdiction].append(json.loads(line))
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.statutes[jurisdiction].extend(data)
                        else:
                            self.statutes[jurisdiction].append(data)
            except Exception as e:
                print(f"❌ 加载法条文件失败 {file_path}: {e}")

        print(f"📜 加载 {jurisdiction} 法条: {len(self.statutes[jurisdiction])} 条")

    def _get_case_text(self, case: Dict, jurisdiction: str) -> str:
        """获取案例的文本表示用于嵌入"""
        if jurisdiction == "CN":
            return f"{case.get('facts', '')} {case.get('reasoning', '')} {case.get('verdict', {})}"
        else:
            return f"{case.get('facts', '')} {case.get('holding', '')} {case.get('reasoning', '')}"

    def _get_statute_text(self, statute: Dict, jurisdiction: str) -> str:
        """获取法条的文本表示用于嵌入"""
        if jurisdiction == "CN":
            base = f"{statute.get('law_name', '')} {statute.get('title', '')} {statute.get('content', '')}"
            # 添加司法解释
            interps = statute.get('interpretations', [])
            if interps:
                base += " " + " ".join([i.get('content', '') for i in interps[:2]])
            return base
        else:
            return f"{statute.get('code_name', '')} {statute.get('section_title', '')} {statute.get('content', '')}"

    def _build_case_index(self, jurisdiction: str):
        """构建案例FAISS索引"""
        cases = self.cases[jurisdiction]
        if not cases:
            print(f"⚠️  无 {jurisdiction} 案例数据，跳过索引构建")
            return

        print(f"🔨 构建 {jurisdiction} 案例索引...")

        # 获取嵌入
        model = self._get_embedding_model(jurisdiction)
        texts = [self._get_case_text(c, jurisdiction) for c in cases]
        embeddings = model.encode(texts, show_progress_bar=True)

        # 构建FAISS索引
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # 内积相似度

        # L2归一化后使用内积等价于余弦相似度
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))

        self.case_indices[jurisdiction] = index
        print(f"✅ {jurisdiction} 案例索引: {index.ntotal} 向量, 维度 {dimension}")

    def _build_statute_index(self, jurisdiction: str):
        """构建法条FAISS索引"""
        statutes = self.statutes[jurisdiction]
        if not statutes:
            print(f"⚠️  无 {jurisdiction} 法条数据，跳过索引构建")
            return

        print(f"🔨 构建 {jurisdiction} 法条索引...")

        model = self._get_embedding_model(jurisdiction)
        texts = [self._get_statute_text(s, jurisdiction) for s in statutes]
        embeddings = model.encode(texts, show_progress_bar=True)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)

        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))

        self.statute_indices[jurisdiction] = index
        print(f"✅ {jurisdiction} 法条索引: {index.ntotal} 向量, 维度 {dimension}")

    async def search_cases(
        self,
        query: str,
        jurisdiction: str = "CN",
        top_k: int = 3,
        legal_domain: str = None
    ) -> List[Dict]:
        """检索相关案例"""
        if not self.initialized.get(jurisdiction):
            print(f"⚠️  {jurisdiction} 检索器未初始化")
            return []

        index = self.case_indices.get(jurisdiction)
        if index is None or index.ntotal == 0:
            return []

        # 编码查询
        model = self._get_embedding_model(jurisdiction)
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)

        # 搜索
        scores, indices = index.search(query_embedding.astype('float32'), top_k * 2)

        # 过滤和排序
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.cases[jurisdiction]):
                continue

            case = self.cases[jurisdiction][idx]

            # 领域过滤
            if legal_domain:
                case_domain = case.get('legal_domain', '')
                if legal_domain.lower() not in case_domain.lower():
                    continue

            results.append({
                **case,
                'relevance_score': float(score)
            })

            if len(results) >= top_k:
                break

        return results

    async def search_statutes(
        self,
        query: str,
        jurisdiction: str = "CN",
        top_k: int = 5,
        legal_domain: str = None
    ) -> List[Dict]:
        """检索相关法条"""
        if not self.initialized.get(jurisdiction):
            print(f"⚠️  {jurisdiction} 检索器未初始化")
            return []

        index = self.statute_indices.get(jurisdiction)
        if index is None or index.ntotal == 0:
            return []

        model = self._get_embedding_model(jurisdiction)
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = index.search(query_embedding.astype('float32'), top_k * 2)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.statutes[jurisdiction]):
                continue

            statute = self.statutes[jurisdiction][idx]

            if legal_domain:
                statute_domain = statute.get('legal_domain', '')
                if legal_domain.lower() not in statute_domain.lower():
                    continue

            results.append({
                **statute,
                'relevance_score': float(score)
            })

            if len(results) >= top_k:
                break

        return results

    def save_indices(self, output_dir: str = None):
        """保存FAISS索引到磁盘"""
        output_dir = Path(output_dir or self.data_dir / "indices")
        output_dir.mkdir(parents=True, exist_ok=True)

        for jurisdiction in ['CN', 'US']:
            if self.case_indices[jurisdiction] and self.case_indices[jurisdiction].ntotal > 0:
                faiss.write_index(
                    self.case_indices[jurisdiction],
                    str(output_dir / f"{jurisdiction.lower()}_cases.index")
                )
                print(f"💾 保存 {jurisdiction} 案例索引")

            if self.statute_indices[jurisdiction] and self.statute_indices[jurisdiction].ntotal > 0:
                faiss.write_index(
                    self.statute_indices[jurisdiction],
                    str(output_dir / f"{jurisdiction.lower()}_statutes.index")
                )
                print(f"💾 保存 {jurisdiction} 法条索引")

    def load_indices(self, input_dir: str = None):
        """从磁盘加载FAISS索引"""
        input_dir = Path(input_dir or self.data_dir / "indices")

        for jurisdiction in ['CN', 'US']:
            case_path = input_dir / f"{jurisdiction.lower()}_cases.index"
            statute_path = input_dir / f"{jurisdiction.lower()}_statutes.index"

            if case_path.exists():
                self.case_indices[jurisdiction] = faiss.read_index(str(case_path))
                print(f"📂 加载 {jurisdiction} 案例索引: {self.case_indices[jurisdiction].ntotal} 向量")

            if statute_path.exists():
                self.statute_indices[jurisdiction] = faiss.read_index(str(statute_path))
                print(f"📂 加载 {jurisdiction} 法条索引: {self.statute_indices[jurisdiction].ntotal} 向量")

    def get_stats(self) -> Dict:
        """获取检索器统计信息"""
        return {
            'CN': {
                'cases': len(self.cases['CN']),
                'statutes': len(self.statutes['CN']),
                'case_index_size': self.case_indices['CN'].ntotal if self.case_indices['CN'] else 0,
                'statute_index_size': self.statute_indices['CN'].ntotal if self.statute_indices['CN'] else 0,
                'initialized': self.initialized['CN']
            },
            'US': {
                'cases': len(self.cases['US']),
                'statutes': len(self.statutes['US']),
                'case_index_size': self.case_indices['US'].ntotal if self.case_indices['US'] else 0,
                'statute_index_size': self.statute_indices['US'].ntotal if self.statute_indices['US'] else 0,
                'initialized': self.initialized['US']
            }
        }


def test_retriever():
    """测试检索器"""
    print("\n" + "="*60)
    print("🧪 测试法律检索器")
    print("="*60)

    retriever = LegalRetriever(data_dir="data/legal")
    print(f"\n检索器统计: {retriever.get_stats()}")

    # 尝试初始化（如果有数据）
    # retriever.initialize(['CN'])


if __name__ == "__main__":
    test_retriever()
