#!/usr/bin/env python3
"""
法律领域模块 - 支持中国(CN)和美国(US)双法律体系

Legal Domain Module - Supporting China (CN) and United States (US) dual legal systems
"""

from .operators import (
    DirectAnswer,
    CaseLearning,
    StatuteLearning,
    Debate,
    LegalEnsemble,
    LegalRevise
)
from .retriever import LegalRetriever

__all__ = [
    # Operators
    'DirectAnswer',
    'CaseLearning',
    'StatuteLearning',
    'Debate',
    'LegalEnsemble',
    'LegalRevise',
    # Retriever
    'LegalRetriever',
]

# Supported jurisdictions
SUPPORTED_JURISDICTIONS = ['CN', 'US']

# Task types
TASK_TYPES = [
    'case_prediction',   # 案件分析/Case Analysis
    'statute_qa',        # 法条问答/Statute Q&A
    'document_gen',      # 法律文书/Legal Document
    'consultation'       # 法律咨询/Legal Consultation
]

# Legal domains
LEGAL_DOMAINS = {
    'CN': ['criminal', 'civil', 'administrative', 'commercial', 'labor'],
    'US': ['criminal', 'civil', 'constitutional', 'contract', 'tort', 'property']
}
