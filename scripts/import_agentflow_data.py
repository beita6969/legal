#!/usr/bin/env python3
"""
AgentFlow 数据导入工具
将 AgentFlow 项目使用的 NQ (Search) 和 DeepMath (Math) 数据集转换为本项目支持的格式，
并与本地的 HumanEval (Code) 数据集混合，生成高质量的训练数据。
"""
import os
import json
import random
import datasets
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# 配置
OUTPUT_DIR = "data/mixed"
OUTPUT_FILE = "train_agentflow_hybrid.jsonl"
HUMANEVAL_PATH = "data/humaneval/humaneval_full.jsonl"

# 采样数量配置 (根据 README 的 4:3:3 比例或 AgentFlow 的原始规模)
# AgentFlow 原始: ~180k total. 我们可能不需要这么多，或者全部利用。
# 建议：Math 20k, QA 20k, Code (HumanEval只有164个，需重复采样或寻找更多)
# 这里我们先全量加载，然后允许用户指定采样数
SAMPLE_COUNTS = {
    "math": 20000,  # DeepMath 很大，采样 20k
    "qa": 20000,    # NQ 很大，采样 20k
    "code": None    # None 表示使用全部本地数据 (HumanEval ~164)
}

def process_golden_answers(golden_answers, to_string=True):
    """
    复制自 AgentFlow: 处理 NQ 数据集的答案格式
    """
    items = []
    if isinstance(golden_answers, np.ndarray):
        items = [str(item) for item in golden_answers.flatten() if item is not None and pd.notna(item)]
    elif isinstance(golden_answers, (list, tuple)):
        items = [str(item) for item in golden_answers if item is not None and pd.notna(item)]
    elif isinstance(golden_answers, str):
        cleaned = golden_answers.strip()
        if cleaned:
            items = [cleaned]
    elif isinstance(golden_answers, (int, float, np.generic)):
        if not pd.isna(golden_answers):
            items = [str(golden_answers).strip()]
    elif golden_answers is None or (isinstance(golden_answers, str) and not golden_answers.strip()):
        items = []
    else:
        s = str(golden_answers).strip()
        if s and s != "nan":
            items = [s]

    if to_string:
        return "; ".join(items) if items else ""
    else:
        return items

def load_agentflow_qa() -> List[Dict]:
    """加载 NQ 数据集 (QA/Search)"""
    print("📥 下载/加载 NQ 数据集 (AgentFlow Search Source)...")
    try:
        # 使用 AgentFlow 同款数据集
        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq', split='train')
        
        processed = []
        for item in tqdm(dataset, desc="Processing NQ"):
            question = item.get("question", "").strip()
            if question and not question.endswith('?'):
                question += '?'
            
            golden_answers = item.get("golden_answers", [])
            final_result = process_golden_answers(golden_answers, to_string=True)
            
            if not final_result:
                continue

            processed.append({
                "problem": question,
                "problem_type": "qa",  # 映射为我们的 qa 类型
                "ground_truth": final_result,
                "source": "agentflow_nq",
                "id": f"nq_{item.get('id', random.randint(0, 999999))}"
            })
        
        print(f"✅ 加载 NQ 数据: {len(processed)} 条")
        return processed
    except Exception as e:
        print(f"❌ 加载 NQ 失败: {e}")
        return []

def load_agentflow_math() -> List[Dict]:
    """加载 DeepMath 数据集 (Math)"""
    print("📥 下载/加载 DeepMath 数据集 (AgentFlow Math Source)...")
    try:
        # 使用 AgentFlow 同款数据集
        dataset = datasets.load_dataset('zwhe99/DeepMath-103K', split='train')
        
        processed = []
        for idx, item in enumerate(tqdm(dataset, desc="Processing DeepMath")):
            question = item.get('question') or item.get('Problem')
            solution = item.get('final_answer') or item.get('Answer')
            
            if not question or not solution:
                continue

            processed.append({
                "problem": question,
                "problem_type": "math",  # 映射为我们的 math 类型
                "ground_truth": str(solution),
                "source": "agentflow_deepmath",
                "id": f"math_{idx}"
            })
            
        print(f"✅ 加载 DeepMath 数据: {len(processed)} 条")
        return processed
    except Exception as e:
        print(f"❌ 加载 DeepMath 失败: {e}")
        return []

def load_local_code() -> List[Dict]:
    """加载本地 Code 数据集 (HumanEval)"""
    print("📂 加载本地 Code 数据 (HumanEval)...")
    path = Path(HUMANEVAL_PATH)
    if not path.exists():
        print(f"❌ 未找到代码数据: {path}")
        return []
    
    processed = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            # 转换为我们的训练格式
            # 注意：HumanEval 的 'prompt' 是函数签名，'canonical_solution' 是实现
            # 我们需要构造一个让模型生成 Workflow 的 problem
            
            # 构造 Problem:
            # HumanEval 的 prompt 实际上就是题目描述（包含函数头）
            problem_text = item.get("prompt", "")
            
            processed.append({
                "problem": problem_text,
                "problem_type": "code",
                "ground_truth": item.get("canonical_solution", ""),
                "source": "humaneval",
                "entry_point": item.get("entry_point"),
                "test": item.get("test"),
                "task_id": item.get("task_id")
            })
            
    print(f"✅ 加载 Local Code 数据: {len(processed)} 条")
    return processed

def main():
    print("="*60)
    print("🚀 AgentFlow 数据集导入与混合工具")
    print("="*60)
    
    # 1. 加载各源数据
    qa_data = load_agentflow_qa()
    math_data = load_agentflow_math()
    code_data = load_local_code()
    
    if not qa_data and not math_data:
        print("❌ 未加载到任何 AgentFlow 数据，终止。")
        return

    # 2. 采样与平衡
    final_data = []
    
    # Math
    if SAMPLE_COUNTS["math"] and len(math_data) > SAMPLE_COUNTS["math"]:
        print(f"✂️  Math 数据采样: {len(math_data)} -> {SAMPLE_COUNTS['math']}")
        final_data.extend(random.sample(math_data, SAMPLE_COUNTS["math"]))
    else:
        final_data.extend(math_data)
        
    # QA
    if SAMPLE_COUNTS["qa"] and len(qa_data) > SAMPLE_COUNTS["qa"]:
        print(f"✂️  QA 数据采样: {len(qa_data)} -> {SAMPLE_COUNTS['qa']}")
        final_data.extend(random.sample(qa_data, SAMPLE_COUNTS["qa"]))
    else:
        final_data.extend(qa_data)
        
    # Code (HumanEval 很少，全部保留，甚至可以考虑过采样)
    # 为了平衡，我们将 HumanEval 重复 N 次，使其占比不至于太小（比如凑够 2000 条）
    if code_data:
        target_code_count = 2000
        repeat_factor = target_code_count // len(code_data) + 1
        print(f"🔄 Code 数据增强: {len(code_data)} -> ~{len(code_data) * repeat_factor} (重复 {repeat_factor} 次)")
        extended_code = code_data * repeat_factor
        final_data.extend(extended_code[:target_code_count])
    else:
        print("⚠️  警告: 没有任何 Code 数据！模型将失去代码生成能力。")

    # 3. 打乱
    random.shuffle(final_data)
    
    # 4. 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    print(f"\n💾 保存混合数据集到: {output_path}")
    with open(output_path, 'w') as f:
        for item in final_data:
            f.write(json.dumps(item) + "\n")
            
    # 5. 统计
    stats = {"math": 0, "qa": 0, "code": 0}
    for item in final_data:
        stats[item["problem_type"]] += 1
        
    print(f"✅ 完成！总样本数: {len(final_data)}")
    print(f"📊 分布统计: {stats}")
    print("\n💡 建议: 请在 config/training.yaml 中更新 'train_dataset' 字段指向此文件。")

if __name__ == "__main__":
    main()


