#!/usr/bin/env python3
"""
数据集 LLM 深度扫描与清洗脚本
目的：利用本地 LLM (GPT OSS 120B @ 8002) 对训练集进行"体检"，剔除 Ground Truth 明显错误或与问题不匹配的样本。

扫描逻辑：
1. 构造 Prompt：请判断 Question 和 Ground Truth 是否构成合理的问答对。
2. 关注点：
   - Answer 是否是 Question 的有效答案？
   - Answer 是否明显错误（如 Commercial Paper 定义为 length and width）？
   - Context 是否缺失导致无法回答？
3. 输出：保留高质量样本，生成 bad_samples.jsonl 供审查。
"""

import json
import os
import asyncio
from tqdm import tqdm
from openai import OpenAI

# 配置
INPUT_FILE = "11/integrated_aflow_roll/data/ready_to_train/train.jsonl"
OUTPUT_FILE = "11/integrated_aflow_roll/data/ready_to_train/train_clean_llm.jsonl"
BAD_FILE = "11/integrated_aflow_roll/data/ready_to_train/bad_samples.jsonl"
LLM_BASE_URL = "http://localhost:8002/v1"
LLM_API_KEY = "sk-dummy"
MODEL_NAME = "/home/yijia/lhy/openai/gpt-oss-120b"
CONCURRENCY = 20  # 并发请求数

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

async def check_sample(item, semaphore):
    async with semaphore:
        problem = item['problem']
        answer = item['ground_truth']
        p_type = item['problem_type']
        
        # Code 类型通常比较可靠（来自 HumanEval/MBPP），且 LLM 难以仅凭文本判断代码正确性
        # 除非答案明显太短或非代码
        if p_type == 'code':
            if len(str(answer)) < 10:
                return False, "Code answer too short"
            return True, ""

        # Math/QA 类型进行深度检查
        prompt = f"""You are a Data Quality Auditor. Your task is to verify if the following Question and Ground Truth Answer form a valid, logical, and self-contained pair.

Question: {problem}
Ground Truth Answer: {answer}

Evaluation Criteria:
1. **Relevance**: Does the answer actually answer the question? (e.g., Q: "Define X", A: "length" -> INVALID)
2. **Self-containment**: Does the question make sense without external context? (e.g., Q: "What did he say?" -> INVALID)
3. **Correctness**: Is the answer factually plausible? (Ignore minor formatting or date discrepancies, focus on logic).

Respond ONLY with a JSON object:
{{
    "valid": true/false,
    "reason": "short explanation"
}}
"""
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return result['valid'], result.get('reason', 'No reason')
        except Exception as e:
            # 如果 LLM 调用失败或解析失败，保守起见保留样本（或者是网络抖动）
            # 但为了清洗彻底，我们可以标记为 False 并在最后人工复核
            print(f"⚠️ API Error: {e}")
            return True, "API Error (Skipped check)"

async def scan_dataset():
    print(f"🚀 开始 LLM 数据深度扫描...")
    print(f"  输入: {INPUT_FILE}")
    print(f"  模型: {MODEL_NAME}")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ 输入文件不存在")
        return

    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()
    
    print(f"  总样本数: {len(lines)}")
    
    tasks = []
    semaphore = asyncio.Semaphore(CONCURRENCY)
    
    results = []
    
    # 创建任务
    for i, line in enumerate(lines):
        item = json.loads(line)
        # 为每行绑定原始数据和检查任务
        tasks.append((item, check_sample(item, semaphore)))
    
    # 执行
    valid_count = 0
    bad_count = 0
    
    with open(OUTPUT_FILE, 'w') as f_out, open(BAD_FILE, 'w') as f_bad:
        # 使用 tqdm 显示进度
        for item, task in tqdm(tasks, total=len(tasks), desc="Scanning"):
            is_valid, reason = await task
            
            if is_valid:
                valid_count += 1
                f_out.write(json.dumps(item) + "\n")
            else:
                bad_count += 1
                item['drop_reason'] = reason
                f_bad.write(json.dumps(item) + "\n")
                
    print("\n📊 扫描完成!")
    print(f"  ✅ 有效样本: {valid_count}")
    print(f"  🗑️  剔除样本: {bad_count}")
    print(f"  💾 清洗后文件: {OUTPUT_FILE}")
    print(f"  📝 剔除详情: {BAD_FILE}")

    # 打印几个剔除案例
    if bad_count > 0:
        print("\n🔍 剔除案例示例:")
        with open(BAD_FILE, 'r') as f:
            for _ in range(min(5, bad_count)):
                bad = json.loads(f.readline())
                print(f"  Q: {bad['problem'][:50]}...")
                print(f"  A: {bad['ground_truth'][:50]}...")
                print(f"  Reason: {bad['drop_reason']}\n")

if __name__ == "__main__":
    asyncio.run(scan_dataset())

