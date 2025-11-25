#!/usr/bin/env python3
"""
集成测试 - 验证数据流、执行器和奖励计算是否畅通
"""
import sys
import asyncio
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, '11/integrated_aflow_roll/src')

from data_manager import DataManager
from reward_computer import RewardComputer
# 模拟执行器 (避免调用真实LLM消耗Token)
class MockExecutor:
    async def execute_workflow(self, workflow_code, problem, problem_type, entry_point, test):
        # 模拟执行成功
        print(f"  🤖 [MockExecutor] Executing workflow for {problem_type}...")
        if problem_type == "math":
            return "42", 0.001, {"success": True}
        elif problem_type == "code":
            # 返回一个简单的通过测试的函数
            return f"def {entry_point}(*args): return args[0]", 0.001, {"success": True}
        else:
            return "Paris", 0.001, {"success": True}

async def run_test():
    print("🚀 开始集成测试...")
    
    # 1. 测试 DataManager
    print("\n1️⃣  测试 DataManager...")
    dm = DataManager(data_dir="11/integrated_aflow_roll/data/ready_to_train")
    dm.initialize()
    batch = dm.sample_batch(4, split="train")
    print(f"  ✅ 采样成功: {len(batch)} 条")
    print(f"  📊 分布: {dm.get_batch_stats(batch)}")
    
    # 2. 测试数据字段解析
    print("\n2️⃣  测试数据字段...")
    sample = batch[0]
    print(f"  样本类型: {sample.get('problem_type')}")
    print(f"  Meta信息: {sample.get('meta', {}).keys()}")
    
    # 3. 模拟训练循环中的提取逻辑
    print("\n3️⃣  模拟训练提取逻辑...")
    executor = MockExecutor()
    reward_computer = RewardComputer(debug_logging=True)
    
    for item in batch:
        p_type = item['problem_type']
        # 模拟 GRPO Trainer 中的提取逻辑
        test_code = item.get('test', '')
        entry_point = item.get('entry_point', '')
        
        if not test_code and 'meta' in item:
            test_code = item['meta'].get('test_cases', '')
        if not entry_point and 'meta' in item:
            entry_point = item['meta'].get('entry_point', '')
            
        print(f"  [{p_type}] Test len: {len(test_code)}, Entry: {entry_point}")
        
        # 模拟执行
        ans, cost, meta = await executor.execute_workflow(
            "def workflow(): pass", 
            item['problem'], 
            p_type, 
            entry_point, 
            test_code
        )
        
        # 计算奖励
        reward = reward_computer.compute_reward(
            problem=item['problem'],
            prediction=ans,
            ground_truth=item['ground_truth'],
            problem_type=p_type,
            metadata=meta,
            test=test_code,
            entry_point=entry_point
        )
        print(f"  💰 Reward: {reward}")

    print("\n✅ 集成测试通过！准备就绪。")

if __name__ == "__main__":
    asyncio.run(run_test())


