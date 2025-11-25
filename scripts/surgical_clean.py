#!/usr/bin/env python3
"""
精准手术刀：剔除已知坏样本
"""
import json
import os

INPUT_FILE = "11/integrated_aflow_roll/data/ready_to_train/train.jsonl"
OUTPUT_FILE = "11/integrated_aflow_roll/data/ready_to_train/train_final_clean.jsonl"

def main():
    print(f"🔪 开始精准剔除坏样本...")
    
    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()
        
    clean_data = []
    dropped_count = 0
    
    for line in lines:
        item = json.loads(line)
        problem = item.get("problem", "").lower()
        answer = item.get("ground_truth", "").lower()
        
        # 规则 1: 剔除 "length and width" 这个具体的坏答案
        if "length and width" in answer:
            print(f"  ❌ 剔除: {item['problem'][:50]}... (Reason: Bad Ground Truth 'length and width')")
            dropped_count += 1
            continue
            
        # 规则 2: 剔除极短的 QA 答案 (防止其他噪音)
        if item.get("problem_type") == "qa" and len(answer) < 2 and not answer.isdigit():
             print(f"  ❌ 剔除: {item['problem'][:50]}... (Reason: Answer too short '{answer}')")
             dropped_count += 1
             continue

        clean_data.append(item)
        
    with open(OUTPUT_FILE, 'w') as f:
        for item in clean_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"\n📊 结果:")
    print(f"  原样本: {len(lines)}")
    print(f"  现样本: {len(clean_data)}")
    print(f"  剔除数: {dropped_count}")
    print(f"  输出: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()


