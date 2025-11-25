#!/usr/bin/env python3
"""分析训练日志，按领域统计准确率"""

import re
import sys
from collections import defaultdict

def parse_log(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 解析batch分布
    batch_pattern = r'📦 Batch (\d+): \d+ 样本, 分布: (.+)'
    batch_matches = re.finditer(batch_pattern, content)

    batches = {}
    for match in batch_matches:
        batch_num = int(match.group(1))
        dist_str = match.group(2)
        # 解析字典字符串
        dist = eval(dist_str)  # {'math': 2, 'qa': 1, 'code': 1}
        batches[batch_num] = dist

    # 按batch分割内容
    batch_sections = re.split(r'📦 Batch \d+:', content)

    # 为每个batch收集评分
    batch_scores = defaultdict(list)

    for i, section in enumerate(batch_sections[1:], 1):  # 跳过第一个空section
        if i not in batches:
            continue

        # 在当前batch section中查找所有评分
        score_pattern = r'正确性评分: ([\-\d\.]+)/10\.0'
        scores = re.findall(score_pattern, section.split('📦')[0])  # 只到下一个batch前

        # 每个样本有6个评分（GRPO的K=6）
        domain_dist = batches[i]
        samples_per_domain = []
        for domain, count in domain_dist.items():
            samples_per_domain.extend([domain] * count)

        # 将评分分配给各领域（每个样本6个评分）
        for idx, domain in enumerate(samples_per_domain):
            sample_scores = scores[idx*6:(idx+1)*6]
            if sample_scores:
                # 取平均分或最高分作为该样本的代表分
                avg_score = sum(float(s) for s in sample_scores) / len(sample_scores)
                batch_scores[domain].append(avg_score)

    return batch_scores

def analyze_scores(batch_scores):
    print("="*60)
    print("训练日志 - 按领域准确率分析")
    print("="*60)
    print()

    for domain in ['math', 'code', 'qa']:
        scores = batch_scores.get(domain, [])
        if not scores:
            continue

        total = len(scores)
        # 正确：平均分>=5.0
        correct = sum(1 for s in scores if s >= 5.0)
        accuracy = correct / total * 100 if total > 0 else 0

        avg_score = sum(scores) / total if total > 0 else 0

        print(f"【{domain.upper()}】")
        print(f"  样本数: {total}")
        print(f"  正确数: {correct}")
        print(f"  准确率: {accuracy:.1f}%")
        print(f"  平均分: {avg_score:.2f}/10.0")
        print(f"  最高分: {max(scores):.1f}")
        print(f"  最低分: {min(scores):.1f}")
        print()

    # 总体统计
    all_scores = []
    for scores in batch_scores.values():
        all_scores.extend(scores)

    if all_scores:
        total = len(all_scores)
        correct = sum(1 for s in all_scores if s >= 5.0)
        accuracy = correct / total * 100
        avg_score = sum(all_scores) / total

        print("【总体】")
        print(f"  样本数: {total}")
        print(f"  正确数: {correct}")
        print(f"  准确率: {accuracy:.1f}%")
        print(f"  平均分: {avg_score:.2f}/10.0")
        print()

    print("="*60)

if __name__ == '__main__':
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'logs/train_with_retry_20251120_152648.log'
    batch_scores = parse_log(log_file)
    analyze_scores(batch_scores)
