import re
from collections import Counter

def extract_workflows(log_file):
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find all workflow code blocks
    pattern = r'DEBUG: Qwen 生成的原始文本.*?```python(.*?)```'
    workflows = re.findall(pattern, content, re.DOTALL)
    
    return workflows

def analyze_workflow(workflow_code):
    """Analyze operator usage in a workflow"""
    operators_used = []
    
    # Check for operator usage
    if 'answer_generate' in workflow_code:
        operators_used.append('AnswerGenerate')
    if 'programmer' in workflow_code:
        operators_used.append('Programmer')
    if 'test' in workflow_code:
        operators_used.append('Test')
    if 'review' in workflow_code:
        operators_used.append('Review')
    if 'revise' in workflow_code:
        operators_used.append('Revise')
    if 'custom' in workflow_code:
        operators_used.append('Custom')
    if 'sc_ensemble' in workflow_code:
        operators_used.append('ScEnsemble')
    
    return tuple(sorted(operators_used))

def main():
    log_file = 'logs/train_with_retry_20251120_152648.log'
    
    print("="*80)
    print("Workflow多样性分析")
    print("="*80)
    
    workflows = extract_workflows(log_file)
    print(f"\n找到 {len(workflows)} 个workflow代码块")
    
    # Analyze patterns
    patterns = []
    for wf in workflows:
        pattern = analyze_workflow(wf)
        if pattern:  # Only count non-empty patterns
            patterns.append(pattern)
    
    pattern_counts = Counter(patterns)
    
    print(f"\n不同的operator组合模式: {len(pattern_counts)} 种")
    print("\n按使用频率排序:")
    for i, (pattern, count) in enumerate(pattern_counts.most_common(), 1):
        pct = count / len(patterns) * 100
        print(f"{i}. {' + '.join(pattern)}")
        print(f"   使用次数: {count} ({pct:.1f}%)")
    
    # Operator frequency
    all_ops = []
    for pattern in patterns:
        all_ops.extend(pattern)
    
    op_counts = Counter(all_ops)
    print("\n\n各Operator总使用频率:")
    for op, count in op_counts.most_common():
        pct = count / len(patterns) * 100
        print(f"  {op:15} {count:3}次 ({pct:5.1f}%)")
    
    # Complexity analysis
    complexity = [len(p) for p in patterns]
    avg_complexity = sum(complexity) / len(complexity) if complexity else 0
    
    print("\n\nWorkflow复杂度分析:")
    print(f"  平均使用operator数量: {avg_complexity:.1f}")
    print(f"  最简单workflow: {min(complexity)} 个operator")
    print(f"  最复杂workflow: {max(complexity)} 个operator")
    
    # Check for diversity
    diversity_score = len(pattern_counts) / len(patterns) * 100 if patterns else 0
    print(f"\n多样性评分: {diversity_score:.1f}%")
    print(f"  (不同模式数 / 总workflow数)")
    
    if diversity_score > 30:
        print("  ✅ 多样性良好")
    elif diversity_score > 15:
        print("  ⚠️ 多样性中等")
    else:
        print("  ❌ 多样性偏低")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
