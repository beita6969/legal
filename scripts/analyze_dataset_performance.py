import json
import sys

# Dataset source mapping
source_to_category = {
    'gsm8k': 'Math',
    'math': 'Math',
    'humaneval': 'Code',
    'mbpp': 'Code',
    'mmlu': 'QA',
    'commonsenseqa': 'QA',
    'hotpotqa': 'QA'
}

def analyze_datasets():
    # Training set analysis
    train_file = 'data/mixed/train_mixed.jsonl'
    val_file = 'data/mixed/val_mixed.jsonl'
    test_file = 'data/test/mixed_dataset.jsonl'
    
    def count_by_source(file_path):
        source_counts = {}
        category_counts = {'Math': 0, 'Code': 0, 'QA': 0, 'Mixed': 0}
        total = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                source = data.get('source', 'unknown')
                problem_type = data.get('problem_type', 'unknown')
                
                source_counts[source] = source_counts.get(source, 0) + 1
                
                # Categorize
                if problem_type in ['math', 'code', 'qa']:
                    category = problem_type.title()
                    category_counts[category] += 1
                else:
                    category_counts['Mixed'] += 1
                
                total += 1
        
        return source_counts, category_counts, total
    
    print("="*80)
    print("数据集详细分析报告")
    print("="*80)
    
    # Training
    print("\n【训练集】")
    train_sources, train_cats, train_total = count_by_source(train_file)
    print(f"总计: {train_total} 样本")
    print("\n按数据源:")
    for source, count in sorted(train_sources.items(), key=lambda x: -x[1]):
        pct = count / train_total * 100
        print(f"  {source:15} {count:5} ({pct:5.1f}%)")
    print("\n按任务类型:")
    for cat, count in sorted(train_cats.items(), key=lambda x: -x[1]):
        pct = count / train_total * 100
        print(f"  {cat:10} {count:5} ({pct:5.1f}%)")
    
    # Validation
    print("\n【验证集】")
    val_sources, val_cats, val_total = count_by_source(val_file)
    print(f"总计: {val_total} 样本")
    print("\n按数据源:")
    for source, count in sorted(val_sources.items(), key=lambda x: -x[1]):
        pct = count / val_total * 100
        print(f"  {source:15} {count:5} ({pct:5.1f}%)")
    print("\n按任务类型:")
    for cat, count in sorted(val_cats.items(), key=lambda x: -x[1]):
        pct = count / val_total * 100
        print(f"  {cat:10} {count:5} ({pct:5.1f}%)")
    
    # Test
    print("\n【测试集】")
    test_sources, test_cats, test_total = count_by_source(test_file)
    print(f"总计: {test_total} 样本")
    print("\n按数据源:")
    for source, count in sorted(test_sources.items(), key=lambda x: -x[1]):
        pct = count / test_total * 100
        print(f"  {source:15} {count:5} ({pct:5.1f}%)")
    print("\n按任务类型:")
    for cat, count in sorted(test_cats.items(), key=lambda x: -x[1]):
        pct = count / test_total * 100
        print(f"  {cat:10} {count:5} ({pct:5.1f}%)")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    analyze_datasets()
