import json

def analyze_datasets():
    train_file = 'data/mixed/train_mixed.jsonl'
    val_file = 'data/mixed/val_mixed.jsonl'
    test_file = 'data/test/mixed_dataset.jsonl'
    
    def count_by_source(file_path):
        source_counts = {}
        type_counts = {}
        total = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                source = data.get('source', 'unknown')
                ptype = data.get('problem_type', 'unknown')
                
                source_counts[source] = source_counts.get(source, 0) + 1
                type_counts[ptype] = type_counts.get(ptype, 0) + 1
                total += 1
        
        return source_counts, type_counts, total
    
    print("="*80)
    print("数据集详细分析 (含mmlu, gsm8k, math, code等)")
    print("="*80)
    
    for name, fpath in [('训练集', train_file), ('验证集', val_file), ('测试集', test_file)]:
        print(f"\n【{name}】")
        try:
            sources, types, total = count_by_source(fpath)
            print(f"总样本数: {total}")
            print("\n按数据源分布:")
            for src in sorted(sources.keys(), key=lambda x: -sources[x]):
                pct = sources[src] / total * 100
                print(f"  {src:15} {sources[src]:6,} ({pct:5.1f}%)")
            print("\n按问题类型分布:")
            for ptype in sorted(types.keys(), key=lambda x: -types[x]):
                pct = types[ptype] / total * 100
                print(f"  {ptype:10} {types[ptype]:6,} ({pct:5.1f}%)")
        except FileNotFoundError:
            print(f"  文件未找到: {fpath}")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    analyze_datasets()
