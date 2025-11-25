#!/usr/bin/env python3
"""
Fix MBPP samples in mixed dataset:
1. Extract entry_point from ground_truth code
2. Convert test_list to proper test format
3. Ensure all required fields are present
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List

def extract_entry_point_from_code(code: str) -> str:
    """
    Extract function name from ground_truth code.

    Example:
        "def geometric_sum(n):\n  ..." -> "geometric_sum"
    """
    # Try to find function definition
    match = re.search(r'^def\s+(\w+)\s*\(', code, re.MULTILINE)
    if match:
        return match.group(1)

    # Fallback: try class definition
    match = re.search(r'^class\s+(\w+)\s*[\(:]', code, re.MULTILINE)
    if match:
        return match.group(1)

    # If all else fails, return a default
    return "solution"

def convert_test_list_to_test(test_list: List[str], entry_point: str) -> str:
    """
    Convert MBPP test_list to HumanEval-style test format.

    Args:
        test_list: List of assert statements like ["assert func(1) == 2"]
        entry_point: Function name to test

    Returns:
        Formatted test code like HumanEval format
    """
    if not test_list:
        return f"""def check(candidate):
    # No tests available
    pass
"""

    # Create test function
    test_lines = ["def check(candidate):"]
    test_lines.append("    # Test cases from MBPP")

    for test_case in test_list:
        # Replace original function name with 'candidate'
        # Handle various assertion formats
        modified = test_case.strip()

        # Replace function calls with candidate calls
        # Pattern: function_name(...) -> candidate(...)
        modified = re.sub(
            rf'\b{re.escape(entry_point)}\s*\(',
            'candidate(',
            modified
        )

        # Add indentation
        test_lines.append(f"    {modified}")

    return "\n".join(test_lines) + "\n"

def fix_mbpp_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix a single MBPP sample.

    Returns:
        Fixed sample with proper entry_point and test fields
    """
    if sample.get('source') != 'mbpp':
        return sample

    # 1. Extract entry_point from ground_truth
    ground_truth = sample.get('ground_truth', '')
    if ground_truth and ('entry_point' not in sample or sample.get('entry_point') == 'solution'):
        sample['entry_point'] = extract_entry_point_from_code(ground_truth)

    # 2. Convert test_list to test format
    if 'test_list' in sample and sample.get('test_list'):
        entry_point = sample.get('entry_point', 'solution')
        sample['test'] = convert_test_list_to_test(sample['test_list'], entry_point)
    elif 'test' not in sample or not sample.get('test'):
        # No test_list and no test, create placeholder
        sample['test'] = """def check(candidate):
    # No tests available from MBPP
    pass
"""

    return sample

def process_file(input_path: Path, output_path: Path) -> Dict[str, int]:
    """
    Process a JSONL file and fix all MBPP samples.

    Returns:
        Statistics about processing
    """
    stats = {
        'total': 0,
        'mbpp': 0,
        'fixed': 0,
        'entry_point_extracted': 0,
        'test_converted': 0
    }

    print(f"\nProcessing: {input_path}")

    samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                stats['total'] += 1

                if sample.get('source') == 'mbpp':
                    stats['mbpp'] += 1

                    # Track what we're fixing
                    old_entry_point = sample.get('entry_point')
                    had_test_list = 'test_list' in sample

                    # Fix the sample
                    fixed_sample = fix_mbpp_sample(sample)

                    # Update stats
                    if old_entry_point != fixed_sample.get('entry_point'):
                        stats['entry_point_extracted'] += 1
                    if had_test_list and 'test' in fixed_sample:
                        stats['test_converted'] += 1
                    if old_entry_point != fixed_sample.get('entry_point') or had_test_list:
                        stats['fixed'] += 1

                    samples.append(fixed_sample)
                else:
                    samples.append(sample)

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"  Total samples: {stats['total']}")
    print(f"  MBPP samples: {stats['mbpp']}")
    print(f"  Fixed samples: {stats['fixed']}")
    print(f"    - Entry points extracted: {stats['entry_point_extracted']}")
    print(f"    - Tests converted: {stats['test_converted']}")
    print(f"  Output: {output_path}")

    return stats

def main():
    """Fix MBPP samples in all mixed dataset files."""
    print("="*60)
    print("Fix MBPP Samples")
    print("="*60)

    data_dir = Path('data/mixed')

    # Files to process
    files_to_process = [
        ('train_mixed_with_math_fixed.jsonl', 'train_mixed_mbpp_fixed.jsonl'),
        ('val_mixed.jsonl', 'val_mixed_mbpp_fixed.jsonl'),
        ('test_mixed.jsonl', 'test_mixed_mbpp_fixed.jsonl'),
    ]

    total_stats = {
        'total': 0,
        'mbpp': 0,
        'fixed': 0,
        'entry_point_extracted': 0,
        'test_converted': 0
    }

    for input_file, output_file in files_to_process:
        input_path = data_dir / input_file
        output_path = data_dir / output_file

        if not input_path.exists():
            print(f"\nSkipping {input_file} (not found)")
            continue

        stats = process_file(input_path, output_path)

        # Accumulate stats
        for key in total_stats:
            total_stats[key] += stats[key]

    print("\n" + "="*60)
    print("Overall Statistics")
    print("="*60)
    print(f"Total samples processed: {total_stats['total']}")
    print(f"MBPP samples found: {total_stats['mbpp']}")
    print(f"MBPP samples fixed: {total_stats['fixed']}")
    print(f"  - Entry points extracted: {total_stats['entry_point_extracted']}")
    print(f"  - Tests converted: {total_stats['test_converted']}")
    print("\n✅ MBPP samples fixed successfully!")
    print("="*60)

if __name__ == '__main__':
    main()
