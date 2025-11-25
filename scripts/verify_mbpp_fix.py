#!/usr/bin/env python3
"""
Final verification that MBPP fixes are complete and working.
"""

import json
from pathlib import Path
from collections import Counter

def check_dataset_file(filepath: Path) -> dict:
    """
    Check a dataset file for MBPP completeness.
    """
    stats = {
        'total': 0,
        'mbpp': 0,
        'mbpp_with_entry_point': 0,
        'mbpp_with_test': 0,
        'mbpp_complete': 0,
        'entry_points': Counter(),
        'issues': []
    }

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                sample = json.loads(line)
                stats['total'] += 1

                if sample.get('source') == 'mbpp':
                    stats['mbpp'] += 1

                    # Check entry_point
                    entry_point = sample.get('entry_point', '')
                    if entry_point and entry_point != 'solution':
                        stats['mbpp_with_entry_point'] += 1
                        stats['entry_points'][entry_point] += 1
                    else:
                        stats['issues'].append(f"Line {i}: Missing/invalid entry_point")

                    # Check test field
                    test = sample.get('test', '')
                    if test and 'def check(candidate):' in test:
                        stats['mbpp_with_test'] += 1
                    else:
                        stats['issues'].append(f"Line {i}: Missing/invalid test field")

                    # Check if complete
                    if (entry_point and entry_point != 'solution' and
                        test and 'def check(candidate):' in test):
                        stats['mbpp_complete'] += 1

            except json.JSONDecodeError as e:
                stats['issues'].append(f"Line {i}: JSON decode error: {e}")

    return stats

def main():
    print("="*60)
    print("MBPP Fix Verification")
    print("="*60)

    data_dir = Path('data/mixed')
    files_to_check = [
        'train_mixed_with_math_fixed.jsonl',
        'val_mixed.jsonl',
        'test_mixed.jsonl'
    ]

    overall_stats = {
        'total_mbpp': 0,
        'complete_mbpp': 0,
        'total_issues': 0
    }

    for filename in files_to_check:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"\n⚠️  File not found: {filename}")
            continue

        print(f"\n{'='*60}")
        print(f"File: {filename}")
        print(f"{'='*60}")

        stats = check_dataset_file(filepath)

        print(f"\nTotal samples: {stats['total']}")
        print(f"MBPP samples: {stats['mbpp']}")
        print(f"\nMBPP Completeness:")
        print(f"  - With valid entry_point: {stats['mbpp_with_entry_point']}/{stats['mbpp']}")
        print(f"  - With test field: {stats['mbpp_with_test']}/{stats['mbpp']}")
        print(f"  - Complete (both): {stats['mbpp_complete']}/{stats['mbpp']}")

        if stats['mbpp_complete'] == stats['mbpp']:
            print(f"  ✅ All MBPP samples are complete!")
        else:
            print(f"  ⚠️  {stats['mbpp'] - stats['mbpp_complete']} incomplete MBPP samples")

        # Show top entry points
        if stats['entry_points']:
            print(f"\nTop 5 Entry Points:")
            for entry_point, count in stats['entry_points'].most_common(5):
                print(f"  - {entry_point}: {count}")

        # Show issues
        if stats['issues']:
            print(f"\n⚠️  Issues found: {len(stats['issues'])}")
            for issue in stats['issues'][:5]:  # Show first 5
                print(f"  - {issue}")
            if len(stats['issues']) > 5:
                print(f"  ... and {len(stats['issues']) - 5} more")
        else:
            print(f"\n✅ No issues found")

        overall_stats['total_mbpp'] += stats['mbpp']
        overall_stats['complete_mbpp'] += stats['mbpp_complete']
        overall_stats['total_issues'] += len(stats['issues'])

    # Overall summary
    print(f"\n{'='*60}")
    print("Overall Summary")
    print(f"{'='*60}")
    print(f"Total MBPP samples across all files: {overall_stats['total_mbpp']}")
    print(f"Complete MBPP samples: {overall_stats['complete_mbpp']}")
    print(f"Total issues: {overall_stats['total_issues']}")

    if overall_stats['complete_mbpp'] == overall_stats['total_mbpp']:
        print(f"\n✅ SUCCESS: All MBPP samples are properly fixed!")
        print(f"   - All {overall_stats['total_mbpp']} samples have:")
        print(f"     • Valid entry_point (extracted from ground_truth)")
        print(f"     • Proper test field (converted from test_list)")
        return 0
    else:
        missing = overall_stats['total_mbpp'] - overall_stats['complete_mbpp']
        print(f"\n⚠️  WARNING: {missing} MBPP samples still need fixing")
        return 1

if __name__ == '__main__':
    exit(main())
