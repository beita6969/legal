#!/usr/bin/env python3
"""
Test MBPP evaluation to ensure the fixes work properly.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from unified_evaluator import UnifiedEvaluator
    from answer_extractor import AnswerExtractor
except ImportError:
    # Try direct import
    import importlib.util
    spec = importlib.util.spec_from_file_location("unified_evaluator", "src/unified_evaluator.py")
    unified_evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unified_evaluator)
    UnifiedEvaluator = unified_evaluator.UnifiedEvaluator

    spec = importlib.util.spec_from_file_location("answer_extractor", "src/answer_extractor.py")
    answer_extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(answer_extractor)
    AnswerExtractor = answer_extractor.AnswerExtractor

def test_mbpp_evaluation():
    """
    Test MBPP evaluation with fixed samples.
    """
    print("="*60)
    print("Test MBPP Evaluation")
    print("="*60)

    # Load a few MBPP samples
    data_file = Path('data/mixed/train_mixed_mbpp_fixed.jsonl')
    if not data_file.exists():
        print(f"\n❌ File not found: {data_file}")
        return

    samples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip():
                sample = json.loads(line)
                if sample.get('source') == 'mbpp':
                    samples.append(sample)
                    if len(samples) >= 3:
                        break

    if not samples:
        print("\n❌ No MBPP samples found")
        return

    print(f"\nLoaded {len(samples)} MBPP samples\n")

    evaluator = UnifiedEvaluator()
    extractor = AnswerExtractor(use_llm_fallback=False)

    for i, sample in enumerate(samples, 1):
        print(f"\n{'='*60}")
        print(f"Sample {i}: {sample.get('task_id', 'unknown')}")
        print(f"{'='*60}")

        # Check required fields
        print("\n✅ Required fields:")
        print(f"  - problem: {len(sample.get('problem', ''))} chars")
        print(f"  - ground_truth: {len(sample.get('ground_truth', ''))} chars")
        print(f"  - entry_point: {sample.get('entry_point', 'MISSING')}")
        print(f"  - test: {len(sample.get('test', ''))} chars")
        print(f"  - test_list: {len(sample.get('test_list', []))} items")

        # Show entry_point extraction
        ground_truth = sample.get('ground_truth', '')
        entry_point = sample.get('entry_point', '')
        print(f"\n📝 Ground Truth (first 100 chars):")
        print(f"  {ground_truth[:100]}...")
        print(f"\n🎯 Extracted Entry Point: {entry_point}")

        # Show test conversion
        test_list = sample.get('test_list', [])
        test = sample.get('test', '')
        print(f"\n🧪 Test Conversion:")
        print(f"  Original test_list ({len(test_list)} items):")
        for t in test_list[:2]:  # Show first 2
            print(f"    - {t}")
        print(f"\n  Converted test (first 200 chars):")
        print(f"    {test[:200]}...")

        # Test evaluation with ground truth (should pass)
        print(f"\n🔍 Testing Evaluation:")
        print(f"  Evaluating ground_truth against itself...")
        result = evaluator.evaluate_code(
            prediction=ground_truth,
            ground_truth=ground_truth,
            test=test
        )
        print(f"  Result: {result}")

        if result.get('correct'):
            print(f"  ✅ PASSED: Ground truth passes its own tests")
        else:
            print(f"  ❌ FAILED: {result.get('message', 'unknown error')}")

        # Test with a simple incorrect solution
        print(f"\n  Testing with incorrect solution...")
        bad_code = f"def {entry_point}(*args, **kwargs):\n    return None"
        result_bad = evaluator.evaluate_code(
            prediction=bad_code,
            ground_truth=ground_truth,
            test=test
        )
        print(f"  Result: {result_bad}")

        if not result_bad.get('correct'):
            print(f"  ✅ CORRECT: Bad solution correctly fails")
        else:
            print(f"  ⚠️  WARNING: Bad solution incorrectly passes")

    print(f"\n{'='*60}")
    print("✅ MBPP Evaluation Test Complete")
    print(f"{'='*60}")

if __name__ == '__main__':
    test_mbpp_evaluation()
