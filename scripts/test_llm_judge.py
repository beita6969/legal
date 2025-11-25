#!/usr/bin/env python3
"""Test the effectiveness of the improved LLM Judge parsing."""

import json
import re
from typing import Dict, Tuple

# Simulate the enhanced LLM Judge parsing logic
def parse_llm_judge_response(response: str) -> Tuple[float, str]:
    """Enhanced parsing with 5-tier cascade for robustness."""
    
    # Tier 1: Standard XML tags
    xml_pattern = r'<answer>([^<]+)</answer>.*?<score>([^<]+)</score>'
    match = re.search(xml_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        score_str = match.group(2).strip()
        try:
            score = float(score_str)
            return score, answer
        except ValueError:
            pass
    
    # Tier 2: Colon-separated format
    colon_pattern = r'Answer:\s*([^\n]+).*?Score:\s*([\d.-]+)'
    match = re.search(colon_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        try:
            score = float(match.group(2))
            return score, answer
        except ValueError:
            pass
    
    # Tier 3: Markdown bold format
    md_pattern = r'\*\*Answer\*\*:?\s*([^\n*]+).*?\*\*Score\*\*:?\s*([\d.-]+)'
    match = re.search(md_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        try:
            score = float(match.group(2))
            return score, answer
        except ValueError:
            pass
    
    # Tier 4: Simple key-value format
    simple_pattern = r'score[:\s]+([\d.-]+)'
    match = re.search(simple_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            score = float(match.group(1))
            # Try to find answer separately
            answer_match = re.search(r'answer[:\s=]+([^\n]+)', response, re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                answer = "[extracted]"
            return score, answer
        except ValueError:
            pass
    
    # Tier 5: Fallback - search for score anywhere
    score_patterns = [
        r'score[:\s]*([\d.-]+)',
        r'([\d.-]+)\s*(?:points?|score)',
        r'score\s*=\s*([\d.-]+)',
        r'\bscore\b.*?([\d.-]+)'
    ]

    for pattern in score_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            score_str = match.group(1)
            try:
                score = float(score_str)
                # Try to extract answer from beginning
                answer_match = re.search(r'^\s*([^\n]{1,100})', response)
                answer = answer_match.group(1) if answer_match else "[extracted]"
                return score, answer
            except ValueError:
                continue
    
    return 0.0, "[parsing failed]"

# Test cases with various LLM response formats
test_responses = [
    # Standard XML format
    {
        "response": "<answer>100</answer>\n<score>5.0</score>",
        "expected_score": 5.0,
        "description": "Standard XML format"
    },
    # Colon-separated format
    {
        "response": "Answer: The solution is correct.\nScore: 5",
        "expected_score": 5.0,
        "description": "Colon-separated format"
    },
    # Markdown bold format
    {
        "response": "**Answer**: 42\n**Score**: -5.0",
        "expected_score": -5.0,
        "description": "Markdown bold format"
    },
    # Simple key-value
    {
        "response": "The answer is 100\nscore is 5",
        "expected_score": 5.0,
        "description": "Simple key-value format"
    },
    # Complex response with score at end
    {
        "response": "After careful analysis of the solution...\nThe implementation correctly handles edge cases...\nScore: 5 points",
        "expected_score": 5.0,
        "description": "Score at end of response"
    },
    # Mixed format
    {
        "response": "Looking at this problem:\n<answer>Correct</answer>\nThe score is <score>5</score>",
        "expected_score": 5.0,
        "description": "Mixed XML tags"
    },
    # Verbose response
    {
        "response": "The student's answer demonstrates a good understanding. Answer: Acceptable. The methodology is sound. Score: 3.5",
        "expected_score": 3.5,
        "description": "Verbose with inline score"
    },
    # Malformed but parseable
    {
        "response": "answer = wrong solution\n\n\nscore = -5",
        "expected_score": -5.0,
        "description": "Malformed but parseable"
    },
    # Edge case: negative score
    {
        "response": "This is incorrect. Score: -5.0",
        "expected_score": -5.0,
        "description": "Negative score"
    },
    # Edge case: unparseable
    {
        "response": "The solution looks good but I cannot determine a numerical score.",
        "expected_score": 0.0,
        "description": "Unparseable response"
    }
]

def test_parsing():
    """Test the enhanced parsing logic."""
    print("\n" + "="*60)
    print("Testing Enhanced LLM Judge Parsing")
    print("="*60 + "\n")
    
    success_count = 0
    failure_count = 0
    
    for i, test_case in enumerate(test_responses, 1):
        response = test_case["response"]
        expected = test_case["expected_score"]
        description = test_case["description"]
        
        print(f"Test {i}: {description}")
        print(f"  Response: {response[:50]}..." if len(response) > 50 else f"  Response: {response}")
        print(f"  Expected Score: {expected}")
        
        score, answer = parse_llm_judge_response(response)
        print(f"  Parsed Score: {score}")
        print(f"  Parsed Answer: {answer[:30]}..." if len(answer) > 30 else f"  Parsed Answer: {answer}")
        
        if abs(score - expected) < 0.01:
            print(f"  ✓ SUCCESS")
            success_count += 1
        else:
            print(f"  ✗ FAILED (got {score}, expected {expected})")
            failure_count += 1
        print()
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    total = len(test_responses)
    print(f"Total Tests: {total}")
    print(f"Successful: {success_count} ({success_count/total*100:.1f}%)")
    print(f"Failed: {failure_count} ({failure_count/total*100:.1f}%)")
    
    success_rate = success_count / total * 100
    if success_rate >= 90:
        print(f"\n✓ EXCELLENT: {success_rate:.1f}% success rate (target: >90%)")
    elif success_rate >= 80:
        print(f"\n⚠ GOOD: {success_rate:.1f}% success rate (target: >90%)")
    else:
        print(f"\n✗ NEEDS IMPROVEMENT: {success_rate:.1f}% success rate (target: >90%)")
    
    return success_count, failure_count

if __name__ == "__main__":
    test_parsing()
