"""Improved prompt templates based on AFlow design principles.

Key improvements:
1. Clear, focused prompts without meta-instructions
2. Direct problem-solving approach
3. No workflow scaffolding in prompts
4. Clean formatting without excessive emojis or markers
"""

# For rl_workflow_generator.py - Replace complex workflow generation prompt
WORKFLOW_GENERATION_PROMPT = """
Solve this problem step by step.

Problem: {problem}

Provide a clear solution with your reasoning.
"""

# For GRPO actor model - Direct answer generation
GRPO_ANSWER_PROMPT = """
{problem}

Provide the answer directly.
"""

# For reward_computer.py - Improved LLM Judge prompt
LLM_JUDGE_PROMPT = """
Evaluate if the given answer correctly solves the problem.

Problem: {problem}
Answer: {answer}
Expected: {ground_truth}

Provide:
1. Score: 5 if correct, -5 if incorrect
2. Brief explanation

Format:
<score>{score}</score>
<explanation>{explanation}</explanation>
"""

# For code generation tasks
CODE_GENERATION_PROMPT = """
Write a Python function to solve this problem:

{problem}

Requirements:
- Function name: {entry_point}
- Return the result directly
- Include necessary imports
"""

# For math problems
MATH_PROBLEM_PROMPT = """
Solve this math problem:

{problem}

Show your work step by step, then provide the final numerical answer.
"""

# For QA tasks
QA_PROMPT = """
{problem}

Answer the question based on the information provided.
"""

# For ensemble/voting
ENSEMBLE_PROMPT = """
Multiple solutions have been provided for this problem:

Problem: {problem}

Solutions:
{solutions}

Select the best solution and explain why.

Format:
<selected>{letter}</selected>
<reason>{reason}</reason>
"""

# For reflection/revision
REFLECTION_PROMPT = """
The following solution has an error:

Problem: {problem}
Solution: {solution}
Error: {error}

Provide a corrected solution.
"""

# Simplified prompt optimizer rules
PROMPT_OPTIMIZER_RULES = """
Optimization guidelines:
1. Be direct and concise
2. Focus on the problem, not meta-instructions
3. Use clear formatting
4. Avoid unnecessary complexity
"""
