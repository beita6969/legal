# LLM Judge 优化实施指南

> **文档版本**: 1.0
> **创建日期**: 2025-11-23
> **目标**: 为不同数据集设计专属的LLM Judge，减少误判，提升准确率

---

## 📋 目录

1. [背景和动机](#背景和动机)
2. [当前问题分析](#当前问题分析)
3. [设计方案](#设计方案)
4. [配置文件说明](#配置文件说明)
5. [实施步骤](#实施步骤)
6. [测试和验证](#测试和验证)
7. [预期效果](#预期效果)
8. [FAQ](#faq)

---

## 背景和动机

### 当前状态

**训练日志分析结果** (基于 `logs/train_restored_v10.log`):
- 总样本: 848
- 当前准确率: **64.9%**
- 误判率: **12-20%** (35-60个样本)
- 主要问题:
  1. **格式问题** (30-50个样本): 代码泄漏、空输出、单位差异
  2. **标注歧义** (5-10个样本): Drawstring bag等主观性问题
  3. **Judge推断错误** (未知数量): 选项题的"脑补"等价性

### 优化目标

- 将误判率从 **12-20%** 降至 **< 5%**
- 将准确率从 **64.9%** 提升至 **70-75%**
- 为每种数据集提供针对性的评估策略

---

## 当前问题分析

### 问题1: 一刀切的评估策略

**当前实现** (`src/reward_computer.py:114-308`):
- 使用**单一的通用LLM Judge**处理所有类型问题
- 同一个prompt模板适用于Math、Code、QA

**问题**:
```python
# 当前的通用prompt（简化版）
query_prompt = """
You are a precise mathematical and logical equivalence evaluator.

**Step 1**: Extract the Final Answer from both prediction and ground truth
**Step 2**: Normalize Both Answers
**Step 3**: Compare Equivalence

Prediction: {prediction}
Ground Truth: {ground_truth}
"""
```

这个prompt对于不同类型的问题存在以下局限：

| 数据集类型 | 问题 | 示例 |
|-----------|------|------|
| **GSM8K** | 未识别 `####` 格式 | `#### 72` 被当作普通文本 |
| **Math** | 未优先使用 `meta.short_answer` | 从长文本中提取答案可能出错 |
| **Code** | 使用文本匹配而非执行 | 变量名不同被判错 |
| **QA** | 推断选项等价性 | `"E"` ≠ `"might dream"` 被判为等价 |

### 问题2: 格式提取不鲁棒

**典型案例**:
```python
# 模型输出
prediction = "\\boxed{def solve() -> int:\n    return 50}"

# Ground Truth
ground_truth = "50"

# 当前Judge评分: 0.0 ❌
# 应该: 提取代码执行结果或识别为格式错误
```

**根本原因**:
- Answer Extractor (`src/answer_extractor.py`) 未处理代码泄漏
- Judge未识别 `\\boxed{def ...}` 是异常格式

### 问题3: 选项题的"脑补"问题

**观察到的模式**:

| 模型答案 | Ground Truth | 当前评分 | 应该 | 问题 |
|---------|-------------|---------|------|------|
| `"might dream"` | `"E"` | 1.0 ✅ | 1.0 ✅ | 正确 |
| `"E"` | `"might dream"` | 1.0 ✅ | 0.0 ❌ | **误判** |

**分析**:
- 第一种情况：模型���出完整答案，Judge推断这是E选项的内容 → 合理
- 第二种情况：模型只给出字母E，Judge推断E对应"might dream" → **不合理**

**原因**: Judge的prompt未明确禁止这种推断

---

## 设计方案

### 核心思想: 数据集专属策略

**设计原则**:
1. **针对性**: 每种数据集使用定制化的评估策略
2. **优先级**: 明确的评估方法选择顺序
3. **鲁棒性**: 处理各种边缘情况和格式问题
4. **可扩展**: 易于添加新数据集支持

### 架构设计

```
┌─────────────────────────────────────────────────────┐
│            GRPOTrainer.train_step()                 │
└────────────────┬────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────┐
│       RewardComputer.compute_reward()               │
│  ┌──────────────────────────────────────────────┐  │
│  │  1. 识别数据集类型 (source/problem_type)      │  │
│  │  2. 选择评估策略                              │  │
│  │  3. 调用对应的Judge                           │  │
│  └──────────────────────────────────────────────┘  │
└────┬─────────────┬─────────────┬──────────────┬────┘
     │             │             │              │
     ↓             ↓             ↓              ↓
┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐
│ GSM8K   │  │  Math    │  │  Code   │  │    QA    │
│ Judge   │  │  Judge   │  │  Exec   │  │  Judge   │
└─────────┘  └──────────┘  └─────────┘  └──────────┘
     │             │             │              │
     ↓             ↓             ↓              ↓
┌─────────────────────────────────────────────────────┐
│            配置文件: judge_prompts.yaml              │
│  ┌──────────────────────────────────────────────┐  │
│  │ - GSM8K: 识别####格式，忽略<<calc>>          │  │
│  │ - Math: 优先short_answer，LaTeX标准化       │  │
│  │ - Code: 完全依赖测试执行                      │  │
│  │ - QA: 禁止推断选项等价性                      │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 评估策略决策树

```python
def select_evaluation_strategy(sample: Dict) -> str:
    """根据样本特征选择评估策略"""

    # 优先级1: 检查source字段
    source = sample.get('source', '')

    if source in ['humaneval', 'mbpp']:
        # Code题：使用测试执行
        return 'test_execution'

    elif source == 'gsm8k':
        # GSM8K：使用GSM8K专属Judge
        return 'gsm8k_judge'

    elif source == 'math':
        # Math：使用Math专属Judge
        return 'math_judge'

    elif source in ['hotpotqa', 'squad_v2', 'commonsenseqa', 'mmlu']:
        # QA题：使用QA专属Judge
        return f'{source}_judge'

    # 优先级2: 根据problem_type
    problem_type = sample.get('problem_type', '')

    if problem_type == 'math':
        return 'math_judge'  # 默认Math Judge
    elif problem_type == 'code':
        return 'test_execution'
    elif problem_type == 'qa':
        return 'qa_judge'  # 默认QA Judge

    # 优先级3: 默认策略
    return 'default_judge'
```

---

## 配置文件说明

### 文件位置

**主配置文件**: `config/judge_prompts.yaml`

### 配置结构

```yaml
# 全局配置
global:
  model: "gpt-oss-120b"
  temperature: 0.0
  max_tokens: 200

# 各数据集配置
gsm8k:
  enabled: true
  answer_extraction: {...}
  judge_prompt: |...

math:
  enabled: true
  answer_extraction: {...}
  judge_prompt: |...

humaneval:
  enabled: false  # 使用测试执行
  evaluation_method: "test_execution"

# ... 其他数据集

# 数据集映射
dataset_mapping:
  by_source:
    gsm8k: "gsm8k"
    math: "math"
    # ...
```

### 关键配置项

#### 1. GSM8K配置

**特点**:
- Ground truth格式: `中间步骤 <<calc>> ... #### 最终答案`
- 需要提取 `####` 后的数字

**配置**:
```yaml
gsm8k:
  answer_extraction:
    priority:
      - "#### 后的数字"  # 最高优先级
      - "\\boxed{}内容"

    patterns:
      - regex: "####\\s*(-?\\d+\\.?\\d*)"
      - regex: "<<([^>]+>>"  # 忽略中间计算
        action: "ignore"

  judge_prompt: |
    **Special Rules for GSM8K**:
    1. Ground truth ends with "#### ANSWER"
    2. Extract ONLY the number after "####"
    3. Ignore "<<calc>>" markers
```

#### 2. Math配置

**特点**:
- Ground truth包含LaTeX: `\\frac{1}{2}`, `\\boxed{}`
- 有 `meta.short_answer` 字段（标准化答案）

**配置**:
```yaml
math:
  answer_extraction:
    priority:
      - "meta.short_answer字段"  # 优先使用！
      - "\\boxed{}内容"

    latex_normalization:
      enabled: true
      rules:
        - "\\frac{a}{b} → a/b"
        - "\\sqrt{x} → sqrt(x)"

  judge_prompt: |
    **Special Rules for MATH Dataset**:
    1. If meta.short_answer exists, use it as canonical answer
    2. Handle LaTeX: \\frac{1}{2} = 0.5
    3. Allow equivalent forms: 1/2 = 0.5 = 50%
```

#### 3. HumanEval/MBPP配置

**特点**:
- 有测试用例
- 应该执行代码验证，不用LLM Judge

**配置**:
```yaml
humaneval:
  enabled: false  # 禁用LLM Judge！
  evaluation_method: "test_execution"

  test_execution:
    timeout: 5.0
    sandbox: true

  # 仅在缺少测试用例时fallback到Judge
  fallback_judge_prompt: |
    ⚠️ WARNING: Test cases missing, using semantic comparison.
    This is NOT reliable.
```

#### 4. QA配置（HotpotQA, SQuAD, CommonsenseQA, MMLU）

**特点**:
- 多选题（ABCDE选项）
- 容易出现"推断"等价性的问题

**配置**:
```yaml
hotpotqa:
  judge_prompt: |
    🚫 **PROHIBITION #1: No Inference of Option Labels**
    - If Prediction="E" and Ground Truth="might dream":
      → Judge as **False** (Do NOT assume E means "might dream")
    - If Prediction="might dream" and Ground Truth="E":
      → Judge as **True** (Prediction matches option content)

    **Example**:
    Prediction: "E"
    Ground Truth: "might dream"
    → **False** ✅ Correct judgment

commonsenseqa:
  multiple_choice:
    enabled: true
    option_format: "A-E"

  judge_prompt: |
    **Evaluation Logic**:

    Case 1: Both are letters (A-E)
      → Simple comparison

    Case 2: Prediction is letter, Ground Truth is text
      → **False** (禁止推断)

    Case 3: Prediction is text, Ground Truth is letter
      → Check if text matches option content

    Case 4: Both are text
      → Normalize and compare
```

---

## 实施步骤

### 阶段0: 准备工作 ✅

- [x] 分析训练日志，识别误判模式
- [x] 创建误判分析报告 (`docs/MISJUDGMENT_ANALYSIS.md`)
- [x] 创建详细错误分析报告 (`docs/ERROR_PATTERNS_DETAILED.md`)
- [x] 设计数据集专属Judge配置 (`config/judge_prompts.yaml`)

### 阶段1: 核心实现（需要代码修改）⏸️

> ⚠️ **注意**: 根据您的要求，代码修改部分暂停。以下是实施计划，待您确认策略后再执行。

#### 1.1 修改 RewardComputer

**文件**: `src/reward_computer.py`

**任务**:
1. 添加配置加载方法
2. 实现数据集识别逻辑
3. 根据数据集选择Judge prompt
4. 为每种数据集添加专门的预处理

**伪代码**:
```python
class RewardComputer:
    def __init__(self, ...):
        # 加载Judge配置
        self.judge_config = self._load_judge_config()

    def _load_judge_config(self) -> Dict:
        """加载judge_prompts.yaml配置"""
        with open('config/judge_prompts.yaml') as f:
            return yaml.safe_load(f)

    def _select_judge_prompt(self, sample: Dict) -> str:
        """根据样本选择Judge prompt"""
        source = sample.get('source', '')
        problem_type = sample.get('problem_type', '')

        # 优先级：source > problem_type > default
        if source in self.judge_config['dataset_mapping']['by_source']:
            dataset = self.judge_config['dataset_mapping']['by_source'][source]
            return self.judge_config[dataset]['judge_prompt']

        # Fallback
        return self.judge_config['global']['output_format']

    def compute_reward(self, problem, prediction, ground_truth,
                      problem_type, metadata, test, entry_point):
        # 1. 识别数据集
        dataset = self._identify_dataset(sample)

        # 2. 选择评估策略
        if dataset in ['humaneval', 'mbpp'] and test:
            # Code: 使用测试执行
            return self._check_code_solution(...)

        elif dataset == 'gsm8k':
            # GSM8K: 提取####答案
            gt_extracted = self._extract_gsm8k_answer(ground_truth)
            pred_extracted = self._extract_answer(prediction)
            return self._llm_judge_compare(
                pred_extracted, gt_extracted,
                prompt_template=self.judge_config['gsm8k']['judge_prompt']
            )

        elif dataset == 'math':
            # Math: 优先使用short_answer
            gt_answer = metadata.get('short_answer', ground_truth)
            return self._llm_judge_compare(
                prediction, gt_answer,
                prompt_template=self.judge_config['math']['judge_prompt']
            )

        # ... 其他数据集
```

#### 1.2 增强 Answer Extractor

**文件**: `src/answer_extractor.py`

**任务**:
1. 识别并处理代码泄漏 (`\\boxed{def ...}`)
2. 提取GSM8K的 `####` 答案
3. 处理LaTeX格式
4. 标准化QA答案（小写、去标点）

**伪代码**:
```python
class AnswerExtractor:
    @staticmethod
    def extract_gsm8k_answer(text: str) -> str:
        """提取GSM8K的#### 后答案"""
        match = re.search(r'####\\s*(-?\\d+\\.?\\d*)', text)
        if match:
            return match.group(1)
        return text

    @staticmethod
    def extract_from_boxed(text: str) -> Optional[str]:
        """从\\boxed{}提取，处理特殊情况"""
        match = re.search(r'\\\\boxed\\{([^}]+)\\}', text)
        if match:
            content = match.group(1)

            # 检查是否是代码泄漏
            if content.startswith('def ') or 'return ' in content:
                logger.warning("检测到代码泄漏，尝试提取执行结果")
                return None  # 需要重新处理

            # 检查是否是错误信息
            if content.startswith('Error:'):
                logger.warning("检测到错误信息")
                return None

            return content
        return None

    @staticmethod
    def normalize_latex(text: str) -> str:
        """标准化LaTeX表达式"""
        # \\frac{a}{b} → a/b
        text = re.sub(r'\\\\frac\\{([^}]+)\\}\\{([^}]+)\\}', r'(\\1)/(\\2)', text)
        # \\sqrt{x} → sqrt(x)
        text = re.sub(r'\\\\sqrt\\{([^}]+)\\}', r'sqrt(\\1)', text)
        return text
```

#### 1.3 创建数据集验证器

**新文件**: `src/dataset_validators.py`

**任务**:
为每种数据集创建专门的验证器类

**伪代码**:
```python
class GSM8KValidator:
    """GSM8K数据集验证器"""

    @staticmethod
    def validate_answer(prediction: str, ground_truth: str) -> bool:
        # 1. 提取#### 后的数字
        gt_number = AnswerExtractor.extract_gsm8k_answer(ground_truth)

        # 2. 从预测中提取数字
        pred_number = AnswerExtractor.extract_number(prediction)

        # 3. 数值比较
        try:
            return abs(float(pred_number) - float(gt_number)) < 1e-4
        except ValueError:
            return False

class MathValidator:
    """Math数据集验证器"""

    @staticmethod
    def validate_answer(prediction: str, ground_truth: str,
                       short_answer: Optional[str] = None) -> bool:
        # 1. 优先使用short_answer
        if short_answer:
            target = short_answer
        else:
            target = AnswerExtractor.extract_from_boxed(ground_truth)

        # 2. LaTeX标准化
        pred_normalized = AnswerExtractor.normalize_latex(prediction)
        target_normalized = AnswerExtractor.normalize_latex(target)

        # 3. 数学等价性判断
        return MathEquivalence.check(pred_normalized, target_normalized)

class QAValidator:
    """QA数据集验证器"""

    @staticmethod
    def validate_answer(prediction: str, ground_truth: str) -> bool:
        # 规则1: 禁止选项推断
        if len(prediction) == 1 and prediction.isalpha():
            # 预测是单字母
            if len(ground_truth) > 1:
                # 真值是文本 → False
                return False
            else:
                # 真值也是字母 → 直接比较
                return prediction.upper() == ground_truth.upper()

        # 规则2: 标准化比较
        pred_norm = QAValidator.normalize(prediction)
        gt_norm = QAValidator.normalize(ground_truth)

        return pred_norm == gt_norm or pred_norm in gt_norm

    @staticmethod
    def normalize(text: str) -> str:
        # 小写
        text = text.lower()
        # 移除冠词
        text = re.sub(r'\\b(a|an|the)\\b', '', text)
        # 移除标点
        text = re.sub(r'[^\\w\\s]', '', text)
        # 去空格
        return text.strip()
```

#### 1.4 修复验证集阈值Bug

**文件**: `src/grpo_trainer.py`
**行号**: 737

**当前代码**:
```python
num_correct = sum(1 for score in correctness_scores if score >= 5.0)
```

**修复后**:
```python
num_correct = sum(1 for score in correctness_scores if score >= 0.9)
```

**影响**: 这个bug导致验证准确率��直显示为0%，修复后可以看到真实的验证性能

### 阶段2: 测试和验证

#### 2.1 单元测试

**创建**: `tests/test_dataset_judges.py`

```python
import pytest
from src.reward_computer import RewardComputer

class TestGSM8KJudge:
    def test_extract_final_answer(self):
        ground_truth = "Natalia sold 48/2 = <<48/2=24>>24...\\n#### 72"
        expected = "72"
        result = AnswerExtractor.extract_gsm8k_answer(ground_truth)
        assert result == expected

    def test_ignore_intermediate_calc(self):
        prediction = "24"
        ground_truth = "<<48/2=24>>...#### 72"
        # Should extract 72, not 24
        assert not GSM8KValidator.validate_answer(prediction, ground_truth)

class TestQAJudge:
    def test_prohibit_option_inference(self):
        """测试禁止选项推断"""
        prediction = "E"
        ground_truth = "might dream"
        # 应该判为False（禁止推断E=might dream）
        assert not QAValidator.validate_answer(prediction, ground_truth)

    def test_allow_text_match(self):
        """测试允许文本匹配"""
        prediction = "might dream"
        ground_truth = "E"
        # 应该判为True（文本匹配选项内容）
        assert QAValidator.validate_answer(prediction, ground_truth)
```

#### 2.2 集成测试

**创建**: `tests/test_reward_computer_integration.py`

```python
class TestRewardComputerIntegration:
    @pytest.fixture
    def reward_computer(self):
        return RewardComputer(config_path='config/training.yaml')

    def test_gsm8k_sample(self, reward_computer):
        sample = {
            'problem': '...',
            'source': 'gsm8k',
            'problem_type': 'math'
        }
        prediction = "The answer is 72."
        ground_truth = "...#### 72"

        reward = reward_computer.compute_reward(
            problem=sample['problem'],
            prediction=prediction,
            ground_truth=ground_truth,
            problem_type='math',
            metadata={},
            test='',
            entry_point=''
        )

        assert reward == 1.0  # Should be correct

    # ... 更多测试
```

#### 2.3 回归测试

**使用已知的误判案例**:

```python
# tests/test_misjudgment_cases.py

MISJUDGMENT_CASES = [
    {
        'name': 'Drawstring Bag',
        'prediction': 'D. tied up',
        'ground_truth': 'A. safe',
        'expected': 0.0,  # 不同选项
        'dataset': 'commonsenseqa'
    },
    {
        'name': 'Code Format Output',
        'prediction': '\\boxed{def solve(): return 50}',
        'ground_truth': '50',
        'expected': 0.0,  # 格式错误
        'dataset': 'math'
    },
    {
        'name': 'Option Inference Prohibited',
        'prediction': 'E',
        'ground_truth': 'might dream',
        'expected': 0.0,  # 禁止推断
        'dataset': 'commonsenseqa'
    },
    # ... 添加所有12个已知误判案例
]

@pytest.mark.parametrize('case', MISJUDGMENT_CASES)
def test_misjudgment_case(case, reward_computer):
    """测试已知误判案例是否被修复"""
    reward = reward_computer.compute_reward(
        problem='',
        prediction=case['prediction'],
        ground_truth=case['ground_truth'],
        problem_type=case.get('problem_type', 'qa'),
        metadata={'source': case['dataset']},
        test='',
        entry_point=''
    )

    assert reward == case['expected'], f"Failed on case: {case['name']}"
```

### 阶段3: 部署和监控

#### 3.1 生成改进报告

**运行训练**:
```bash
python train.py --config config/training.yaml
```

**对比指标**:
- 修复前准确率: 64.9%
- 修复后准确率: ?
- 误判率变化: 12-20% → ?

**生成报告**: `docs/evaluation_improvement_report.md`

#### 3.2 持续监控

在 `src/reward_computer.py` 中添加监控:

```python
class RewardComputer:
    def __init__(self, ...):
        self.eval_stats = {
            'gsm8k': {'calls': 0, 'success': 0, 'failures': 0},
            'math': {'calls': 0, 'success': 0, 'failures': 0},
            'qa': {'calls': 0, 'success': 0, 'failures': 0},
            # ...
        }

    def compute_reward(self, ...):
        dataset = self._identify_dataset(sample)
        self.eval_stats[dataset]['calls'] += 1

        try:
            reward = self._evaluate(...)
            self.eval_stats[dataset]['success'] += 1
            return reward
        except Exception as e:
            self.eval_stats[dataset]['failures'] += 1
            logger.error(f"Evaluation failed for {dataset}: {e}")
            raise

    def print_stats(self):
        """打印评估统计"""
        for dataset, stats in self.eval_stats.items():
            success_rate = stats['success'] / stats['calls'] if stats['calls'] > 0 else 0
            print(f"{dataset}: {stats['calls']} calls, {success_rate:.2%} success")
```

---

## 预期效果

### 定量指标

| 指标 | 当前 | 预期 | 提升 |
|------|------|------|------|
| **总准确率** | 64.9% | 70-75% | +5.1-10.1% |
| **误判率** | 12-20% | <5% | -60-75% |
| **Math准确率** | 波动大 | +5-8% | 稳定性提升 |
| **Code准确率** | 80-100% | +3-5% | 接近上限 |
| **QA准确率** | 相对稳定 | +2-3% | 小幅提升 |

### 定性改进

1. **GSM8K**: 准确提取 `####` 后的最终答案，不受中间步骤干扰
2. **Math**: 优先使用 `meta.short_answer`，处理LaTeX格式
3. **Code**: 完全依赖测试执行，不再误判变量名差异
4. **QA**: 禁止推断选项等价性，减少"脑补"错误
5. **格式鲁棒性**: 识别代码泄漏、空输出等异常格式

### 案例对比

#### 案例1: GSM8K #### 提取

**修复前**:
```python
Prediction: "24"
Ground Truth: "Natalia sold 48/2 = <<48/2=24>>24...#### 72"
Judge: 1.0 ✅  # 错误：提取到中间计算24
```

**修复后**:
```python
Prediction: "24"
Ground Truth: "...#### 72"
Extracted GT: "72"
Judge: 0.0 ❌  # 正确：24 != 72
```

#### 案例2: QA选项推断

**修复前**:
```python
Prediction: "E"
Ground Truth: "might dream"
Judge: 1.0 ✅  # 错误：Judge推断E=might dream
```

**修复后**:
```python
Prediction: "E"
Ground Truth: "might dream"
Judge: 0.0 ❌  # 正确：禁止推断
```

#### 案例3: 代码格式输出

**修复前**:
```python
Prediction: "\\boxed{def solve(): return 50}"
Ground Truth: "50"
Judge: 0.0 ❌  # 正确判错，但原因不明
```

**修复后**:
```python
Prediction: "\\boxed{def solve(): return 50}"
Extracted: None  # 识别为代码泄漏
Judge: 0.0 ❌  # 正确判错，且记录原因："代码泄漏"
```

---

## FAQ

### Q1: 为什么不直接修改代码，而是先创建配置文件？

**A**: 设计先行，确保策略正确后再实施。配置文件作为设计文档，明确了每种数据集的评估规则，避免代码实现时遗漏或误解需求。

### Q2: 如果新增数据集，如何添加支持？

**A**: 在 `config/judge_prompts.yaml` 中添加新的数据集配置即可：

```yaml
new_dataset:
  enabled: true
  description: "新数据集描述"
  answer_extraction: {...}
  judge_prompt: |...
```

然后在 `dataset_mapping` 中注册：

```yaml
dataset_mapping:
  by_source:
    new_dataset: "new_dataset"
```

### Q3: 配置文件修改后需要重启训练吗？

**A**: 是的。配置文件在 `RewardComputer` 初始化时加载，修改配置需要重新启动训练进程。

**未来改进**: 可以实现配置热重载，无需重启。

### Q4: 如果LLM Judge解析失败怎么办？

**A**: 实现了多层fallback机制：

1. **重试**: 解析失败时重试1次（已实现）
2. **Fallback到规则**: 如果Judge失败，使用基于规则的比较（如Token F1）
3. **记录日志**: 所有失败案例记录到日志，定期review

### Q5: Code题为什么禁用LLM Judge？

**A**: Code题有明确的测试用例，执行验证是最准确的方法。LLM Judge只会在以下情况fallback使用：

- 测试用例缺失
- 代码执行超时
- 代码语法错误无法执行

即使fallback到Judge，也会在日志中标记 `⚠️ WARNING: Using semantic comparison for code task`。

### Q6: 如何监控Judge性能？

**A**: 实现了多维度监控：

1. **成功率**: 每种Judge的解析成功率
2. **延迟**: 平均响应时间
3. **一致性**: 与fallback方法的一致性
4. **采样日志**: 10%的判定记录详细日志供审查

统计数据保存在 `RewardComputer.eval_stats`，训练结束时打印汇总。

### Q7: 如果发现新的误判模式怎么办？

**A**: 流程：

1. **记录案例**: 添加到 `evaluation_cases/misjudged_cases.jsonl`
2. **分析原因**: 更新 `docs/MISJUDGMENT_ANALYSIS.md`
3. **修改配置**: 在 `config/judge_prompts.yaml` 中调整相关Judge prompt
4. **添加测试**: 在 `tests/test_misjudgment_cases.py` 中添加回归测试
5. **验证修复**: 运行测试确保修复生效

### Q8: 准确率提升不到预期怎么办？

**A**: 诊断步骤：

1. **检查配置**: 确认配置文件加载成功
2. **验证识别**: 检查数据集识别是否正确
   ```python
   # 添加调试日志
   logger.info(f"Sample source: {sample.get('source')}, using judge: {judge_name}")
   ```
3. **分析失败案例**: 查看新的误判模式
4. **A/B测试**: 对比新旧Judge的判定差异
5. **增量修复**: 逐个数据集启用优化，定位问题

---

## 总结

本指南提供了完整的实施方案，从问题分析、设计方案、配置文件到测试验证。

**关键要点**:

1. ✅ **已完成**: 分析文档、配置文件、实施指南
2. ⏸️ **待实施**: 代码修改（等待策略确认）
3. 🎯 **目标**: 误判率<5%，准确率70-75%
4. 📊 **监控**: 持续跟踪各Judge性能
5. 🔄 **迭代**: 发现新问题及时修复

**下一步**:
请审查本指南和配置文件，确认策略后我们再进行代码修改和测试。

---

## 相关文档

- [误判分析报告](./MISJUDGMENT_ANALYSIS.md)
- [详细错误分析](./ERROR_PATTERNS_DETAILED.md)
- [Judge配置文件](../config/judge_prompts.yaml)
- [训练日志](../logs/train_restored_v10.log)

**文档结束**
