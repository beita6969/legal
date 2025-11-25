# WandB监控系统设计方案

## 📊 系统研究总结

### 1. 当前训练架构分析

#### 训练循环结构 (`src/grpo_trainer.py`)
- **主训练循环**: `train()` 方法 (第763-811行)
  - 每个step调用 `train_step()`
  - 支持验证集评估 (`evaluate_on_val_set()`)
  - 定期保存检查点

- **单步训练**: `train_step()` 方法 (第278-515行)
  - 采样batch (支持混合数据集)
  - 为每个问题生成K个工作流 (GRPO组)
  - 执行工作流并计算奖励
  - 策略梯度更新

#### 数据集架构
当前系统支持以下数据集:

| 数据集 | 类型 | source字段 | 评估方式 |
|--------|------|-----------|----------|
| GSM8K | math | "gsm8k" | 数值匹配 + LLM Judge |
| MATH | math | "math" | 数值匹配 + LLM Judge |
| HumanEval | code | "humaneval" | 测试执行 |
| MBPP | code | "mbpp" | 测试执行 (已过滤) |
| HotpotQA | qa | "hotpotqa" | Token匹配 + LLM Judge |
| CommonsenseQA | qa | "commonsenseqa" | 选项匹配 + LLM Judge |
| MMLU | qa | "mmlu" | 选项匹配 + LLM Judge |

#### 奖励计算系统 (`src/reward_computer.py`)
- **二元奖励**: 正确=1.0, 错误=0.0
- **LLM Judge**: 使用GPT OSS 120B @ port 8002
- **数据集专属Prompt**: 通过 `source` 字段选择专属评估策略
- **统计计数器**: `eval_stats` 跟踪LLM Judge成功率

#### 当前WandB集成状态
- ✅ 基础集成已完成 (第71-132行)
- ✅ 训练指标记录 (第492-513行)
- ✅ 样本级记录 (第379-385行)
- ❌ **缺少**: 数据集维度的细分统计
- ❌ **缺少**: LLM Judge性能监控
- ❌ **缺少**: 验证集详细分析

---

## 🎯 监控指标设计

### 1. 核心训练指标 (已实现)

```python
wandb.log({
    "train/loss": loss,
    "train/kl_div": kl_div,
    "train/avg_reward": np.mean(all_rewards),
    "train/max_reward": np.max(all_rewards),
    "train/min_reward": np.min(all_rewards),
    "train/accuracy": accuracy,  # 总体准确率
    "train/temperature": current_temp,
    "train/step": step,
})
```

### 2. 问题类型维度 (已实现)

```python
for ptype in ['math', 'code', 'qa']:
    wandb.log({
        f"train/accuracy_{ptype}": stats['accuracy'],
        f"train/avg_score_{ptype}": stats['avg_score'],
        f"train/count_{ptype}": stats['count'],
    })
```

### 3. **数据集维度 (需新增)** ⭐

这是关键改进点！需要为每个数据集单独统计准确率:

```python
# 目标监控结构
wandb.log({
    # GSM8K
    "dataset/gsm8k/accuracy": 0.85,
    "dataset/gsm8k/count": 20,
    "dataset/gsm8k/avg_reward": 0.85,

    # MATH
    "dataset/math/accuracy": 0.42,
    "dataset/math/count": 15,
    "dataset/math/avg_reward": 0.42,

    # HotpotQA
    "dataset/hotpotqa/accuracy": 0.68,
    "dataset/hotpotqa/count": 18,
    "dataset/hotpotqa/avg_reward": 0.68,

    # HumanEval
    "dataset/humaneval/accuracy": 0.55,
    "dataset/humaneval/count": 12,
    "dataset/humaneval/avg_reward": 0.55,

    # ... (其他数据集)
})
```

### 4. LLM Judge性能监控 (需新增)

```python
wandb.log({
    # Judge成功率
    "judge/success_rate": judge_success / total_evals,
    "judge/parse_failure_rate": parse_failures / total_evals,
    "judge/api_failure_rate": api_failures / total_evals,

    # Judge判决分布
    "judge/correct_ratio": correct_preds / (correct_preds + incorrect_preds),
    "judge/total_calls": total_evals,

    # 按数据集的Judge性能
    "judge/gsm8k_success_rate": ...,
    "judge/hotpotqa_success_rate": ...,
})
```

### 5. 验证集详细监控 (需增强)

```python
# 验证集总体指标
wandb.log({
    "val/accuracy": val_accuracy,
    "val/avg_correctness": avg_correctness,
    "val/success_rate": success_rate,
    "val/avg_cost": avg_cost,
})

# 验证集按数据集分解
for source in ['gsm8k', 'math', 'hotpotqa', 'humaneval', ...]:
    wandb.log({
        f"val/{source}/accuracy": ...,
        f"val/{source}/count": ...,
        f"val/{source}/avg_cost": ...,
    })
```

### 6. 成本统计监控

```python
wandb.log({
    "cost/total_cost": cumulative_cost,
    "cost/avg_cost_per_sample": avg_cost,
    "cost/executor_calls": total_executor_calls,
    "cost/judge_calls": total_judge_calls,
})
```

---

## 🔧 实现方案

### 方案A: 最小侵入式改进 (推荐)

在 `train_step()` 中添加数据集维度统计:

```python
# 在train_step()中 (第456行后)

# 3. 按数据集统计准确率 (新增)
dataset_stats = {}  # {source: {'correct': [], 'rewards': []}}

for sample_idx, sample in enumerate(batch):
    source = sample.get('source', 'unknown')  # 获取数据集来源

    # 初始化数据集统计
    if source not in dataset_stats:
        dataset_stats[source] = {'correct': [], 'rewards': []}

    # ... (执行工作流，计算奖励)

    # 记录到数据集统计
    dataset_stats[source]['correct'].append(correctness > 0.9)
    dataset_stats[source]['rewards'].append(reward)

# 4. 计算数据集维度指标
for source, stats in dataset_stats.items():
    if stats['correct']:
        dataset_accuracy = sum(stats['correct']) / len(stats['correct']) * 100
        dataset_avg_reward = np.mean(stats['rewards'])

        wandb.log({
            f"dataset/{source}/accuracy": dataset_accuracy,
            f"dataset/{source}/count": len(stats['correct']),
            f"dataset/{source}/avg_reward": dataset_avg_reward,
        }, step=step)
```

### 方案B: 全面重构 (可选)

创建专门的 `MetricsCollector` 类:

```python
class MetricsCollector:
    """统一的指标收集器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = []
        self.by_problem_type = defaultdict(list)
        self.by_dataset = defaultdict(list)
        self.judge_stats = {
            'success': 0, 'parse_failures': 0, 'api_failures': 0
        }

    def add_sample(self, sample: Dict, result: Dict):
        """添加单个样本结果"""
        self.samples.append(result)

        # 按问题类型
        ptype = sample['problem_type']
        self.by_problem_type[ptype].append(result)

        # 按数据集
        source = sample.get('source', 'unknown')
        self.by_dataset[source].append(result)

    def get_wandb_logs(self, step: int) -> Dict:
        """生成WandB日志字典"""
        logs = {'step': step}

        # 总体指标
        logs['train/accuracy'] = self._compute_accuracy(self.samples)

        # 问题类型维度
        for ptype, results in self.by_problem_type.items():
            logs[f'train/accuracy_{ptype}'] = self._compute_accuracy(results)

        # 数据集维度
        for source, results in self.by_dataset.items():
            logs[f'dataset/{source}/accuracy'] = self._compute_accuracy(results)
            logs[f'dataset/{source}/count'] = len(results)

        return logs

    def _compute_accuracy(self, results: List[Dict]) -> float:
        if not results:
            return 0.0
        correct = sum(1 for r in results if r.get('correctness', 0) > 0.9)
        return correct / len(results) * 100
```

---

## 📈 WandB仪表板配置

### 1. 训练总览面板

```yaml
panels:
  - type: line
    title: "Training Loss & KL Divergence"
    metrics:
      - train/loss
      - train/kl_div

  - type: line
    title: "Overall Accuracy"
    metrics:
      - train/accuracy
      - val/accuracy

  - type: scalar
    title: "Current Step"
    metric: train/step
```

### 2. 数据集性能面板 (新增)

```yaml
panels:
  - type: bar
    title: "Accuracy by Dataset (Latest)"
    metrics:
      - dataset/gsm8k/accuracy
      - dataset/math/accuracy
      - dataset/hotpotqa/accuracy
      - dataset/humaneval/accuracy
      - dataset/commonsenseqa/accuracy
      - dataset/mmlu/accuracy

  - type: line
    title: "GSM8K Performance Over Time"
    metrics:
      - dataset/gsm8k/accuracy
      - dataset/gsm8k/avg_reward

  - type: line
    title: "MATH Performance Over Time"
    metrics:
      - dataset/math/accuracy
      - dataset/math/avg_reward

  - type: table
    title: "Dataset Statistics"
    columns:
      - source
      - accuracy
      - count
      - avg_reward
```

### 3. LLM Judge监控面板 (新增)

```yaml
panels:
  - type: line
    title: "LLM Judge Success Rate"
    metrics:
      - judge/success_rate
      - judge/parse_failure_rate
      - judge/api_failure_rate

  - type: pie
    title: "Judge Verdict Distribution"
    metrics:
      - judge/correct_predictions
      - judge/incorrect_predictions

  - type: bar
    title: "Judge Performance by Dataset"
    metrics:
      - judge/gsm8k_success_rate
      - judge/math_success_rate
      - judge/hotpotqa_success_rate
```

### 4. 验证集详细面板

```yaml
panels:
  - type: line
    title: "Validation Accuracy by Dataset"
    metrics:
      - val/gsm8k/accuracy
      - val/math/accuracy
      - val/hotpotqa/accuracy
      - val/humaneval/accuracy

  - type: scatter
    title: "Validation Cost vs Accuracy"
    x_axis: val/avg_cost
    y_axis: val/accuracy
```

---

## 🚀 实现步骤

### Step 1: 数据集维度统计 (优先级: P0)

**文件**: `src/grpo_trainer.py`

**修改位置**: `train_step()` 方法 (第278-515行)

**代码改动**:

```python
# 在第307行后添加
dataset_stats = defaultdict(lambda: {'correctness': [], 'rewards': []})

# 在第312行的循环中 (for sample_idx, sample in enumerate...)
source = sample.get('source', 'unknown')  # 获取数据集来源

# 在第370-393行 (计算correctness后)
# 记录到数据集统计
dataset_stats[source]['correctness'].append(correctness)
dataset_stats[source]['rewards'].append(reward)

# 在第486行后 (metrics字典定义后) 添加
# 计算数据集维度指标
for source, stats in dataset_stats.items():
    if stats['correctness']:
        source_correct = sum(1 for c in stats['correctness'] if c >= 0.9)
        source_total = len(stats['correctness'])
        source_accuracy = (source_correct / source_total * 100) if source_total > 0 else 0.0
        source_avg_reward = np.mean(stats['rewards'])

        metrics[f'dataset_{source}_accuracy'] = source_accuracy
        metrics[f'dataset_{source}_count'] = source_total
        metrics[f'dataset_{source}_avg_reward'] = source_avg_reward

# 在第493行后 (wandb_log_data定��后) 添加
# 添加数据集维度指标到wandb
for source, stats in dataset_stats.items():
    if stats['correctness']:
        source_correct = sum(1 for c in stats['correctness'] if c >= 0.9)
        source_total = len(stats['correctness'])
        source_accuracy = (source_correct / source_total * 100) if source_total > 0 else 0.0

        wandb_log_data[f"dataset/{source}/accuracy"] = source_accuracy
        wandb_log_data[f"dataset/{source}/count"] = source_total
        wandb_log_data[f"dataset/{source}/avg_reward"] = np.mean(stats['rewards'])
```

### Step 2: LLM Judge性能监控 (优先级: P1)

**文件**: `src/reward_computer.py`

**修改位置**: 在 `train_step()` 中定期读取judge统计

**代码改动**:

```python
# 在 grpo_trainer.py 的 train_step() 末尾 (第513行后)

# 获取LLM Judge统计 (如果启用)
if self.reward_computer.use_llm_judge:
    judge_stats = self.reward_computer.eval_stats
    total = judge_stats['total_evaluations']

    if total > 0:
        wandb_log_data['judge/success_rate'] = judge_stats['llm_judge_success'] / total
        wandb_log_data['judge/parse_failure_rate'] = judge_stats['llm_judge_parse_failures'] / total
        wandb_log_data['judge/api_failure_rate'] = judge_stats['llm_judge_api_failures'] / total
        wandb_log_data['judge/total_calls'] = total

        judged = judge_stats['correct_predictions'] + judge_stats['incorrect_predictions']
        if judged > 0:
            wandb_log_data['judge/correct_ratio'] = judge_stats['correct_predictions'] / judged
```

### Step 3: 验证集数据集分解 (优先级: P1)

**文件**: `src/grpo_trainer.py`

**修改位置**: `evaluate_on_val_set()` 方法 (第649-761行)

**代码改动**:

```python
# 在第674行后添加
val_dataset_stats = defaultdict(lambda: {'correctness': [], 'cost': []})

# 在第678行的循环中
source = sample.get('source', 'unknown')

# 在第720-732行 (计算correctness后)
val_dataset_stats[source]['correctness'].append(correctness)
val_dataset_stats[source]['cost'].append(cost)

# 在第752行后 (metrics定义后) 添加
# 计算验证集数据集维度指标
for source, stats in val_dataset_stats.items():
    if stats['correctness']:
        source_correct = sum(1 for c in stats['correctness'] if c >= 0.9)
        source_total = len(stats['correctness'])
        source_accuracy = (source_correct / source_total * 100) if source_total > 0 else 0.0

        metrics[f'val_{source}_accuracy'] = source_accuracy
        metrics[f'val_{source}_count'] = source_total
        metrics[f'val_{source}_avg_cost'] = np.mean(stats['cost'])

# 在第800行 (wandb.log调用处) 添加数据集维度日志
val_dataset_logs = {}
for source, stats in val_dataset_stats.items():
    if stats['correctness']:
        source_correct = sum(1 for c in stats['correctness'] if c >= 0.9)
        source_total = len(stats['correctness'])
        source_accuracy = (source_correct / source_total * 100) if source_total > 0 else 0.0

        val_dataset_logs[f"val/{source}/accuracy"] = source_accuracy
        val_dataset_logs[f"val/{source}/count"] = source_total
        val_dataset_logs[f"val/{source}/avg_cost"] = np.mean(stats['cost'])

wandb.log(val_dataset_logs, step=step)
```

### Step 4: 成本统计累积 (优先级: P2)

**文件**: `src/grpo_trainer.py`

**修改位置**: `__init__()` 和 `train_step()`

**代码改动**:

```python
# 在 __init__() 中添加 (第79行后)
self.cumulative_stats = {
    'total_cost': 0.0,
    'total_samples': 0,
    'executor_calls': 0,
    'judge_calls': 0,
}

# 在 train_step() 中累积统计 (第353行后)
self.cumulative_stats['total_cost'] += cost
self.cumulative_stats['executor_calls'] += 1

# 在 wandb.log() 前添加 (第513行前)
wandb_log_data['cost/total_cost'] = self.cumulative_stats['total_cost']
wandb_log_data['cost/avg_cost_per_sample'] = (
    self.cumulative_stats['total_cost'] / max(self.cumulative_stats['total_samples'], 1)
)
wandb_log_data['cost/executor_calls'] = self.cumulative_stats['executor_calls']
```

---

## 📊 预期效果

### 1. WandB Dashboard预览

实施后，WandB仪表板将显示:

#### 训练面板
- ✅ 总体准确率曲线
- ✅ Loss和KL divergence曲线
- ✅ 问题类型准确率 (math/code/qa)
- 🆕 **数据集准确率** (GSM8K, MATH, HotpotQA等)
- 🆕 **LLM Judge性能监控**

#### 验证面板
- ✅ 验证集总体准确率
- 🆕 **验证集数据集分解**
- 🆕 **成本vs准确率分析**

#### 统计面板
- 🆕 **数据集样本分布**
- 🆕 **Judge成功率统计**
- 🆕 **累积成本追踪**

### 2. 关键问题的可见性

实施后可以回答:

1. **"GSM8K准确率是多少？"** → `dataset/gsm8k/accuracy`
2. **"MATH数据集性能如何？"** → `dataset/math/accuracy`
3. **"HotpotQA的LLM Judge成功率？"** → `judge/hotpotqa_success_rate`
4. **"验证集在各数据集的表现？"** → `val/{source}/accuracy`
5. **"累积训练成本？"** → `cost/total_cost`

### 3. 性能开销

预计开销:
- **计算**: < 1% (仅统计操作)
- **WandB上传**: ~ 2-3 KB/step (新增指标)
- **内存**: < 10 MB (临时统计字典)

---

## 🔍 测试与验证

### 测试步骤

1. **本地测试** (offline模式)
   ```bash
   # 修改config/training.yaml
   wandb:
     enabled: true
     mode: offline  # 本地测试

   # 运行训练
   python train.py

   # 检查离线日志
   ls wandb/offline-run-*
   ```

2. **验证指标完整性**
   ```python
   # 检查wandb日志中是否包含所有新指标
   import wandb
   run = wandb.Api().run("project/run_id")

   # 检查数据集维度指标
   assert 'dataset/gsm8k/accuracy' in run.summary
   assert 'dataset/math/accuracy' in run.summary
   assert 'dataset/hotpotqa/accuracy' in run.summary

   # 检查Judge统计
   assert 'judge/success_rate' in run.summary
   ```

3. **在线测试** (online模式)
   ```bash
   # 配置正式环境
   wandb:
     enabled: true
     mode: online
     project: "aflow-roll-integration"

   # 运行训练
   python train.py
   ```

### 验证清单

- [ ] 训练步正常记录数据集维度指标
- [ ] 验证步正常记录数据集维度指标
- [ ] LLM Judge统计正确累积
- [ ] 成本统计正确累积
- [ ] WandB仪表板可正常可视化
- [ ] 离线模式可正常工作
- [ ] 在线模式可正常同步

---

## 📝 配置更新

**文件**: `config/training.yaml`

```yaml
# WandB配置更新
wandb:
  enabled: true
  project: "aflow-roll-integration"
  entity: "yao110002-sdfsdfsdfsdf-com"
  api_key: "b42ca0000cf06f97b05eba34f58823ad5f3122a4"
  mode: "online"  # online或offline

  # 新增：自定义仪表板配置
  dashboard:
    # 数据集列表 (用于自动生成监控面板)
    datasets:
      - gsm8k
      - math
      - hotpotqa
      - humaneval
      - commonsenseqa
      - mmlu

    # 监控频率
    log_frequency: 1  # 每step记录

    # 是否启用详细调试日志
    debug_logging: false
```

---

## 🎯 总结

### 核心改进

1. **数据集维度监控** (P0)
   - 为每个数据集单独统计准确率
   - 支持GSM8K, MATH, HotpotQA等所有数据集
   - 训练集和验证集均支持

2. **LLM Judge性能监控** (P1)
   - 成功率、失败率统计
   - 判决分布分析
   - 按数据集的Judge性能

3. **验证集详细分析** (P1)
   - 按数据集分解的验证性能
   - 成本vs准确率关联分析

4. **成本统计** (P2)
   - 累积成本追踪
   - 平均成本per样本
   - Executor/Judge调用次数

### 实施优先级

1. **P0 (立即实施)**: 数据集维度统计
2. **P1 (短期)**: LLM Judge监控 + 验证集分解
3. **P2 (中期)**: 成本统计累积

### 预期收益

- ✅ **完整的数据集级性能可见性**
- ✅ **LLM Judge质量监控**
- ✅ **验证集详细诊断能力**
- ✅ **成本追踪和优化依据**

---

## 附录

### A. 数据集source字段映射

```python
DATASET_SOURCE_MAPPING = {
    # Math datasets
    'gsm8k': 'math',
    'math': 'math',

    # Code datasets
    'humaneval': 'code',
    'mbpp': 'code',  # (已过滤)

    # QA datasets
    'hotpotqa': 'qa',
    'commonsenseqa': 'qa',
    'mmlu': 'qa',
}
```

### B. WandB API查询示例

```python
import wandb

api = wandb.Api()
run = api.run("entity/project/run_id")

# 查询特定数据集的准确率历史
history = run.history(keys=['dataset/gsm8k/accuracy'])
print(history)

# 查询最新的所有数据集准确率
for source in ['gsm8k', 'math', 'hotpotqa', 'humaneval']:
    accuracy = run.summary.get(f'dataset/{source}/accuracy', 0)
    print(f"{source}: {accuracy:.2f}%")
```

### C. 故障排查

**问题1**: 数据集统计为空
- **原因**: batch中缺少该数据集的样本
- **解决**: 检查 `domain_ratios` 配置

**问题2**: LLM Judge统计不更新
- **原因**: `use_llm_judge=False`
- **解决**: 在 `config/training.yaml` 中启用Judge

**问题3**: WandB离线日志未生成
- **原因**: `mode` 未设置为 `offline`
- **解决**: 修改配置文件或设置环境变量 `WANDB_MODE=offline`

---

**文档版本**: v1.0
**创建时间**: 2025-11-23
**维护者**: AI Training Team
