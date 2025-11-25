# WandB监控系统 - 快速开始指南

## 📚 文档概览

本项目提供了完整的WandB监控系统设计和实现代码。包含以下文档:

1. **设计文档**: `docs/WANDB_MONITORING_DESIGN.md` - 详细的系统分析和设计方案
2. **实现代码**: `src/wandb_metrics_collectors.py` - 可复用的指标收集工具类
3. **补丁指南**: `docs/WANDB_PATCH_GUIDE.md` - 手动应用补丁的详细步骤
4. **补丁脚本**: `scripts/apply_wandb_patch.py` - 自动/半自动应用补丁的工具

---

## 🎯 核心功能

### 新增监控维度

✅ **数据集维度统计** (P0)
- 为每个数据集单独统计准确率
- 支持: GSM8K, MATH, HotpotQA, HumanEval, CommonsenseQA, MMLU
- 训练集和验证集均支持

✅ **LLM Judge性能监控** (P1)
- 成功率、失败率统计
- 判决分布分析
- 按数据集的Judge性能

✅ **验证集详细分析** (P1)
- 按数据集分解的验证性能
- 成本vs准确率关联分析

✅ **成本统计** (P2)
- 累积成本追踪
- 平均成本per样本
- Executor/Judge调用次数

### WandB仪表板预览

实施后可监控的关键指标:

```yaml
# 数据集准确率
dataset/gsm8k/accuracy: 85.2%
dataset/math/accuracy: 42.1%
dataset/hotpotqa/accuracy: 68.3%
dataset/humaneval/accuracy: 55.7%

# LLM Judge性能
judge/success_rate: 0.85
judge/parse_failure_rate: 0.10
judge/api_failure_rate: 0.05
judge/correct_ratio: 0.706

# 成本统计
cost/total_cost: $12.34
cost/avg_cost_per_sample: $0.0123
cost/executor_calls: 1000
cost/judge_calls: 800

# 验证集分解
val/gsm8k/accuracy: 83.5%
val/math/accuracy: 40.2%
val/hotpotqa/accuracy: 65.8%
```

---

## 🚀 快速实施

### 方式1: 测试工具类 (推荐先做)

```bash
# 1. 测试指标收集器
python3 src/wandb_metrics_collectors.py

# 预期输出:
# 🧪 测试DatasetMetricsCollector
# WandB日志:
#   dataset/gsm8k/accuracy: 66.66666666666666
#   dataset/gsm8k/count: 3
#   ...
# ✅ 所有测试通过！
```

### 方式2: 手动应用补丁 (推荐)

```bash
# 1. 备份原文件
cp src/grpo_trainer.py src/grpo_trainer.py.backup

# 2. 查看手动补丁指南
cat docs/WANDB_PATCH_GUIDE.md

# 3. 使用编辑器打开grpo_trainer.py
vim src/grpo_trainer.py
# 或
code src/grpo_trainer.py

# 4. 按照指南逐个应用10个补丁
# Patch 1-10 的具体位置和代码见 WANDB_PATCH_GUIDE.md

# 5. 验证语法
python3 -m py_compile src/grpo_trainer.py

# 6. 测试导入
python3 -c "from src.grpo_trainer import GRPOTrainer; print('✅ 导入成功')"
```

### 方式3: 半自动应用补丁 (实验性)

```bash
# 1. 运行补丁脚本 (dry run)
python3 scripts/apply_wandb_patch.py

# 输出会显示:
# ✅ Patch 1: 添加导入语句
# ✅ Patch 2: 在__init__中添加成本追踪器
# ✅ Patch 3: 在train_step开始添加数据集收集器
# ⚠️  Patch 4-5: 需要手动添加...
# ✅ Patch 6: 在train_step末尾添加wandb日志
# ⚠️  Patch 7-10: 需要手动添加...

# 2. 如果满意，实际应用补丁
python3 -c "
from scripts.apply_wandb_patch import apply_patch_to_file
apply_patch_to_file('src/grpo_trainer.py', dry_run=False)
"

# 3. 手动完成剩余的Patch 4-5和7-10 (参考WANDB_PATCH_GUIDE.md)
```

---

## 📊 验证实施

### 步骤1: 检查语法

```bash
python3 -m py_compile src/grpo_trainer.py
```

### 步骤2: 测试导入

```bash
python3 -c "
from src.grpo_trainer import GRPOTrainer
from src.wandb_metrics_collectors import DatasetMetricsCollector
print('✅ 所有导入成功')
"
```

### 步骤3: 运行离线测试

修改 `config/training.yaml`:

```yaml
wandb:
  enabled: true
  mode: offline  # 离线模式测试
  project: "aflow-roll-integration"
```

运行训练:

```bash
python3 train.py
```

检查日志输出:

```bash
# 应该看到新的统��输出:
# 📊 数据集统计摘要:
#   gsm8k          :   2/  3 =  66.7%
#   math           :   1/  2 =  50.0%
#
# 🤖 LLM Judge统计 (总计: 100 次):
#   成功: 85 (85.0%)
#   ...
#
# 💰 成本统计:
#   总成本: $0.0350
#   ...
```

### 步骤4: 检查WandB离线日志

```bash
# 查看离线日志目录
ls -la wandb/

# 使用wandb CLI同步日志
wandb sync wandb/offline-run-*

# 或者直接查看日志内容
cat wandb/offline-run-*/files/wandb-summary.json | jq .
```

### 步骤5: 在线测试

确认离线测试无误后，修改配置:

```yaml
wandb:
  enabled: true
  mode: online  # 在线模式
  project: "aflow-roll-integration"
  api_key: "your_api_key_here"
```

运行训练并访问WandB仪表板:

```bash
python3 train.py

# 训练开始后会打印URL:
# ✅ wandb初始化完成
#   模式: online
#   项目: aflow-roll-integration
#   Run名称: grpo-training-20251123-120000
#   Run URL: https://wandb.ai/your-entity/aflow-roll-integration/runs/xxx
```

---

## 🔍 故障排查

### 问题1: 导入错误

```
ImportError: cannot import name 'DatasetMetricsCollector'
```

**解决方案**:
```bash
# 检查文件是否存在
ls src/wandb_metrics_collectors.py

# 检查Python路径
python3 -c "import sys; print('\n'.join(sys.path))"

# 如果需要，添加到PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### 问题2: 数据集统计为空

**原因**: batch中缺少该数据集的样本

**解决方案**:
```yaml
# 检查config/training.yaml中的domain_ratios
domain_ratios:
  math: 0.4
  code: 0.3
  qa: 0.3

# 确保数据文件存在
ls data/mixed/train_mixed*.jsonl

# 检查数据中的source字段
head -3 data/mixed/train_mixed_with_math.jsonl | jq .source
```

### 问题3: LLM Judge统计不更新

**原因**: `use_llm_judge=False`

**解决方案**:
```python
# 检查reward_computer初始化
# src/grpo_trainer.py 第198行
self.reward_computer = RewardComputer(
    reward_weights=self.config.get('reward_weights'),
    use_llm_judge=True,  # ← 确保为True
    llm_config={...}
)
```

### 问题4: WandB离线日志未生成

**解决方案**:
```bash
# 1. 检查配置
grep -A 5 "wandb:" config/training.yaml

# 2. 设置环境变量
export WANDB_MODE=offline

# 3. 重新运行
python3 train.py

# 4. 检查目录
ls -la wandb/
```

---

## 📈 预期效果对比

### 实施前 (当前状态)

```python
# WandB日志
wandb.log({
    "train/loss": 0.5,
    "train/accuracy": 65.0,  # 总体准确率
    "train/accuracy_math": 70.0,  # 问题类型维度
    "train/accuracy_code": 55.0,
    "train/accuracy_qa": 68.0,
})
```

**局限性**:
- ❌ 无法区分GSM8K和MATH的性能
- ❌ 无法监控LLM Judge质量
- ❌ 无法追踪成本
- ❌ 验证集缺乏详细分析

### 实施后 (增强状态)

```python
# WandB日志
wandb.log({
    # 原有指标
    "train/loss": 0.5,
    "train/accuracy": 65.0,
    "train/accuracy_math": 70.0,
    "train/accuracy_code": 55.0,
    "train/accuracy_qa": 68.0,

    # 🆕 数据集维度
    "dataset/gsm8k/accuracy": 85.2,
    "dataset/gsm8k/count": 20,
    "dataset/math/accuracy": 42.1,
    "dataset/math/count": 15,
    "dataset/hotpotqa/accuracy": 68.3,
    "dataset/hotpotqa/count": 18,
    "dataset/humaneval/accuracy": 55.7,
    "dataset/humaneval/count": 12,

    # 🆕 LLM Judge监控
    "judge/success_rate": 0.85,
    "judge/parse_failure_rate": 0.10,
    "judge/api_failure_rate": 0.05,
    "judge/correct_ratio": 0.706,

    # 🆕 成本统计
    "cost/total_cost": 12.34,
    "cost/avg_cost_per_sample": 0.0123,
    "cost/executor_calls": 1000,
    "cost/judge_calls": 800,
})
```

**优势**:
- ✅ 完整的数据集级性能可见性
- ✅ LLM Judge质量监控
- ✅ 成本追踪和优化依据
- ✅ 验证集详细诊断能力

---

## 🎓 使用示例

### 场景1: 分析GSM8K性能下降

```python
import wandb

api = wandb.Api()
run = api.run("entity/project/run_id")

# 查询GSM8K准确率历史
history = run.history(keys=['dataset/gsm8k/accuracy', 'train/step'])
print(history)

# 检查是否在某个step后下降
import pandas as pd
df = pd.DataFrame(history)
print(df[df['train/step'] > 100])  # 查看100步后的表现
```

### 场景2: 对比不同数据集的表现

```python
import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
run = api.run("entity/project/run_id")

# 获取所有数据集的最新准确率
datasets = ['gsm8k', 'math', 'hotpotqa', 'humaneval', 'commonsenseqa', 'mmlu']
accuracies = []

for dataset in datasets:
    key = f'dataset/{dataset}/accuracy'
    acc = run.summary.get(key, 0)
    accuracies.append(acc)
    print(f"{dataset:15s}: {acc:5.1f}%")

# 绘制条形图
plt.bar(datasets, accuracies)
plt.ylabel('Accuracy (%)')
plt.title('Performance by Dataset')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('dataset_comparison.png')
print("✅ 已保存图表: dataset_comparison.png")
```

### 场景3: 监控LLM Judge健康度

```python
import wandb

api = wandb.Api()
run = api.run("entity/project/run_id")

# 查询Judge统计
judge_stats = {
    'success_rate': run.summary.get('judge/success_rate', 0),
    'parse_failure_rate': run.summary.get('judge/parse_failure_rate', 0),
    'api_failure_rate': run.summary.get('judge/api_failure_rate', 0),
    'correct_ratio': run.summary.get('judge/correct_ratio', 0),
}

print("🤖 LLM Judge健康度:")
for key, value in judge_stats.items():
    print(f"  {key:20s}: {value:.3f}")

# 警报检查
if judge_stats['success_rate'] < 0.8:
    print("\n⚠️  警告: Judge成功率低于80%！")
if judge_stats['api_failure_rate'] > 0.1:
    print("⚠️  警告: API失败率高于10%！")
```

### 场景4: 成本优化分析

```python
import wandb

api = wandb.Api()
run = api.run("entity/project/run_id")

# 查询成本历史
history = run.history(keys=[
    'cost/total_cost',
    'cost/avg_cost_per_sample',
    'train/accuracy',
    'train/step'
])

import pandas as pd
df = pd.DataFrame(history)

# 分析成本效率
df['cost_per_accuracy'] = df['cost/avg_cost_per_sample'] / (df['train/accuracy'] / 100)
print("\n💰 成本效率分析:")
print(df[['train/step', 'train/accuracy', 'cost/avg_cost_per_sample', 'cost_per_accuracy']].tail(10))

# 找出最高效的step
best_step = df.loc[df['cost_per_accuracy'].idxmin()]
print(f"\n✨ 最高效的训练步: Step {best_step['train/step']:.0f}")
print(f"   准确率: {best_step['train/accuracy']:.1f}%")
print(f"   成本/样本: ${best_step['cost/avg_cost_per_sample']:.6f}")
```

---

## 📚 进一步阅读

- **详细设计文档**: `docs/WANDB_MONITORING_DESIGN.md`
  - 系统架构分析
  - 指标设计详解
  - WandB仪表板配置
  - 性能开销分析

- **实现代码**: `src/wandb_metrics_collectors.py`
  - `DatasetMetricsCollector` - 数据集指标收集
  - `JudgeMetricsCollector` - LLM Judge监控
  - `CostTracker` - 成本追踪

- **补丁指南**: `docs/WANDB_PATCH_GUIDE.md`
  - 10个补丁的详细位置和代码
  - 手动应用步骤
  - 验证清单

---

## 🤝 贡献

如果你发现问题或有改进建议，请:

1. 创建Issue描述问题
2. 提交Pull Request修复
3. 更新相关文档

---

## 📄 许可证

本项目遵循MIT许可证。

---

**最后更新**: 2025-11-23
**维护者**: AI Training Team
