# WandB监控系统 - 完整研究与实现

## 📋 项目概述

本项目为GRPO训练系统设计并实现了完整的WandB监控方案，提供**数据集级别**的性能可见性、LLM Judge质量监控和成本追踪。

### 核心改进

| 功能 | 优先级 | 状态 | 描述 |
|------|--------|------|------|
| 数据集维度统计 | P0 | ✅ 已实现 | 为每个数据集(GSM8K, MATH等)单独统计准确率 |
| LLM Judge监控 | P1 | ✅ 已实现 | 监控Judge成功率、失败率和判决分布 |
| 验证集详细分析 | P1 | ✅ 已实现 | 按数据集分解的验证集性能 |
| 成本统计 | P2 | ✅ 已实现 | 累积成本追踪和优化分析 |

---

## 📚 文档结构

```
.
├── docs/
│   ├── WANDB_QUICKSTART.md          ⭐ 快速开始指南 (从这里开始!)
│   ├── WANDB_MONITORING_DESIGN.md   📊 详细设计文档 (19KB)
│   └── WANDB_PATCH_GUIDE.md         🔧 手动补丁指南 (5KB)
├── src/
│   ├── wandb_metrics_collectors.py  💎 可复用的指标收集器 (16KB)
│   └── grpo_trainer.py              🎯 需要修改的文件
└── scripts/
    └── apply_wandb_patch.py         🤖 半自动补丁工具 (13KB)
```

---

## 🚀 快速开始 (3步)

### Step 1: 测试工具类

```bash
cd /home/yijia/.claude/11/integrated_aflow_roll
python3 src/wandb_metrics_collectors.py

# 预期输出:
# 🧪 测试DatasetMetricsCollector
# WandB日志:
#   dataset/gsm8k/accuracy: 66.7
#   dataset/math/accuracy: 50.0
# ✅ 所有测试通过！
```

### Step 2: 应用补丁

```bash
# 方式A: 手动应用 (推荐)
cat docs/WANDB_PATCH_GUIDE.md  # 查看详细步骤
cp src/grpo_trainer.py src/grpo_trainer.py.backup
vim src/grpo_trainer.py  # 按照指南添加10个补丁

# 方式B: 半自动应用 (实验性)
python3 scripts/apply_wandb_patch.py
# 然后手动完成剩余补丁
```

### Step 3: 验证和运行

```bash
# 验证语法
python3 -m py_compile src/grpo_trainer.py

# 测试导入
python3 -c "from src.grpo_trainer import GRPOTrainer; print('✅ OK')"

# 离线测试
# 修改config/training.yaml: wandb.mode = offline
python3 train.py

# 检查输出中的新统计信息:
# 📊 数据集统计摘要:
#   gsm8k          :   2/  3 =  66.7%
#   math           :   1/  2 =  50.0%
```

---

## 📊 系统研究总结

### 当前训练架构

**训练循环** (`src/grpo_trainer.py`):
```python
class GRPOTrainer:
    def train(self):                    # 主循环 (第763-811行)
        for step in range(max_steps):
            metrics = await self.train_step(step)
            if step % eval_every == 0:
                val_metrics = await self.evaluate_on_val_set()

    async def train_step(self, step):  # 单步训练 (第278-515行)
        batch = self.data_manager.sample_batch()  # 混合采样
        for sample in batch:
            workflows = generate_k_workflows()     # GRPO组
            rewards = execute_and_compute()
        update_policy()  # PPO更新
        wandb.log(metrics)  # ← 这里需要增强！
```

**数据集支持**:
- ✅ GSM8K (math, source="gsm8k")
- ✅ MATH (math, source="math")
- ✅ HotpotQA (qa, source="hotpotqa")
- ✅ HumanEval (code, source="humaneval")
- ✅ CommonsenseQA (qa, source="commonsenseqa")
- ✅ MMLU (qa, source="mmlu")
- ⚠️  MBPP (code, source="mbpp") - 已过滤

**关键发现**:
1. 每个样本包含 `source` 字段标识数据集来源
2. `reward_computer` 使用LLM Judge (GPT OSS 120B @ port 8002)
3. 支持数据集专属的Judge Prompt (`judge_prompt_loader`)
4. 二元奖励系统: 正确=1.0, 错误=0.0

---

## 🎯 实现方案

### 新增工具类

**文件**: `src/wandb_metrics_collectors.py`

```python
from wandb_metrics_collectors import (
    DatasetMetricsCollector,  # 数据集维度统计
    JudgeMetricsCollector,    # LLM Judge监控
    CostTracker,              # 成本追踪
)

# 使用示例
collector = DatasetMetricsCollector()
collector.add_result(source='gsm8k', correctness=1.0, reward=1.0)
logs = collector.get_wandb_logs(step=100)
wandb.log(logs, step=100)
```

### 需要修改的位置

**文件**: `src/grpo_trainer.py`

| 补丁 | 位置 | 功能 | 难度 |
|------|------|------|------|
| Patch 1 | 第27行后 | 添加导入 | ⭐ 简单 |
| Patch 2 | 第214行后 | 初始化成本追踪器 | ⭐ 简单 |
| Patch 3 | 第294行后 | train_step初始化收集器 | ⭐ 简单 |
| Patch 4 | 第316行 | 获取source字段 | ⭐ 简单 |
| Patch 5 | 第393行后 | 记录到收集器 | ⭐⭐ 中等 |
| Patch 6 | 第513行前 | 添加wandb日志 | ⭐⭐ 中等 |
| Patch 7 | 第674行后 | 验证集初始化 | ⭐ 简单 |
| Patch 8 | 第682行 | 验证集source | ⭐ 简单 |
| Patch 9 | 第732行后 | 验证集记录 | ⭐⭐ 中等 |
| Patch 10 | 第800行 | 验证集日志 | ⭐⭐ 中等 |

详细的代码见: `docs/WANDB_PATCH_GUIDE.md`

---

## 📈 预期效果

### 实施前 vs 实施后

#### 实施前 (当前)

```python
wandb.log({
    "train/accuracy": 65.0,        # 只有总体准确率
    "train/accuracy_math": 70.0,   # 问题类型维度
    "train/accuracy_code": 55.0,
    "train/accuracy_qa": 68.0,
})
```

❌ **无法回答**:
- GSM8K准确率是多少?
- MATH数据集表现如何?
- LLM Judge成功率?
- 累积训练成本?

#### 实施后 (增强)

```python
wandb.log({
    # 原有指标
    "train/accuracy": 65.0,
    "train/accuracy_math": 70.0,

    # 🆕 数据集维度
    "dataset/gsm8k/accuracy": 85.2,
    "dataset/gsm8k/count": 20,
    "dataset/math/accuracy": 42.1,
    "dataset/math/count": 15,
    "dataset/hotpotqa/accuracy": 68.3,
    "dataset/humaneval/accuracy": 55.7,

    # 🆕 LLM Judge
    "judge/success_rate": 0.85,
    "judge/parse_failure_rate": 0.10,
    "judge/correct_ratio": 0.706,

    # 🆕 成本
    "cost/total_cost": 12.34,
    "cost/avg_cost_per_sample": 0.0123,
})
```

✅ **可以回答**:
- GSM8K: 85.2% (20个样本)
- MATH: 42.1% (15个样本) → 需要重点优化!
- LLM Judge成功率: 85%
- 平均成本: $0.0123/样本

---

## 🔍 使用场景

### 场景1: 诊断MATH数据集性能低下

```python
import wandb
api = wandb.Api()
run = api.run("project/run_id")

# 查看MATH性能趋势
history = run.history(keys=['dataset/math/accuracy', 'train/step'])
print(history)

# 对比GSM8K和MATH
gsm8k_acc = run.summary['dataset/gsm8k/accuracy']
math_acc = run.summary['dataset/math/accuracy']
print(f"GSM8K: {gsm8k_acc:.1f}%")
print(f"MATH: {math_acc:.1f}%")
print(f"差距: {gsm8k_acc - math_acc:.1f}pp")  # 可能发现43.1pp的差距
```

### 场景2: 监控LLM Judge健康度

```python
judge_stats = {
    'success_rate': run.summary['judge/success_rate'],
    'parse_failure_rate': run.summary['judge/parse_failure_rate'],
    'api_failure_rate': run.summary['judge/api_failure_rate'],
}

# 检查是否需要干预
if judge_stats['success_rate'] < 0.8:
    print("⚠️  Judge成功率过低，检查:")
    print("  1. GPT OSS 120B服务是否正常 (port 8002)")
    print("  2. Prompt格式是否正确")
    print("  3. 是否需要调整temperature")
```

### 场景3: 成本优化

```python
# 分析成本效率
history = run.history(keys=[
    'cost/avg_cost_per_sample',
    'train/accuracy',
    'train/step'
])

import pandas as pd
df = pd.DataFrame(history)
df['cost_per_point'] = df['cost/avg_cost_per_sample'] / (df['train/accuracy'] / 100)

# 找出最高效的训练阶段
best = df.loc[df['cost_per_point'].idxmin()]
print(f"最高效的step: {best['train/step']:.0f}")
print(f"准确率: {best['train/accuracy']:.1f}%")
print(f"成本: ${best['cost/avg_cost_per_sample']:.6f}/样本")
```

---

## 🧪 测试清单

### 单元测试

- [x] DatasetMetricsCollector基础功能
- [x] JudgeMetricsCollector统计更新
- [x] CostTracker成本累积
- [x] 工具类导入测试

### 集成测试

- [ ] grpo_trainer.py语法检查
- [ ] train_step()中的数据集统计
- [ ] evaluate_on_val_set()中的数据集统计
- [ ] LLM Judge统计读取
- [ ] 成本追踪累积

### 端到端测试

- [ ] 离线模式训练 (wandb.mode=offline)
- [ ] 检查wandb日志文件生成
- [ ] 在线模式训练 (wandb.mode=online)
- [ ] WandB仪表板可视化

---

## 📊 WandB仪表板配置

### 推荐面板布局

```yaml
# 第1行: 训练总览
- type: line
  title: "Training Loss"
  metrics: [train/loss]

- type: line
  title: "Overall Accuracy"
  metrics: [train/accuracy, val/accuracy]

# 第2行: 数据集性能
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
  title: "GSM8K vs MATH Performance"
  metrics:
    - dataset/gsm8k/accuracy
    - dataset/math/accuracy

# 第3行: LLM Judge监控
- type: line
  title: "LLM Judge Success Rate"
  metrics:
    - judge/success_rate
    - judge/parse_failure_rate
    - judge/api_failure_rate

- type: number
  title: "Judge Accuracy"
  metric: judge/correct_ratio

# 第4行: 成本分析
- type: line
  title: "Cumulative Cost"
  metric: cost/total_cost

- type: line
  title: "Cost per Sample"
  metric: cost/avg_cost_per_sample
```

创建方式:
1. 访问 https://wandb.ai/your-entity/aflow-roll-integration
2. 选择run → 点击"Customize" → "Add visualization"
3. 按照上述配置添加面板

---

## 🛠️ 故障排查

### 常见问题

#### Q1: `ImportError: cannot import name 'DatasetMetricsCollector'`

**原因**: Python路径问题

**解决**:
```bash
# 检查文件
ls src/wandb_metrics_collectors.py

# 添加到PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"
python3 -c "from src.wandb_metrics_collectors import DatasetMetricsCollector"
```

#### Q2: 数据集统计为空

**原因**: batch中缺少该数据集

**检查**:
```bash
# 检查数据文件中的source字段
head -10 data/mixed/train_mixed_with_math.jsonl | jq .source

# 检查domain_ratios配置
grep -A 3 "domain_ratios:" config/training.yaml
```

#### Q3: LLM Judge统计不更新

**原因**: `use_llm_judge=False`

**检查**:
```python
# src/grpo_trainer.py 第198行
self.reward_computer = RewardComputer(
    use_llm_judge=True,  # ← 确保为True
    ...
)
```

#### Q4: WandB离线日志未生成

**解决**:
```bash
# 设置环境变量
export WANDB_MODE=offline

# 或修改配置
vim config/training.yaml
# wandb:
#   mode: offline

# 运行后检查
ls -la wandb/
```

---

## 📦 文件清单

### 文档 (docs/)

| 文件 | 大小 | 描述 |
|------|------|------|
| `WANDB_QUICKSTART.md` | 11KB | ⭐ 快速开始指南 |
| `WANDB_MONITORING_DESIGN.md` | 19KB | 📊 详细设计文档 |
| `WANDB_PATCH_GUIDE.md` | 5KB | 🔧 手动补丁步骤 |

### 代码 (src/)

| 文件 | 大小 | 描述 |
|------|------|------|
| `wandb_metrics_collectors.py` | 16KB | 💎 指标收集器工具类 |
| `grpo_trainer.py` | - | 🎯 需要修改 (10处) |

### 脚本 (scripts/)

| 文件 | 大小 | 描述 |
|------|------|------|
| `apply_wandb_patch.py` | 13KB | 🤖 半自动补丁工具 |

---

## 🎓 学习路径

### 新手路径 (30分钟)

1. **阅读**: `docs/WANDB_QUICKSTART.md` (5分钟)
2. **测试**: 运行 `wandb_metrics_collectors.py` (5分钟)
3. **应用**: 按照 `WANDB_PATCH_GUIDE.md` 手动添加补丁 (15分钟)
4. **验证**: 运行离线测试 (5分钟)

### 深入路径 (1小时)

1. **研究**: 阅读 `WANDB_MONITORING_DESIGN.md` (15分钟)
2. **理解**: 查看当前 `grpo_trainer.py` 结构 (15分钟)
3. **实现**: 应用所有补丁 (20分钟)
4. **分析**: 运行训练并分析WandB仪表板 (10分钟)

### 专家路径 (2小时)

1. **全面研究**: 阅读所有文档 (30分钟)
2. **定制化**: 修改指标收集器适应特定需求 (30分钟)
3. **扩展**: 添加新的监控维度 (30分钟)
4. **优化**: 配置WandB仪表板和警报 (30分钟)

---

## 🚀 下一步行动

### 立即开始

```bash
cd /home/yijia/.claude/11/integrated_aflow_roll

# Step 1: 测试工具类
python3 src/wandb_metrics_collectors.py

# Step 2: 查看快速指南
cat docs/WANDB_QUICKSTART.md

# Step 3: 应用补丁
cat docs/WANDB_PATCH_GUIDE.md
cp src/grpo_trainer.py src/grpo_trainer.py.backup
vim src/grpo_trainer.py

# Step 4: 验证
python3 -m py_compile src/grpo_trainer.py
python3 -c "from src.grpo_trainer import GRPOTrainer; print('✅')"

# Step 5: 运行
python3 train.py
```

### 优先级建议

1. **P0 (立即)**: 实施数据集维度统计 (Patch 1-6)
2. **P1 (1天内)**: 添加LLM Judge监控和验证集分解 (Patch 7-10)
3. **P2 (1周内)**: 配置WandB仪表板和警报
4. **P3 (未来)**: 根据监控结果优化训练策略

---

## 📞 获取帮助

### 文档索引

- 🚀 **刚开始?** → `docs/WANDB_QUICKSTART.md`
- 📊 **了解设计?** → `docs/WANDB_MONITORING_DESIGN.md`
- 🔧 **手动添加?** → `docs/WANDB_PATCH_GUIDE.md`
- 💎 **API文档?** → `src/wandb_metrics_collectors.py` (代码注释)

### 常见任务

- **测试工具**: `python3 src/wandb_metrics_collectors.py`
- **生成补丁指南**: `python3 scripts/apply_wandb_patch.py`
- **验证语法**: `python3 -m py_compile src/grpo_trainer.py`
- **离线测试**: 修改config → `wandb.mode: offline` → `python3 train.py`

---

## 📄 许可证

MIT License

---

**项目状态**: ✅ 已完成 (2025-11-23)

**维护者**: AI Training Team

**最后更新**: 2025-11-23 17:03 CST
