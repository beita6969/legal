
# ============================================================================
# 手动应用补丁指南
# ============================================================================

本指南包含需要手动添加到grpo_trainer.py的所有代码片段。

## 准备工作

1. 备份原文件:
   ```bash
   cp src/grpo_trainer.py src/grpo_trainer.py.backup
   ```

2. 确保wandb_metrics_collectors.py已就位:
   ```bash
   ls src/wandb_metrics_collectors.py
   ```

---

## Patch 1: 添加导入 (第27行后)

在现有导入后添加:

```python

# WandB监控增强 - 导入指标收集器
try:
    from wandb_metrics_collectors import (
        DatasetMetricsCollector,
        JudgeMetricsCollector,
        CostTracker
    )
except ImportError:
    from src.wandb_metrics_collectors import (
        DatasetMetricsCollector,
        JudgeMetricsCollector,
        CostTracker
    )

```

---

## Patch 2: 初始化成本追踪器 (第214行后，optimizer初始化后)

```python

        # 🆕 初始化成本追踪器
        print("\n💰 初始化成本追踪器...")
        self.cost_tracker = CostTracker()
        print("  ✅ 成本追踪器初始化完成")

```

---

## Patch 3: train_step()开始处初始化收集器 (第294行后)

在 `batch_stats = self.data_manager.get_batch_stats(batch)` 后添加:

```python

        # 🆕 初始化数据集指标收集器
        dataset_collector = DatasetMetricsCollector()

```

---

## Patch 4: 在样本循环中获取source (第316行)

在 `problem_type = sample['problem_type']` 后添加:

```python

            source = sample.get('source', 'unknown')  # 🆕 获取数据集来源

```

---

## Patch 5: 在样本循环中记录数据 (第393行后)

在 `group_correctness.append(correctness)` 后添加:

```python

                        # 🆕 记录到数据集收集器
                        dataset_collector.add_result(
                            source=source,
                            correctness=correctness,
                            reward=reward,
                            cost=cost if 'cost' in locals() else 0.0
                        )

                        # 🆕 记录成本
                        self.cost_tracker.add_cost(
                            cost=cost if 'cost' in locals() else 0.0,
                            is_executor=True
                        )

```

---

## Patch 6: train_step()末尾添加wandb日志 (第513行前)

在 `wandb.log(wandb_log_data, step=step)` 前添加:

```python

        # 🆕 添加数据集维度指标
        dataset_logs = dataset_collector.get_wandb_logs(step=step, prefix="dataset")
        wandb_log_data.update(dataset_logs)

        # 🆕 添加LLM Judge监控
        if self.reward_computer.use_llm_judge:
            judge_collector = JudgeMetricsCollector()
            judge_collector.update_from_reward_computer(self.reward_computer)
            judge_logs = judge_collector.get_wandb_logs()
            wandb_log_data.update(judge_logs)

        # 🆕 添加成本统计
        cost_logs = self.cost_tracker.get_wandb_logs()
        wandb_log_data.update(cost_logs)

        # 🆕 打印数据集统计摘要 (每10步)
        if step % 10 == 0:
            dataset_collector.print_summary()
            if self.reward_computer.use_llm_judge:
                judge_collector.print_summary()

```

---

## Patch 7: evaluate_on_val_set()开始处初始化 (第674行后)

在 `batch_stats = self.data_manager.get_batch_stats(val_batch)` 后添加:

```python

        # 🆕 初始化验证集数据集指标收集器
        val_dataset_collector = DatasetMetricsCollector()

```

---

## Patch 8: 验证集循环中获取source (第682行)

在 `problem_type = sample['problem_type']` 后添加:

```python

            source = sample.get('source', 'unknown')  # 🆕 获取数据集来源

```

---

## Patch 9: 验证集循环中记录数据 (第732行后)

在 `if idx <= 5:` 代码块前添加:

```python

                    # 🆕 记录到验证集收集器
                    val_dataset_collector.add_result(
                        source=source,
                        correctness=correctness,
                        reward=correctness,
                        cost=cost
                    )

```

---

## Patch 10: evaluate_on_val_set()末尾添加日志 (第800行)

在 `wandb.log(val_metrics, step=step)` 后添加:

```python

        # 🆕 添加验证集数据集维度指标
        val_dataset_logs = val_dataset_collector.get_wandb_logs(step=step, prefix="val")
        wandb.log(val_dataset_logs, step=step)

        # 🆕 打印验证集数据集统计摘要
        val_dataset_collector.print_summary()

```

---

## 验证

完成后运行:

```bash
python3 -m py_compile src/grpo_trainer.py
```

如果没有语法错误，继续测试:

```bash
python3 -c "from src.grpo_trainer import GRPOTrainer; print('✅ 导入成功')"
```

---

## 回滚

如果出现问题，恢复备份:

```bash
mv src/grpo_trainer.py.backup src/grpo_trainer.py
```
