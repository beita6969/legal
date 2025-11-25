#!/usr/bin/env python3
"""
GRPO Trainer WandB增强补丁

这个文件包含了需要添加到grpo_trainer.py中的具体代码片段。

使用方法:
1. 备份原文件: cp src/grpo_trainer.py src/grpo_trainer.py.backup
2. 手动应用以下补丁，或使用提供的函数自动应用
"""

import re
from pathlib import Path


# ============================================================================
# 补丁内容定义
# ============================================================================

PATCH_1_IMPORT = '''
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
'''

PATCH_2_INIT_COST_TRACKER = '''
        # 🆕 初始化成本追踪器
        print("\\n💰 初始化成本追踪器...")
        self.cost_tracker = CostTracker()
        print("  ✅ 成本追踪器初始化完成")
'''

PATCH_3_TRAIN_STEP_INIT = '''
        # 🆕 初始化数据集指标收集器
        dataset_collector = DatasetMetricsCollector()
'''

PATCH_4_TRAIN_STEP_SOURCE = '''
            source = sample.get('source', 'unknown')  # 🆕 获取数据集来源
'''

PATCH_5_TRAIN_STEP_COLLECT = '''
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
'''

PATCH_6_TRAIN_STEP_WANDB = '''
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
'''

PATCH_7_VAL_INIT = '''
        # 🆕 初始化验证集数据集指标收集器
        val_dataset_collector = DatasetMetricsCollector()
'''

PATCH_8_VAL_SOURCE = '''
            source = sample.get('source', 'unknown')  # 🆕 获取数据集来源
'''

PATCH_9_VAL_COLLECT = '''
                    # 🆕 记录到验证集收集器
                    val_dataset_collector.add_result(
                        source=source,
                        correctness=correctness,
                        reward=correctness,
                        cost=cost
                    )
'''

PATCH_10_VAL_WANDB = '''
        # 🆕 添加验证集数据集维度指标
        val_dataset_logs = val_dataset_collector.get_wandb_logs(step=step, prefix="val")
        wandb.log(val_dataset_logs, step=step)

        # 🆕 打印验证集数据集统计摘要
        val_dataset_collector.print_summary()
'''


# ============================================================================
# 自动应用补丁函数
# ============================================================================

def apply_patch_to_file(filepath: str, dry_run: bool = True):
    """
    自动应用补丁到grpo_trainer.py

    Args:
        filepath: grpo_trainer.py的路径
        dry_run: 如果为True，只打印修改内容而不实际修改文件

    Returns:
        bool: 是否成功应用补丁
    """
    file_path = Path(filepath)

    if not file_path.exists():
        print(f"❌ 文件不存在: {filepath}")
        return False

    # 读取原文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 备份
    backup_path = file_path.with_suffix('.py.backup')
    if not dry_run:
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ 已备份原文件到: {backup_path}")

    original_content = content

    # 应用补丁
    patches_applied = 0

    # Patch 1: 添加导入
    if 'from wandb_metrics_collectors import' not in content:
        # 在导入部分末尾添加 (在第27行附近)
        import_end_pattern = r'from operator_prompt_enhancer import OperatorPromptEnhancer'
        if re.search(import_end_pattern, content):
            content = re.sub(
                import_end_pattern,
                import_end_pattern + '\n' + PATCH_1_IMPORT,
                content,
                count=1
            )
            patches_applied += 1
            print("✅ Patch 1: 添加导入语句")

    # Patch 2: 在__init__中添加成本追踪器
    if 'self.cost_tracker' not in content:
        # 在optimizer初始化后添加 (第214行附近)
        init_pattern = r'(self\.optimizer = torch\.optim\.AdamW\([^)]+\))'
        match = re.search(init_pattern, content, re.DOTALL)
        if match:
            insert_pos = match.end()
            content = content[:insert_pos] + '\n\n' + PATCH_2_INIT_COST_TRACKER + content[insert_pos:]
            patches_applied += 1
            print("✅ Patch 2: 在__init__中添加成本追踪器")

    # Patch 3: 在train_step开始添加数据集收集器
    if 'dataset_collector = DatasetMetricsCollector' not in content:
        # 在train_step的batch采样后添加 (第294行附近)
        pattern = r'(batch_stats = self\.data_manager\.get_batch_stats\(batch\)\n\s+print\(f"\\n📦 Batch.*?)\n'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            insert_pos = match.end()
            content = content[:insert_pos] + '\n' + PATCH_3_TRAIN_STEP_INIT + '\n' + content[insert_pos:]
            patches_applied += 1
            print("✅ Patch 3: 在train_step开始添加数据集收集器")

    # Patch 4 & 5: 在样本循环中添加source和收集逻辑
    # 这个补丁比较复杂，建议手动添加
    if 'source = sample.get' not in content:
        print("⚠️  Patch 4-5: 需要手动添加 - 在样本循环中添加source获取和数据收集")
        print("    位置: train_step()方法，第316行 (problem_type = sample['problem_type']后)")
        print("    内容: source = sample.get('source', 'unknown')")
        print("    位置: 第393行后 (group_correctness.append后)")
        print("    内容: dataset_collector.add_result(...) 和 self.cost_tracker.add_cost(...)")

    # Patch 6: 在train_step末尾添加wandb日志
    if 'dataset_logs = dataset_collector.get_wandb_logs' not in content:
        # 在wandb.log之前添加 (第513行附近)
        pattern = r'(wandb\.log\(wandb_log_data, step=step\))'
        match = re.search(pattern, content)
        if match:
            insert_pos = match.start()
            content = content[:insert_pos] + PATCH_6_TRAIN_STEP_WANDB + '\n\n        ' + content[insert_pos:]
            patches_applied += 1
            print("✅ Patch 6: 在train_step末尾添加wandb日志")

    # Patch 7-10: 验证集相关补丁
    if 'val_dataset_collector = DatasetMetricsCollector' not in content:
        print("⚠️  Patch 7-10: 需要手动添加 - 验证集数据集监控")
        print("    位置: evaluate_on_val_set()方法")
        print("    详见设计文档中的Step 3")

    # 检查是否有实际修改
    if content != original_content:
        if dry_run:
            print(f"\n📋 Dry run模式: 共应用 {patches_applied} 个补丁")
            print("\n如需实际应用补丁，请运行:")
            print(f"  apply_patch_to_file('{filepath}', dry_run=False)")
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"\n✅ 成功应用 {patches_applied} 个补丁到: {filepath}")
            print(f"   备份文件: {backup_path}")
        return True
    else:
        print("✅ 文件已包含所有补丁，无需修改")
        return True


def generate_manual_patch_guide():
    """
    生成手动应用补丁的详细指南
    """
    guide = """
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
{}
```

---

## Patch 2: 初始化成本追踪器 (第214行后，optimizer初始化后)

```python
{}
```

---

## Patch 3: train_step()开始处初始化收集器 (第294行后)

在 `batch_stats = self.data_manager.get_batch_stats(batch)` 后添加:

```python
{}
```

---

## Patch 4: 在样本循环中获取source (第316行)

在 `problem_type = sample['problem_type']` 后添加:

```python
{}
```

---

## Patch 5: 在样本循环中记录数据 (第393行后)

在 `group_correctness.append(correctness)` 后添加:

```python
{}
```

---

## Patch 6: train_step()末尾添加wandb日志 (第513行前)

在 `wandb.log(wandb_log_data, step=step)` 前添加:

```python
{}
```

---

## Patch 7: evaluate_on_val_set()开始处初始化 (第674行后)

在 `batch_stats = self.data_manager.get_batch_stats(val_batch)` 后添加:

```python
{}
```

---

## Patch 8: 验证集循环中获取source (第682行)

在 `problem_type = sample['problem_type']` 后添加:

```python
{}
```

---

## Patch 9: 验证集循环中记录数据 (第732行后)

在 `if idx <= 5:` 代码块前添加:

```python
{}
```

---

## Patch 10: evaluate_on_val_set()末尾添加日志 (第800行)

在 `wandb.log(val_metrics, step=step)` 后添加:

```python
{}
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
""".format(
        PATCH_1_IMPORT,
        PATCH_2_INIT_COST_TRACKER,
        PATCH_3_TRAIN_STEP_INIT,
        PATCH_4_TRAIN_STEP_SOURCE,
        PATCH_5_TRAIN_STEP_COLLECT,
        PATCH_6_TRAIN_STEP_WANDB,
        PATCH_7_VAL_INIT,
        PATCH_8_VAL_SOURCE,
        PATCH_9_VAL_COLLECT,
        PATCH_10_VAL_WANDB
    )

    return guide


if __name__ == "__main__":
    print("="*80)
    print("WandB监控增强补丁工具")
    print("="*80)

    # 生成手动补丁指南
    guide = generate_manual_patch_guide()
    guide_path = Path("docs/WANDB_PATCH_GUIDE.md")
    guide_path.parent.mkdir(parents=True, exist_ok=True)
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide)
    print(f"\n✅ 已生成手动补丁指南: {guide_path}")

    # 尝试自动应用补丁 (dry run)
    print("\n" + "="*80)
    print("尝试自动应用补丁 (Dry Run模式)")
    print("="*80 + "\n")

    trainer_path = "src/grpo_trainer.py"
    if Path(trainer_path).exists():
        apply_patch_to_file(trainer_path, dry_run=True)
    else:
        print(f"⚠️  未找到 {trainer_path}，请手动应用补丁")
        print(f"   参考指南: {guide_path}")

    print("\n" + "="*80)
    print("下一步")
    print("="*80)
    print("\n1. 查看手动补丁指南:")
    print(f"   cat {guide_path}")
    print("\n2. 手动编辑grpo_trainer.py，应用所有补丁")
    print("\n3. 验证语法:")
    print("   python3 -m py_compile src/grpo_trainer.py")
    print("\n4. 测试导入:")
    print("   python3 -c 'from src.grpo_trainer import GRPOTrainer'")
    print("\n5. 运行训练测试")
