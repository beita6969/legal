#!/usr/bin/env python3
"""
WandB监控系统实现示例

这个文件包含了在grpo_trainer.py中需要添加的具体代码片段
"""

from collections import defaultdict
import numpy as np
import wandb


class DatasetMetricsCollector:
    """
    数据集维度的指标收集器

    使用方法:
        collector = DatasetMetricsCollector()
        collector.add_result(source='gsm8k', correctness=1.0, reward=1.0)
        metrics = collector.get_wandb_logs(step=100)
        wandb.log(metrics, step=100)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计数据"""
        self.dataset_stats = defaultdict(lambda: {
            'correctness': [],
            'rewards': [],
            'costs': [],
        })

    def add_result(self, source: str, correctness: float, reward: float, cost: float = 0.0):
        """
        添加单个样本结果

        Args:
            source: 数据集来源 (如'gsm8k', 'math', 'hotpotqa')
            correctness: 正确性分数 (0.0 或 1.0)
            reward: 奖励值
            cost: 执行成本
        """
        stats = self.dataset_stats[source]
        stats['correctness'].append(correctness)
        stats['rewards'].append(reward)
        stats['costs'].append(cost)

    def get_wandb_logs(self, step: int, prefix: str = "dataset") -> dict:
        """
        生成WandB日志字典

        Args:
            step: 当前训练步数
            prefix: 日志前缀 (如'dataset'或'val')

        Returns:
            适合wandb.log()的字典
        """
        logs = {}

        for source, stats in self.dataset_stats.items():
            if not stats['correctness']:
                continue

            # 计算准确率 (使用0.9阈值适应二元奖励)
            num_correct = sum(1 for c in stats['correctness'] if c >= 0.9)
            num_total = len(stats['correctness'])
            accuracy = (num_correct / num_total * 100) if num_total > 0 else 0.0

            # 计算平均奖励
            avg_reward = np.mean(stats['rewards']) if stats['rewards'] else 0.0

            # 计算平均成本
            avg_cost = np.mean(stats['costs']) if stats['costs'] else 0.0

            # 添加到日志
            logs[f"{prefix}/{source}/accuracy"] = accuracy
            logs[f"{prefix}/{source}/count"] = num_total
            logs[f"{prefix}/{source}/avg_reward"] = avg_reward

            if avg_cost > 0:
                logs[f"{prefix}/{source}/avg_cost"] = avg_cost

        return logs

    def print_summary(self):
        """打印统计摘要 (用于调试)"""
        print("\n📊 数据集统计摘要:")
        for source, stats in sorted(self.dataset_stats.items()):
            if not stats['correctness']:
                continue

            num_correct = sum(1 for c in stats['correctness'] if c >= 0.9)
            num_total = len(stats['correctness'])
            accuracy = (num_correct / num_total * 100) if num_total > 0 else 0.0

            print(f"  {source:15s}: {num_correct:3d}/{num_total:3d} = {accuracy:5.1f}%")


class JudgeMetricsCollector:
    """
    LLM Judge性能监控器

    使用方法:
        collector = JudgeMetricsCollector()
        collector.update_from_reward_computer(reward_computer)
        metrics = collector.get_wandb_logs()
        wandb.log(metrics, step=step)
    """

    def __init__(self):
        self.judge_stats = {
            'total_evaluations': 0,
            'llm_judge_success': 0,
            'llm_judge_parse_failures': 0,
            'llm_judge_api_failures': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
        }

    def update_from_reward_computer(self, reward_computer):
        """
        从RewardComputer读取统计数据

        Args:
            reward_computer: RewardComputer实例
        """
        if hasattr(reward_computer, 'eval_stats'):
            self.judge_stats = reward_computer.eval_stats.copy()

    def get_wandb_logs(self) -> dict:
        """
        生成WandB日志字典

        Returns:
            适合wandb.log()的字典
        """
        logs = {}
        total = self.judge_stats['total_evaluations']

        if total == 0:
            return logs

        # 成功率统计
        logs['judge/success_rate'] = self.judge_stats['llm_judge_success'] / total
        logs['judge/parse_failure_rate'] = self.judge_stats['llm_judge_parse_failures'] / total
        logs['judge/api_failure_rate'] = self.judge_stats['llm_judge_api_failures'] / total
        logs['judge/total_calls'] = total

        # 判决分布
        judged = self.judge_stats['correct_predictions'] + self.judge_stats['incorrect_predictions']
        if judged > 0:
            logs['judge/correct_ratio'] = self.judge_stats['correct_predictions'] / judged
            logs['judge/correct_count'] = self.judge_stats['correct_predictions']
            logs['judge/incorrect_count'] = self.judge_stats['incorrect_predictions']

        return logs

    def print_summary(self):
        """打印统计摘要 (用于调试)"""
        total = self.judge_stats['total_evaluations']
        if total == 0:
            print("\n🤖 LLM Judge统计: 无评估记录")
            return

        print(f"\n🤖 LLM Judge统计 (总计: {total} 次):")
        print(f"  成功: {self.judge_stats['llm_judge_success']} ({self.judge_stats['llm_judge_success']/total*100:.1f}%)")
        print(f"  解析失败: {self.judge_stats['llm_judge_parse_failures']} ({self.judge_stats['llm_judge_parse_failures']/total*100:.1f}%)")
        print(f"  API失败: {self.judge_stats['llm_judge_api_failures']} ({self.judge_stats['llm_judge_api_failures']/total*100:.1f}%)")

        judged = self.judge_stats['correct_predictions'] + self.judge_stats['incorrect_predictions']
        if judged > 0:
            accuracy = self.judge_stats['correct_predictions'] / judged * 100
            print(f"  判决准确率: {accuracy:.1f}% (正确: {self.judge_stats['correct_predictions']}, 错误: {self.judge_stats['incorrect_predictions']})")


class CostTracker:
    """
    成本追踪器

    使用方法:
        tracker = CostTracker()
        tracker.add_cost(cost=0.01, is_executor=True)
        metrics = tracker.get_wandb_logs()
        wandb.log(metrics, step=step)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置统计"""
        self.total_cost = 0.0
        self.total_samples = 0
        self.executor_calls = 0
        self.judge_calls = 0

    def add_cost(self, cost: float, is_executor: bool = True):
        """
        添加成本记录

        Args:
            cost: 成本值
            is_executor: 是否为executor调用 (否则为judge调用)
        """
        self.total_cost += cost
        self.total_samples += 1

        if is_executor:
            self.executor_calls += 1
        else:
            self.judge_calls += 1

    def get_wandb_logs(self) -> dict:
        """
        生成WandB日志字典

        Returns:
            适合wandb.log()的字典
        """
        logs = {
            'cost/total_cost': self.total_cost,
            'cost/total_samples': self.total_samples,
            'cost/executor_calls': self.executor_calls,
            'cost/judge_calls': self.judge_calls,
        }

        if self.total_samples > 0:
            logs['cost/avg_cost_per_sample'] = self.total_cost / self.total_samples

        return logs

    def print_summary(self):
        """打印统计摘要 (用于调试)"""
        print(f"\n💰 成本统计:")
        print(f"  总成本: ${self.total_cost:.4f}")
        print(f"  样本数: {self.total_samples}")
        print(f"  Executor调用: {self.executor_calls}")
        print(f"  Judge调用: {self.judge_calls}")

        if self.total_samples > 0:
            avg_cost = self.total_cost / self.total_samples
            print(f"  平均成本/样本: ${avg_cost:.6f}")


# ============================================================================
# 集成到grpo_trainer.py的示例代码
# ============================================================================

def example_integration_train_step():
    """
    演示如何在train_step()中集成数据集监控

    这段代码应该插入到grpo_trainer.py的train_step()方法中
    """

    # ==================== 在train_step()开始处初始化 ====================
    # (插入到第307行后)

    # 初始化数据集指标收集器
    dataset_collector = DatasetMetricsCollector()

    # ==================== 在样本循环中收集数据 ====================
    # (修改第312-437行的循环)

    for sample_idx, sample in enumerate(batch):
        problem = sample['problem']
        ground_truth = sample['ground_truth']
        problem_type = sample['problem_type']
        source = sample.get('source', 'unknown')  # 🆕 获取数据集来源

        # ... (原有的工作流生成和执行代码)

        # 计算奖励和正确性
        if metadata['success']:
            reward = self.reward_computer.compute_reward(
                problem=problem,
                prediction=answer,
                ground_truth=ground_truth,
                problem_type=problem_type,
                metadata=metadata,
                test=sample.get('test', ''),
                entry_point=sample.get('entry_point', ''),
                source=source  # 🆕 传递source
            )

            correctness = reward  # 二元奖励: 1.0或0.0

            # 🆕 记录到数据集收集器
            dataset_collector.add_result(
                source=source,
                correctness=correctness,
                reward=reward,
                cost=cost
            )

        # ... (其余代码)

    # ==================== 在step末尾记录到WandB ====================
    # (插入到第513行前)

    # 获取数据集维度指标
    dataset_logs = dataset_collector.get_wandb_logs(step=step, prefix="dataset")
    wandb_log_data.update(dataset_logs)

    # 打印数据集统计摘要
    dataset_collector.print_summary()

    # 最终记录
    wandb.log(wandb_log_data, step=step)


def example_integration_evaluate_on_val_set():
    """
    演示如何在evaluate_on_val_set()中集成数据集监控

    这段代码应该插入到grpo_trainer.py的evaluate_on_val_set()方法中
    """

    # ==================== 在evaluate_on_val_set()开始处初始化 ====================
    # (插入到第674行后)

    # 初始化验证集数据集指标收集器
    val_dataset_collector = DatasetMetricsCollector()

    # ==================== 在样本循环中收集数据 ====================
    # (修改第678-736行的循环)

    for idx, sample in enumerate(val_batch):
        problem = sample['problem']
        ground_truth = sample['ground_truth']
        problem_type = sample['problem_type']
        source = sample.get('source', 'unknown')  # 🆕 获取数据集来源

        # ... (原有的工作流生成和执行代码)

        # 计算正确性
        if metadata['success']:
            correctness = self.reward_computer.compute_reward(
                problem=problem,
                prediction=answer,
                ground_truth=ground_truth,
                problem_type=problem_type,
                test=sample.get('test', ''),
                entry_point=sample.get('entry_point', ''),
                source=source  # 🆕 传递source
            )

            # 🆕 记录到验证集收集器
            val_dataset_collector.add_result(
                source=source,
                correctness=correctness,
                reward=correctness,
                cost=cost
            )

        # ... (其余代码)

    # ==================== 在evaluate_on_val_set()末尾记录到WandB ====================
    # (插入到第800行)

    # 获取验证集数据集维度指标
    val_dataset_logs = val_dataset_collector.get_wandb_logs(step=step, prefix="val")
    wandb.log(val_dataset_logs, step=step)

    # 打印验证集数据集统计摘要
    val_dataset_collector.print_summary()


def example_judge_monitoring():
    """
    演示如何在train_step()中监控LLM Judge性能

    这段代码应该插入到grpo_trainer.py的train_step()末尾
    """

    # ==================== 在train_step()末尾添加 ====================
    # (插入到第513行前)

    # 监控LLM Judge性能 (如果启用)
    if self.reward_computer.use_llm_judge:
        judge_collector = JudgeMetricsCollector()
        judge_collector.update_from_reward_computer(self.reward_computer)

        # 获取Judge指标
        judge_logs = judge_collector.get_wandb_logs()
        wandb_log_data.update(judge_logs)

        # 打印Judge统计摘要 (每10步)
        if step % 10 == 0:
            judge_collector.print_summary()


def example_cost_tracking():
    """
    演示如何在GRPOTrainer中添加成本追踪

    这段代码应该添加到grpo_trainer.py的__init__()和train_step()中
    """

    # ==================== 在__init__()中初始化 ====================
    # (插入到第79行后)

    # 初始化成本追踪器
    self.cost_tracker = CostTracker()

    # ==================== 在train_step()中记录成本 ====================
    # (在第353行后，每次执行后)

    # 记录执行成本
    self.cost_tracker.add_cost(cost=cost, is_executor=True)

    # ==================== 在train_step()末尾记录到WandB ====================
    # (插入到第513行前)

    # 获取成本指标
    cost_logs = self.cost_tracker.get_wandb_logs()
    wandb_log_data.update(cost_logs)

    # 打印成本统计 (每50步)
    if step % 50 == 0:
        self.cost_tracker.print_summary()


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试DatasetMetricsCollector")
    collector = DatasetMetricsCollector()

    # 添加一些测试数据
    collector.add_result('gsm8k', correctness=1.0, reward=1.0, cost=0.01)
    collector.add_result('gsm8k', correctness=1.0, reward=1.0, cost=0.02)
    collector.add_result('gsm8k', correctness=0.0, reward=0.0, cost=0.015)
    collector.add_result('math', correctness=1.0, reward=1.0, cost=0.03)
    collector.add_result('math', correctness=0.0, reward=0.0, cost=0.025)

    # 获取日志
    logs = collector.get_wandb_logs(step=100)
    print("\nWandB日志:")
    for key, value in logs.items():
        print(f"  {key}: {value}")

    # 打印摘要
    collector.print_summary()

    print("\n" + "="*60)
    print("🧪 测试JudgeMetricsCollector")

    # 模拟RewardComputer的统计数据
    class MockRewardComputer:
        def __init__(self):
            self.eval_stats = {
                'total_evaluations': 100,
                'llm_judge_success': 85,
                'llm_judge_parse_failures': 10,
                'llm_judge_api_failures': 5,
                'correct_predictions': 60,
                'incorrect_predictions': 25,
            }

    mock_rc = MockRewardComputer()
    judge_collector = JudgeMetricsCollector()
    judge_collector.update_from_reward_computer(mock_rc)

    # 获取日志
    judge_logs = judge_collector.get_wandb_logs()
    print("\nWandB日志:")
    for key, value in judge_logs.items():
        print(f"  {key}: {value}")

    # 打印摘要
    judge_collector.print_summary()

    print("\n" + "="*60)
    print("🧪 测试CostTracker")

    tracker = CostTracker()
    tracker.add_cost(0.01, is_executor=True)
    tracker.add_cost(0.02, is_executor=True)
    tracker.add_cost(0.005, is_executor=False)  # Judge调用

    # 获取日志
    cost_logs = tracker.get_wandb_logs()
    print("\nWandB日志:")
    for key, value in cost_logs.items():
        print(f"  {key}: {value}")

    # 打印摘要
    tracker.print_summary()

    print("\n✅ 所有测试通过！")
