# NICE: Neural Intelligence for Compound legal rEasoning

**基于强化学习的法律工作流自动优化系统**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 🎯 项目概述

NICE 是一个创新的法律AI系统，融合了 **AFlow**（工作流框架）和 **ROLL**（强化学习框架），通过训练小模型生成工作流代码来调度大模型执行复杂法律推理任务。

### 核心架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NICE 系统架构                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────┐         ┌──────────────────┐         ┌───────────┐  │
│   │  Qwen2.5-7B      │ 生成    │   Workflow Code  │ 执行    │ GPT-4o-   │  │
│   │  + LoRA          │────────▶│   (Python类)     │────────▶│ mini      │  │
│   │  (小模型)         │         │                  │         │ (大模型)   │  │
│   └────────┬─────────┘         └──────────────────┘         └─────┬─────┘  │
│            │                                                       │        │
│            │  GRPO优化                              奖励反馈        │        │
│            └───────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 核心创新

- **双模型协作**：小模型学习"如何组织工作流"，大模型负责"实际推理执行"
- **WA-GRPO算法**：Workflow-Aware GRPO，解决组内同分的全零优势问题
- **法律双系统支持**：中国法律（大陆法系）+ 美国法律（普通法系）
- **6个法律专用Operator**：针对法律推理任务设计的专业算子

---

## 🏛️ 法律模块

### 支持的法律体系

| 体系 | 特点 | 数据源 |
|------|------|--------|
| 🇨🇳 中国法律 | 成文法为主，司法解释补充 | CAIL2018, DISC-Law-SFT |
| 🇺🇸 美国法律 | 判例法为主，遵循先例原则 | LegalBench, CaseHOLD |

### 法律Operator

| Operator | 功能 | 中国特色 | 美国特色 |
|----------|------|----------|----------|
| `DirectAnswer` | 直接生成法律答案 | - | - |
| `CaseLearning` | 案例检索学习 | 指导性案例、典型案例 | Binding/Persuasive precedents |
| `StatuteLearning` | 法条检索学习 | 刑法/民法典 + 司法解释 | U.S. Code + CFR |
| `Debate` | 多角色辩论 | 原告/被告/法官视角 | 同左 |
| `LegalEnsemble` | 集成选择 | 多答案投票选优 | 同左 |
| `LegalRevise` | 法律修订 | 检查罪名表述规范 | 检查Bluebook引用格式 |

### 法律Workflow示例

```python
class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.retriever = LegalRetriever(data_dir="data/legal")
        self.case_learning = CaseLearning(self.llm, self.retriever)
        self.statute_learning = StatuteLearning(self.llm, self.retriever)
        self.debate = Debate(self.llm)
        self.legal_revise = LegalRevise(self.llm)

    async def __call__(self, problem: str):
        # 1. 案例学习 - 检索相关判例
        case_result = await self.case_learning(
            question=problem, jurisdiction="CN", top_k=3
        )

        # 2. 法条学习 - 检索适用法条
        statute_result = await self.statute_learning(
            question=problem, jurisdiction="CN"
        )

        # 3. 整合上下文
        context = f"{case_result['learning_record']}\n{statute_result['learning_record']}"

        # 4. 多角色辩论
        debate_result = await self.debate(
            question=problem, context=context, jurisdiction="CN"
        )

        # 5. 法律修订检查
        final = await self.legal_revise(
            answer=debate_result['final_answer'], question=problem
        )

        return final['revised_answer'], self.llm.get_usage_summary()["total_cost"]
```

---

## 🔬 算法详解

### GRPO (Group Relative Policy Optimization)

```python
# 对每个问题生成 K=4 个工作流
for problem in batch:
    workflows = model.generate(problem, num_return_sequences=4)
    rewards = [execute_and_evaluate(w) for w in workflows]

    # 组内归一化（核心思想）
    advantages = (rewards - mean(rewards)) / std(rewards)

    # PPO-style 策略更新
    loss = -min(ratio * adv, clip(ratio, 0.8, 1.2) * adv)
```

### WA-GRPO 改进

解决 GRPO 的「全零优势」问题：当组内所有回答奖励相同时，标准GRPO无法学习。

```python
# WA-GRPO: 使用 workflow 特征作为 tie-breaker
if std(rewards) < threshold:
    tie_breaker = (
        0.35 * diversity_score +      # 代码多样性
        0.25 * revise_gain +          # Revise改进幅度
        0.20 * exec_success +         # 执行成功度
        0.10 * efficiency +           # 运行效率
        0.10 * op_variety             # Operator覆盖度
    )
    rewards = rewards + alpha * tie_breaker  # alpha=0.12
```

### 5档奖励系统

```
奖励等级: [0.0, 0.2, 0.4, 0.7, 1.0]

评估维度（法律任务）:
├── legal_basis:   35%  # 法律依据准确性
├── reasoning:     25%  # 推理逻辑质量
├── conclusion:    20%  # 结论正确性
└── completeness:  20%  # 答案完整性
```

---

## 📁 项目结构

```
nice-main/
├── src/
│   ├── grpo_trainer.py           # GRPO训练器主类
│   ├── wa_grpo.py                # WA-GRPO优势计算
│   ├── rl_workflow_generator.py  # Qwen2.5-7B工作流生成
│   ├── aflow_executor.py         # AFlow执行引擎
│   ├── reward_computer.py        # 5档奖励计算
│   ├── data_manager.py           # 混合数据集管理
│   ├── workflow_validator.py     # 工作流代码验证
│   └── legal/                    # 法律模块
│       ├── operators.py          # 6个法律Operator
│       ├── retriever.py          # FAISS向量检索
│       ├── data_processor.py     # 法律数据处理
│       └── reward.py             # 法律奖励计算
├── config/
│   ├── training_legal.yaml       # 法律训练配置
│   └── aflow_llm.yaml            # OpenAI API配置
├── data/
│   └── legal/                    # 法律数据集
│       ├── cn/                   # 中国法律数据
│       └── us/                   # 美国法律数据
├── train.py                      # 训练入口
├── tests/
│   └── test_legal_module.py      # 法律模块测试
└── docs/                         # 文档
```

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (推荐)
- 显存 ≥ 24GB (用于Qwen2.5-7B)

### 安装

```bash
# 克隆仓库
git clone https://github.com/beita6969/legal.git
cd legal

# 安装依赖
pip install torch transformers peft accelerate
pip install faiss-cpu sentence-transformers  # 法律检索
pip install openai wandb                      # API和监控
```

### 配置API

编辑 `config/aflow_llm.yaml`:

```yaml
models:
  "gpt-4o-mini":
    api_type: "openai"
    base_url: "https://api.openai.com/v1"
    api_key: "YOUR_OPENAI_API_KEY"  # 替换为你的API Key
    model: "gpt-4o-mini"
```

### 启动训练

```bash
# 法律领域训练
python train.py --config config/training_legal.yaml

# 监控训练（需要wandb）
wandb login
python train.py --config config/training_legal.yaml
```

### 测试法律模块

```bash
python tests/test_legal_module.py
```

---

## ⚙️ 配置说明

### 训练配置 (`config/training_legal.yaml`)

```yaml
# 基本配置
exp_name: "legal_grpo_cn_us_dual"
max_steps: 500
rollout_batch_size: 6
num_return_sequences_in_group: 4   # K=4 (GRPO组大小)

# 法律数据比例
domain_ratios:
  legal_cn: 0.5   # 50% 中国法律
  legal_us: 0.5   # 50% 美国法律

# 模型配置
base_model: "Qwen/Qwen2.5-7B-Instruct"
lora_rank: 64
lora_alpha: 64

# WA-GRPO配置
wa_grpo:
  alpha: 0.12                # tie-breaker系数
  diversity_weight: 0.35
  exec_success_weight: 0.20
```

---

## 📊 性能指标

| 指标 | 说明 |
|------|------|
| `train/accuracy` | 训练集准确率 |
| `train/avg_reward` | 平均奖励 (0-1) |
| `grpo/zero_advantage_ratio` | 全零优势组比例 (越低越好) |
| `train/loss` | PPO损失 |
| `train/kl_div` | KL散度 |

---

## 📚 引用

如果本项目对你有帮助，请引用：

```bibtex
@software{nice2024,
  title={NICE: Neural Intelligence for Compound legal rEasoning},
  author={Zhang Mingda},
  year={2024},
  url={https://github.com/beita6969/legal}
}
```

### 相关工作

- [AFlow](https://github.com/geekan/MetaGPT) - Workflow框架
- [ROLL](https://github.com/alibaba/ROLL) - 强化学习框架
- [GRPO](https://arxiv.org/abs/2402.03300) - DeepSeek的组相对策略优化

---

## 📄 License

MIT License - 详见 [LICENSE](LICENSE)

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**⭐ 如果这个项目对你有帮助，请给个Star！**
