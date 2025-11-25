# Train Restored v10 - 完整错误分析报告

## 数据集配置确认
- **训练集**: `train_final_clean.jsonl` (2000 条)
- **验证/测试集**: `test.jsonl` (100 条)
- **比例**: Math (800/40), QA (600/30), Code (600/30)

## 错误类型统计

### 1. 代码结构性错误 (Code Structural Errors)
**出现频率**: 37 次 (`NameError`)

**典型案例**:
```
Error: name 'solve' is not defined
预测: \boxed{}
评分: ❌ 0.0
```

**原因分析**:
- 模型生成的代码缺少 `solve()` 函数定义
- 或者函数名拼错（如 `solution()` vs `solve()`）
- 这是 **Code Generation Quality** 问题

---

### 2. 算子未初始化错误 (Operator Initialization Error)
**出现频率**: 26 次 (`AttributeError: 'revise'`)

**原因分析**:
- 模型在 `__init__` 中只初始化了部分算子 (如 `review`)
- 但在 `__call__` 的条件分支中调用了未初始化的 `revise`
- 这是 **Workflow Planning Bug**

---

### 3. 格式提取错误 (Output Formatting Errors)
**出现频率**: 多次

**案例 A: 空盒子**
```
预测: \boxed{}
真值: 50
```
- 代码执行成功，但提取 `locals()['result']` 失败（KeyError）

**案例 B: 代码泄漏**
```
预测: \boxed{def solve(): ...代码...}
真值: 50
```
- 返回了代码本身而不是执行结果

---

### 4. QA 答案判决异常 (Judge Anomaly)
**案例**:
```
预测: "might dream" (完整短语)
真值: "E" (单字母)
评分: ✅ 1.0 (Judge 认为等价！)
```

**分析**:
- LLM Judge 可能推断出这是多选题，E 是正确选项的编号
- Judge "脑补"了选项内容与编号的等价性
- 这导致了"假阳性"（应判错却判对）

---

### 5. 验证集统计Bug (Critical Bug - 已确认)
**问题**: `grpo_trainer.py` Line 737
```python
num_correct = sum(1 for score in correctness_scores if score >= 5.0)
```
- 使用了 10 分制的阈值 (5.0)，但实际分数是 0~1
- 导致报告的验证准确率为 **0%**，而实际应为 **~70%**

---

## 改进建议 (优先级排序)

### P0 (立即修复):
1. 修复验证集阈值 Bug (`>= 5.0` -> `>= 0.9`)

### P1 (短期优化):
2. 强化 Prompt 的 `[STRICT RULE]` 部分，减少 AttributeError
3. 在 Workflow 模板中增加更明确的代码提取示例

### P2 (中期优化):
4. 清洗训练集中的多选题数据（统一答案格式）
5. 优化代码执行环境的错误处理和结果提取机制

### P3 (长期目标):
6. 让 RL 自然进化，等待错误率随 Step 增加而下降


