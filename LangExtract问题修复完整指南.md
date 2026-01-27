# LangExtract 问题修复完整指南

## 📋 问题描述

**错误信息**：
```
Extraction text must be a string, integer, or float. Found: <class 'dict'>
```

**影响**：langextract 提取知识图谱时抛出异常，导致任务失败。

---

## 🎯 根本原因

经过多轮分析，找到真正的根本原因：

**`EXAMPLES` 中的 `attributes` 包含了嵌套字典 `"extra": {...}`，违反了 langextract 的类型约束。**

### 错误的 EXAMPLES 定义

```python
# ❌ 错误：包含嵌套字典
lx.data.Extraction(
    extraction_class="entity",
    extraction_text="Acme Bio Inc.",
    attributes={
        "name": "Acme Bio Inc.",
        "type": "Company",
        "description": "...",
        "extra": {"country": "USA"},  # ← 嵌套字典
    },
)
```

### 为什么会出错

根据 **langextract 官方文档**：
> `attributes` 中只能包含简单类型（string, int, float），不能包含嵌套的字典或列表。

**错误链**：
```
EXAMPLES 中有嵌套字典
    ↓
langextract 从 EXAMPLES 学习结构
    ↓
LLM 模仿并返回嵌套字典
    ↓
langextract 验证类型时发现违反约束
    ↓
在库内部抛出异常（我们的代码无法捕获）
```

---

## ✅ 解决方案

### 核心原则

**`attributes` 中所有字段的值必须是简单类型（string, int, float），不能是字典或列表。**

### 修改方案：平铺结构

#### ❌ 修改前（嵌套）
```python
attributes={
    "name": "Acme Bio Inc.",
    "type": "Company",
    "description": "...",
    "extra": {"country": "USA", "stage": "Series A"},
}
```

#### ✅ 修改后（平铺）
```python
attributes={
    "name": "Acme Bio Inc.",
    "type": "Company",
    "description": "...",
    "country": "USA",      # 直接平铺
    "stage": "Series A",   # 直接平铺
}
```

---

## 🔧 已完成的修改

### 1. 修改 EXAMPLES 定义

**文件**：`app/services/langextract_client.py`

**核心改动**：
- ✅ 移除所有 `"extra": {...}` 嵌套字典
- ✅ 将字段平铺到 `attributes` 顶层（如 `country`, `role`, `stage`, `date`）
- ✅ 所有 attributes 只包含简单类型

### 2. 更新数据提取逻辑

**文件**：`app/services/graph_service.py`

**核心改动**：
- ✅ `_extract_graph_items()` 适配平铺结构
- ✅ 从平铺的 attributes 中提取字段
- ✅ 重新构建 `extra` 字典（用于向后兼容）
- ✅ 添加类型检查和警告日志

### 3. 优化提示词

**文件**：`langextract_prompt.md`

**核心改动**：
- ✅ 移除所有 JSON 结构说明
- ✅ 只保留业务逻辑描述（实体类型、关系类型、抽取要求）
- ✅ 让 langextract 自动处理 JSON schema

**设计理念**：
- `prompt_description`：只描述业务逻辑（要提取什么）
- `examples`：定义数据结构（langextract 自动学习）
- 不要在提示词中重复定义结构，避免冲突

### 4. 添加调试日志

**新增日志**：
```python
# 提取参数
logger.info("langextract extract params: model_id=%s, ...")

# 提取成功
logger.info("langextract extract success: total_extractions=%d")

# 类型检查
logger.debug("extraction #%d: class=%s, text_type=%s")

# 类型异常
logger.warning("Unexpected extraction_text type: type=%s value=%s")
logger.warning("Nested dict found in entity attributes key=%s")
```

---

## 🚀 如何重启 Worker

### ⚠️ 重要：必须彻底重启

由于可能有多个旧的 worker 进程在运行（使用旧代码），必须：
1. 杀死所有旧进程
2. 清理 Python 缓存
3. 启动新的 worker

### 完整重启步骤

```bash
# 1. 强制杀死所有 celery 进程
pkill -9 -f "celery.*worker"

# 2. 等待确认
sleep 3
ps aux | grep celery | grep -v grep
# 应该没有输出，如果有，重复步骤 1

# 3. 清理 Python 缓存
cd /Users/zhangrui/Desktop/konwledge-graph/GP_back_end_py
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# 4. 启动新 worker（前台运行）
celery -A app.services.crawl_tasks worker --loglevel=info --concurrency=1

# 或者使用你的启动命令
# celery -A app.main.celery_app worker --loglevel=info
```

### 验证重启成功

启动后应该看到日志：
```
[INFO] langextract extract params: model_id=gpt-4o-mini, ...
```

如果看到这条日志，说明新代码已生效。

---

## 📊 预期效果

### ✅ 成功的标志

提交任务后，日志应该显示：

```
[INFO] langextract extract params: model_id=gpt-4o-mini, extraction_passes=2
[INFO] Starting sequential extraction passes for improved recall with 2 passes
[INFO] HTTP Request: POST https://yinli.one/v1/chat/completions "HTTP/1.1 200 OK"
[INFO] langextract extract success: total_extractions=15
[INFO] langextract response task_id=59 url=https://... entities=10 relations=5
```

**关键**：不再看到此错误：
```
[ERROR] Extraction text must be a string, integer, or float. Found: <class 'dict'>
```

### ⚠️ 如果仍有问题

现在会看到更详细的调试日志：

```
[WARNING] Unexpected extraction_text type: type=dict value={...}
[DEBUG] extraction #0: class=entity, text_type=dict
[WARNING] Nested dict found in entity attributes key=extra
```

这些日志会帮助进一步定位问题。

---

## 🔍 故障排查

### 问题 1：错误仍然发生

**可能原因**：
- Worker 没有完全重启
- 仍有旧进程在运行
- Python 缓存未清理

**解决方法**：
```bash
# 检查进程数量
ps aux | grep celery | grep -v grep | wc -l
# 应该只有 1 个（或 0 如果未启动）

# 如果有多个，强制全部杀死
pkill -9 -f celery
sleep 3

# 重新启动
celery -A app.services.crawl_tasks worker --loglevel=info --concurrency=1
```

### 问题 2：看不到新的调试日志

**可能原因**：
- 代码未更新
- 启动的是错误的模块

**解决方法**：
```bash
# 验证代码已更新
grep '"country": "USA",' app/services/langextract_client.py
# 应该有匹配（平铺结构）

grep '"extra": {' app/services/langextract_client.py
# 应该没有匹配（或只在注释中）

# 确认启动命令
# 检查 config.yaml 或项目文档，确认正确的 celery app 路径
```

### 问题 3：提取速度很慢

**优化配置**（如果需要）：
```yaml
# config.yaml
langextract:
  max_workers: 3          # 增加并发（默认 1）
  extraction_passes: 1    # 减少轮数（默认 2，可能降低质量）
  max_char_buffer: 2000   # 增加缓冲区（默认 1200）
```

---

## 🎓 经验教训

### 关键认识

1. **EXAMPLES 定义至关重要**
   - langextract 从 EXAMPLES 学习数据结构
   - EXAMPLES 中的错误会被 LLM 模仿
   - 必须严格遵守类型约束

2. **类型约束必须遵守**
   - `extraction_text` 必须是 string/int/float
   - `attributes` 中的值必须是 string/int/float
   - 不能有嵌套的字典或列表

3. **提示词职责明确**
   - `prompt_description` 只描述业务逻辑
   - 不要在提示词中定义 JSON 结构
   - 让 langextract 从 EXAMPLES 自动生成 schema

4. **多进程问题**
   - 确保只有一个 worker 进程运行
   - 重启时必须杀死所有旧进程
   - 清理 Python 缓存确保使用新代码

### 设计原则

```
简单类型 > 复杂类型
平铺结构 > 嵌套结构
遵守约束 > 灵活定义
单个进程 > 多个进程
```

---

## 📚 相关文件

### 修改的代码文件

- `app/services/langextract_client.py` - EXAMPLES 定义，添加调试日志
- `app/services/graph_service.py` - 数据提取逻辑，类型检查
- `langextract_prompt.md` - 提示词（只描述业务逻辑）

### 配置文件

- `config.yaml` - langextract 配置参数

---

## 🎯 快速检查清单

在重启后，按顺序检查：

- [ ] 1. 所有旧进程已停止
  ```bash
  ps aux | grep celery | grep -v grep
  # 应该只有 1 个或没有
  ```

- [ ] 2. Python 缓存已清理
  ```bash
  find . -type d -name "__pycache__" | wc -l
  # 应该是 0
  ```

- [ ] 3. 代码已更新
  ```bash
  grep '"country":' app/services/langextract_client.py
  # 应该看到平铺结构
  ```

- [ ] 4. Worker 已启动
  ```bash
  ps aux | grep celery | grep -v grep | wc -l
  # 应该是 1
  ```

- [ ] 5. 看到新的调试日志
  ```
  [INFO] langextract extract params: ...
  ```

- [ ] 6. 没有类型错误
  ```
  # 不应该看到：
  [ERROR] Extraction text must be a string...
  ```

---

## 💡 最佳实践

### 日常操作

1. **重启 Worker**
   ```bash
   pkill -9 -f "celery.*worker"
   sleep 3
   celery -A app.services.crawl_tasks worker --loglevel=info --concurrency=1
   ```

2. **查看日志**
   ```bash
   tail -f worker.log
   # 或者前台运行直接看输出
   ```

3. **检查进程**
   ```bash
   ps aux | grep celery | grep -v grep
   ```

### 修改代码后

1. 必须重启 worker
2. 清理 Python 缓存
3. 验证日志中有新代码的输出

### 调试问题

1. 开启 INFO 或 DEBUG 日志级别
2. 使用小的测试数据
3. 查看详细的错误堆栈
4. 检查 LLM 的原始响应

---

## 🎉 总结

### 问题本质
`EXAMPLES` 中的 `attributes` 包含嵌套字典，违反了 langextract 的类型约束。

### 解决方案
将所有嵌套字段平铺到 `attributes` 顶层，只使用简单类型。

### 关键步骤
1. ✅ 修改 EXAMPLES（移除嵌套字典）
2. ✅ 更新数据提取逻辑（适配平铺结构）
3. ✅ 优化提示词（只描述业务逻辑）
4. ✅ 添加调试日志（便于排查问题）
5. 🔄 **彻底重启 Worker**（杀死所有旧进程）
6. 🧪 **验证效果**（观察日志）

### 成功标志
- 不再看到 `Extraction text must be a string` 错误
- 看到 `langextract extract success` 日志
- 成功提取实体和关系

---

**如需帮助，请提供**：
1. `ps aux | grep celery` 的完整输出
2. 最近 100 行日志（`tail -100 worker.log`）
3. 代码验证结果（`grep '"extra":' app/services/langextract_client.py`）

