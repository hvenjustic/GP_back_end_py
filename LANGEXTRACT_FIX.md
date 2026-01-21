# LangExtract 提取错误修复说明

## 问题描述

在使用 langextract 提取知识图谱时遇到错误：
```
Extraction text must be a string, integer, or float. Found: <class 'dict'>
```

## 根本原因

LLM 模型（gpt-4o-mini）返回的 JSON 数据中，`extraction_text` 字段被错误地返回为**字典类型**，而不是**字符串类型**。

这个问题的发生原因有三个层面：

### 1. **模型理解偏差**
   - LLM 模型在看到 `attributes` 是对象后，可能误认为 `extraction_text` 也应该是对象
   - 即使提示词中说明了要求，某些情况下模型仍可能输出错误格式

### 2. **提示词不够明确**
   - 原提示词虽然说明了要求，但强调不够
   - 缺少反例说明（什么是错误的格式）

### 3. **代码缺少防御性处理**
   - 原代码直接假设 LLM 返回格式正确
   - 没有处理异常格式的容错机制

## 解决方案

采用**双重防护**策略：

### ✅ 方案 1：优化提示词（预防）

**修改文件**: `langextract_prompt.md`

#### 改进点：
1. **增强视觉强调**：使用 `⚠️` 和加粗文字强调关键要求
2. **增加正确示例**：标注 "必须是字符串" 的注释
3. **增加错误示例**：明确展示什么是错误的格式（字典、数组、null）
4. **重复强调**：在多处重复说明 `extraction_text` 必须是字符串

#### 核心改动：
```markdown
## ⚠️ 关键要求：extraction_text 必须是字符串

**`extraction_text` 字段必须是字符串类型（string），绝对不能是数组（array）、对象（object）或 null。**
```

### ✅ 方案 2：代码防御性处理（容错）

**修改文件**: `app/services/graph_service.py`

#### 新增函数：`_normalize_extraction_text()`

这个函数会自动处理 LLM 返回的错误格式：

```python
def _normalize_extraction_text(extraction_text: Any, attrs: dict[str, Any], ex_class: str | None) -> str:
    """
    将 extraction_text 规范化为字符串类型。
    防御性处理 LLM 返回错误格式的情况。
    """
```

#### 处理策略：

1. **如果是字典**：
   - 尝试从字典中提取 `text`、`value`、`content`、`name` 等键的值
   - 如果找不到，转为 JSON 字符串
   - 记录警告日志

2. **如果是列表**：
   - 取第一个非空元素
   - 记录警告日志

3. **如果是 None 或其他类型**：
   - 对于实体：使用 `name` 作为备用值
   - 对于关系：使用 `source + type + target` 组成字符串
   - 记录警告日志

### ✅ 方案 3：强化 OpenAI 模型配置

**修改文件**: `app/services/langextract_client.py`

#### 改进点：
1. **启用 JSON 验证**：`validate_json=True`
2. **强制 JSON 响应格式**：`response_format={"type": "json_object"}`

这样可以确保 OpenAI 模型返回标准的 JSON 格式。

## 验证方法

### 1. 检查日志
运行后检查日志，如果看到以下警告，说明代码正在自动修复错误格式：
```
WARNING: extraction_text is dict (should be string): {...}, attempting to fix
```

### 2. 成功标志
如果不再看到 `Extraction text must be a string, integer, or float` 错误，说明修复成功。

### 3. 监控提取结果
检查提取的实体和关系数据是否正常：
```python
# 在日志中查看
langextract response task_id=X url=Y entities=N relations=M
```

## 后续建议

### 短期（已完成）
- ✅ 优化提示词，增强格式要求说明
- ✅ 增加代码防御性处理
- ✅ 强化 OpenAI 模型配置

### 中期（建议）
- 考虑使用更强大的模型（如 gpt-4o 或 claude-3.5-sonnet）
- 收集错误日志，分析常见错误模式
- 定期审查提取质量

### 长期（建议）
- 建立数据质量监控机制
- 增加单元测试覆盖边界情况
- 考虑使用结构化输出 API（如 OpenAI 的 Structured Outputs）

## 配置文件说明

当前配置（`config.yaml`）：
```yaml
langextract:
  model_id: "gpt-4o-mini"
  openai_api_key: "sk-..."
  openai_base_url: "https://yinli.one/v1"
  extraction_passes: 2
  max_workers: 1
  max_char_buffer: 1200
  prompt_path: "langextract_prompt.md"
```

### 关键参数：
- `extraction_passes: 2` - 执行 2 轮提取，提高准确性
- `max_workers: 1` - 单线程处理，避免并发问题
- `max_char_buffer: 1200` - 每次最多处理 1200 字符

## 测试数据结构

### 正确的数据结构：
```json
{
  "extractions": [
    {
      "extraction_class": "entity",
      "extraction_text": "Acme Bio Inc.",  ← 字符串 ✅
      "attributes": {
        "name": "Acme Bio Inc.",
        "type": "Company",
        "description": "A biotech company",
        "extra": {"country": "USA"}
      }
    }
  ]
}
```

### 错误的数据结构（现在会被自动修复）：
```json
{
  "extractions": [
    {
      "extraction_class": "entity",
      "extraction_text": {"text": "Acme Bio Inc."},  ← 字典 ❌（会被修复）
      "attributes": {...}
    }
  ]
}
```

## 总结

通过这三个层面的改进：
1. **提示词优化** - 预防错误发生
2. **代码防御** - 自动修复错误
3. **模型配置** - 强制正确格式

现在系统具备了更强的鲁棒性，即使 LLM 返回错误格式，也能自动修复并继续处理。

---

**修改文件列表：**
- ✅ `langextract_prompt.md` - 优化提示词
- ✅ `app/services/graph_service.py` - 增加防御性处理
- ✅ `app/services/langextract_client.py` - 强化模型配置

