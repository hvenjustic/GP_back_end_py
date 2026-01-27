# 📌 重要文档说明

## 📄 核心文档

### 1. LangExtract问题修复完整指南.md
**最重要的文档**，包含：
- 问题根本原因分析
- 完整的解决方案
- 重启 Worker 步骤
- 故障排查方法
- 经验教训总结

**遇到任何 langextract 相关问题，请先查看此文档。**

### 2. langextract_prompt.md
**当前使用的提示词文件**，用于 langextract 提取知识图谱。

**特点**：
- 只描述业务逻辑（实体类型、关系类型、抽取要求）
- 不包含 JSON 结构说明
- 由 config.yaml 中的 `prompt_path` 配置引用

**注意**：修改后必须重启 worker 才能生效。

### 3. config.yaml
**项目配置文件**，包含：
- 数据库连接配置
- Redis 配置
- LangExtract 配置（模型、API key、提示词路径等）
- 爬虫配置
- Agent 配置

---

## 🚀 快速重启 Worker

```bash
# 1. 停止所有旧进程
pkill -9 -f "celery.*worker"

# 2. 等待并清理缓存
sleep 3
cd /Users/zhangrui/Desktop/konwledge-graph/GP_back_end_py
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 3. 启动新 worker
celery -A app.services.crawl_tasks worker --loglevel=info --concurrency=1
```

---

## ✅ 验证修复成功

提交任务后，应该看到：

```
[INFO] langextract extract success: total_extractions=X
[INFO] langextract response task_id=Y entities=N relations=M
```

**不应该看到**：
```
[ERROR] Extraction text must be a string, integer, or float. Found: <class 'dict'>
```

---

## 📞 需要帮助

如果问题仍然存在，请检查：

1. **Worker 进程数量**
   ```bash
   ps aux | grep celery | grep -v grep
   # 应该只有 1 个进程
   ```

2. **代码是否更新**
   ```bash
   grep '"country": "USA",' app/services/langextract_client.py
   # 应该有匹配（说明使用平铺结构）
   ```

3. **查看日志**
   ```bash
   tail -100 worker.log
   ```

详细排查方法请参考 **LangExtract问题修复完整指南.md**。

