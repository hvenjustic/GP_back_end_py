# 🚨 紧急：正确重启 Worker

## 问题原因

**发现有 4 个旧的 celery worker 进程同时在运行！**

```
PID 37902 (8:10PM)
PID 31285 (7:29PM)  ← 旧进程
PID 29428 (7:18PM)  ← 旧进程
PID 37907 (8:10PM)
```

这些旧进程仍在使用**旧代码**（包含嵌套字典的 EXAMPLES），所以错误继续发生。

---

## ✅ 解决方案：彻底清理并重启

### 方法 1：使用自动脚本（推荐）

```bash
cd /Users/zhangrui/Desktop/konwledge-graph/GP_back_end_py
./restart_worker.sh
```

这个脚本会：
1. 杀死所有旧的 celery 进程
2. 清理 Python 缓存
3. 验证代码更新
4. 启动新的 worker

### 方法 2：手动执行

```bash
# 1. 强制杀死所有 celery 进程
pkill -9 -f "celery.*worker"

# 2. 等待确认
sleep 3
ps aux | grep celery | grep -v grep
# 应该没有输出

# 3. 清理 Python 缓存
cd /Users/zhangrui/Desktop/konwledge-graph/GP_back_end_py
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# 4. 重新启动（前台运行）
celery -A app.services.crawl_tasks worker --loglevel=info --concurrency=1
```

---

## 🔍 验证 Worker 已更新

### 启动后看到的日志应该包含：

```
[INFO] langextract extract params: model_id=gpt-4o-mini, ...
```

这是新增的调试日志，如果看到说明代码已更新。

### 提交任务后：

**✅ 成功标志**：
```
[INFO] langextract extract success: total_extractions=X
[INFO] langextract response task_id=59 entities=N relations=M
```

**❌ 如果仍然失败**，会看到更详细的日志：
```
[WARNING] Unexpected extraction_text type: type=dict value={...}
```

---

## ⚠️ 常见问题

### Q1: 为什么会有多个 worker 进程？

**原因**：
- 多次运行 `celery worker` 命令
- 没有正确停止旧进程
- 使用 `&` 后台运行但没有记录 PID

**解决**：
- 使用 `pkill -9` 强制杀死所有
- 只启动一个新的 worker
- 或者使用进程管理工具（如 supervisor）

### Q2: 如何确认只有一个 worker？

```bash
ps aux | grep celery | grep -v grep | wc -l
```

应该只输出 `1`（或 `2`，如果有子进程）。

### Q3: Worker 启动后立即退出？

**检查**：
1. 数据库连接是否正常
2. Redis 连接是否正常
3. 配置文件是否正确

```bash
# 查看详细错误
celery -A app.services.crawl_tasks worker --loglevel=debug
```

---

## 📋 完整检查清单

在重启后，按顺序检查：

- [ ] 1. 所有旧进程已停止
  ```bash
  ps aux | grep celery | grep -v grep
  # 应该没有输出
  ```

- [ ] 2. Python 缓存已清理
  ```bash
  find . -type d -name "__pycache__" | wc -l
  # 应该是 0
  ```

- [ ] 3. 代码已更新
  ```bash
  grep '"country":' app/services/langextract_client.py
  # 应该看到 "country": "USA",（平铺）
  ```

- [ ] 4. Worker 已启动
  ```bash
  ps aux | grep celery | grep -v grep | wc -l
  # 应该是 1 或 2
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

## 🎯 立即执行

### 一键重启

```bash
cd /Users/zhangrui/Desktop/konwledge-graph/GP_back_end_py && ./restart_worker.sh
```

### 或者手动执行

```bash
# 杀死旧进程
pkill -9 -f "celery.*worker"

# 等待 3 秒
sleep 3

# 清理缓存
cd /Users/zhangrui/Desktop/konwledge-graph/GP_back_end_py
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 启动新 worker
celery -A app.services.crawl_tasks worker --loglevel=info --concurrency=1
```

---

## 📞 如果问题仍然存在

### 提供以下信息：

1. **确认进程数量**
   ```bash
   ps aux | grep celery | grep -v grep
   ```

2. **确认代码版本**
   ```bash
   grep -A 5 '"name": "Acme Bio Inc."' app/services/langextract_client.py
   ```

3. **最新的 50 行日志**
   ```bash
   tail -50 worker.log
   ```

4. **Worker 启动命令**
   - 你用的是哪个命令？
   - `celery -A app.services.crawl_tasks` 还是 `celery -A app.main.celery_app`？

---

## 🎉 成功标志

启动后提交任务，应该看到：

```
[INFO] langextract extract params: model_id=gpt-4o-mini, extraction_passes=2, max_workers=1
[INFO] Starting sequential extraction passes for improved recall with 2 passes
[INFO] Starting extraction pass 1 of 2
[INFO] HTTP Request: POST https://yinli.one/v1/chat/completions "HTTP/1.1 200 OK"
[INFO] langextract extract success: total_extractions=15
[INFO] langextract response task_id=59 url=https://... entities=10 relations=5
```

**关键**：不再看到 `Extraction text must be a string` 错误！

---

**🚀 现在就执行 `./restart_worker.sh`！**

