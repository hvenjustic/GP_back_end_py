#!/bin/bash

# 重启 Celery Worker 脚本

echo "================================================"
echo "🔄 重启 Celery Worker"
echo "================================================"

# 1. 停止所有旧进程（优雅退出优先）
echo ""
echo "📍 步骤 1: 尝试优雅停止 celery worker 进程..."
pkill -TERM -f "celery.*worker" 2>/dev/null

WAIT_SECONDS=10
for ((i=1; i<=WAIT_SECONDS; i++)); do
    sleep 1
    RUNNING=$(ps aux | grep -i "celery.*worker" | grep -v grep | wc -l)
    if [ "$RUNNING" -eq 0 ]; then
        break
    fi
done

if [ "${RUNNING:-0}" -gt 0 ]; then
    echo "⚠️  优雅退出未完成，执行强制终止..."
    pkill -9 -f "celery.*worker" 2>/dev/null
    sleep 1
fi

# 2. 确认已停止
OLD_PROCESSES=$(ps aux | grep -i "celery.*worker" | grep -v grep | wc -l)
if [ "$OLD_PROCESSES" -gt 0 ]; then
    echo "⚠️  警告: 仍有 $OLD_PROCESSES 个 celery 进程在运行"
    echo "   手动检查: ps aux | grep celery"
else
    echo "✅ 所有旧进程已停止"
fi

# 3. 清理 Python 缓存
echo ""
echo "📍 步骤 2: 清理 Python 缓存..."
cd "$(dirname "$0")"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
echo "✅ Python 缓存已清理"

# 4. 验证代码更新
echo ""
echo "📍 步骤 3: 验证代码更新..."
if grep -q '"country": "USA",' app/services/langextract_client.py; then
    echo "✅ EXAMPLES 已更新（平铺结构）"
else
    echo "⚠️  警告: EXAMPLES 可能未更新"
fi

# 5. 启动新的 worker
echo ""
echo "📍 步骤 4: 启动新的 celery worker..."
echo ""
echo "================================================"
echo "Worker 将在前台运行，按 Ctrl+C 停止"
echo "================================================"
echo ""

# 使用正确的模块路径（根据项目配置）
celery -A app.services.crawl_tasks worker --loglevel=info --concurrency=1

# 或者如果是另一个模块：
# celery -A app.main.celery_app worker --loglevel=info

echo ""
echo "Worker 已停止"
