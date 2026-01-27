# Python 后端（GP_back_end_py）

## 项目说明
本项目是知识图谱爬取与抽取服务，基于 FastAPI + Celery + Redis + MySQL，使用 crawl4ai 进行站点 BFS 爬取，并通过 LangExtract/LLM 生成图谱数据。

## 当前已实现功能
- 站点任务录入与入队：`POST /api/tasks`、`POST /api/tasks/crawl`、`GET /api/tasks/status`、`POST /api/queues/clear`
- 爬虫任务与状态：`POST /crawl`、`GET /status/{job_id}`、`POST /cancel/{job_id}`、`GET /tree/{job_id}`
- 结果与商品列表：`GET /api/results`、`GET /api/results/{id}`、`GET /api/products`（仅 graph_json 非空）
- 图谱抽取与可视化：`POST /graph/build`、`POST /api/results/graph`、`POST /api/results/graph/batch`、`GET /api/results/graph/status`、`GET /api/results/{id}/graph_view`
- 预处理状态：`GET /api/results/preprocess/status`
- 图谱点位：`GET /api/graph_locate`（读取 site_tasks.geo_location）
- Agent 对话与历史：`GET /api/chat/agent/stream`（SSE）、`POST/GET /api/agent/sessions`、`GET /api/agent/sessions/{id}`
- 数据落库：`crawl_jobs`、`site_pages`、`site_tasks`、`agent_sessions`、`agent_messages`

## 技术栈
- FastAPI、SQLAlchemy、Celery、Redis、MySQL
- crawl4ai（爬虫适配）、LangExtract/LLM（图谱抽取）

## 安装
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置
使用 `config.yaml`：

```yaml
mysql:
  dbMysqlAddress:
    - "host:port"
  dbMysqlUserName: "root"
  dbMysqlPassword: "password"
  dbMysqlDatabaseName: "crawl_db"

redis:
  dbAddress:
    - "host:port"
  dbPassWord: "password"
  dbRedisDb: 0

crawl:
  max_depth: 4
  max_pages: 500
  concurrency: 5
  timeout: 20
  retries: 2
  strip_query: false
  strip_tracking_params: true

runtime:
  worker_concurrency: 1
  worker_log_file: "worker.log"

langextract:
  model_id: "gpt-4o-mini"
  openai_api_key: "YOUR_KEY"
  openai_base_url: "https://example.com/v1"
  prompt_path: "langextract_prompt.md"

agent:
  model: "gpt-4o-mini"
  api_key: "YOUR_KEY"
  base_url: "https://example.com/v1"
  use_tools: true
  tools_enabled:
    - "crawl_site"
    - "build_graph"
```

## 启动
开发模式（自动拉起 worker）：
```bash
python app/main.py
```

或分别启动：
```bash
uvicorn app.main:app --reload
celery -A app.services.crawl_tasks worker --loglevel=info --concurrency=1
```

## 开发注意
- 启动时会自动安装依赖与 Playwright 浏览器（可在 `config.yaml` 的 `runtime` 中关闭）。
- 使用 `python app/main.py` 退出时会强杀 worker 并清空 Redis（仅用于开发调试）。
