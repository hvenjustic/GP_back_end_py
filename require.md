
## Codex 提示词（直接复制使用）

你是一个资深 Python 后端/爬虫工程师。请为我实现一个“站内 URL 层次爬虫服务”，要求如下：

### 目标

实现一个 HTTP API 服务，用户调用 `POST /crawl` 传入一个 `root_url`（主页/种子 URL），系统立即异步返回 `job_id`。后台开始爬取该站点的站内链接（internal links），按 **BFS（广度优先）逐层爬取**。爬取过程中把每个页面的“站内子链接数组 childrens”写入 MySQL。用户可通过 `GET /status/{job_id}` 查询爬取状态与进度。

### 技术要求

* Python 3.11
* Web 框架：**FastAPI**（推荐）
* 爬取能力：使用 **crawl4ai 单页模式**获取页面内容与链接列表（要求使用 crawl4ai 的 API；如果版本差异，写适配层并在 README 说明你假设的 crawl4ai 调用方式）
* 异步任务：推荐 **Celery + Redis**（broker+backend），也可以用 RQ/Arq，但必须能：

  * `POST /crawl` 立即返回，不阻塞
  * 后台任务可并发抓取（可配置并发数）
  * `GET /status/{job_id}` 可查到实时进度
* 数据库：MySQL 8.0（使用 SQLAlchemy 或者 mysqlclient/pymysql 均可；推荐 SQLAlchemy）
* 配置：把 MySQL/Redis 连接信息放在 `config.yaml` 或 `.env`（我会自己填密码地址）

### 爬取策略（必须）

* **BFS 广度优先**：从 root_url 开始，depth=0；其内链为 depth=1；以此类推。
* 只继续深爬 **内链（internal link）**：

  * internal 的定义：与 root_url 的 `netloc` 相同（同域名）；可扩展为允许域名白名单（可选）
* 去重：同一规范化 URL 只爬一次（seen 集合）
* URL 规范化（必须实现）：

  * 相对链接转绝对 URL（基于当前页面 URL）
  * 移除 `#fragment`
  * 可配置是否移除常见跟踪参数（utm_* 等）
  * 可配置是否保留 query
* 过滤（必须有默认规则）：

  * 过滤非 http/https
  * 过滤静态资源后缀：`.jpg .png .gif .svg .css .js .ico .mp4 .zip .rar .pdf`（可配置）
* 限制（通过请求参数或配置提供默认值）：

  * `max_depth`（默认 3）
  * `max_pages`（默认 5000）
  * `concurrency`（默认 5）
  * `timeout`（默认 20s）
* 失败重试：对网络失败/临时错误进行有限次重试（默认 2 次），并记录失败原因
* **树形层级信息**：

  * 采用“首次发现 parent”规则：当一个新 URL 第一次被发现入库时，记录 `parent_url = source_url`，`depth = source_depth+1`；后续即便从别处再发现也不改变 parent（保证树稳定）
* 同时保留“图”信息（可选但强烈建议）：

  * 记录所有 `source -> child` 边（即使 child 已存在，也记录边），方便后续分析；但如果你觉得最小需求先不做，至少要把 childrens[] 存在 source 行里

### MySQL 数据模型（必须）

请创建表并在服务启动时自动初始化（或提供 alembic/migrations）。

**表 1：crawl_jobs**（存任务状态）
字段建议：

* `job_id` (PK, UUID)
* `root_url`
* `status`：`PENDING|RUNNING|DONE|FAILED|CANCELLED`
* `created_at`, `started_at`, `updated_at`, `finished_at`
* 进度字段：`discovered_count`, `queued_count`, `crawled_count`, `failed_count`, `current_depth`
* `error_message`（失败原因）
* `params`（JSON，保存 max_depth/max_pages 等）

**表 2：site_pages**（核心：每行一个 source url，含 childrens 数组）
字段要求（至少）：

* `url` (PK)
* `job_id`（索引）
* `childrens` JSON NOT NULL（数组，存该页面发现的站内链接，规范化后的绝对 URL）
* `parent_url` NULL（树的父节点）
* `depth` INT
* `crawled` BOOL
* `last_crawled` DATETIME NULL
* 其他按需：`status_code`, `title`, `canonical_url`, `content_hash`

> 注意：同一站点一个 job 的 url 可能重复；用 `(job_id, url)` 唯一更合理。
> 请你在实现中采用：`PRIMARY KEY (job_id, url)`，以便同一服务可并行跑多个站点/多次任务。

可选 **表 3：site_edges**（记录边）

* `(job_id, src_url, dst_url)` 唯一
* `anchor_text`（若 crawl4ai 可提供）
* `is_internal`

### API 设计（必须）

1. `POST /crawl`
   请求 JSON：

```json
{
  "root_url": "https://example.com/",
  "max_depth": 3,
  "max_pages": 5000,
  "concurrency": 5
}
```

响应（立即返回）：

```json
{
  "job_id": "uuid",
  "status_url": "/status/uuid"
}
```

2. `GET /status/{job_id}`
   返回：

```json
{
  "job_id": "...",
  "root_url": "...",
  "status": "RUNNING",
  "progress": {
    "discovered": 1200,
    "queued": 300,
    "crawled": 900,
    "failed": 12,
    "current_depth": 2
  },
  "timestamps": {...},
  "params": {...},
  "message": "optional human readable"
}
```

可选：

* `GET /tree/{job_id}`：返回一个简化树（从 site_pages 的 parent_url 构建），用于前端展示
* `POST /cancel/{job_id}`：取消任务（可选）

### crawl4ai 集成（必须）

* 使用 crawl4ai 的单页抓取能力获取：

  * 页面 URL（最终 URL）
  * 页面 title（若可取）
  * links（至少要能拿到内链/外链或原始 href 列表）
* 如果 crawl4ai 返回结构里已有 internal/external 分类，直接用；否则自己用 domain 判断分类。
* 写一个 `app/services/crawler_adapter.py`：封装 crawl4ai 的调用和返回结构，保证主逻辑不依赖具体字段名。
* 在 README 中写清楚你假设的 crawl4ai 版本/调用方式，以及如何安装。

### 后台任务调度（必须）

* Celery worker 从 Redis 获取 job，执行 BFS 爬取。
* 需要“断点续爬”：服务重启后 job 仍能继续/至少能正确反映状态（最低要求：job 状态与已爬页数在 MySQL 中可恢复；队列可丢失则将 job 标为 FAILED 并给出提示也可接受，但更佳是用 Redis 持久化队列或 MySQL queue 表）
* 并发抓取：

  * 使用 asyncio + semaphore 或者 Celery 并发任务分片均可
  * 但必须保证 BFS 的 depth 统计正确（你可以按“分层队列”推进：先处理 depth=0 的所有，再 depth=1… 或者允许并发但 depth 写入保持正确）

### 输出代码结构（必须）

请生成一个可运行的项目，建议结构：

* `app/main.py`（FastAPI）
* `app/routes/crawl_routes.py`（路由）
* `app/models.py`（SQLAlchemy models）
* `app/schemas.py`（请求/响应模型）
* `app/db.py`
* `app/services/crawl_tasks.py`（Celery tasks）
* `app/services/crawler_adapter.py`（crawl4ai 封装）
* `app/services/crawl_service.py`（BFS 调度、规范化、过滤）
* `app/core/logger.py`（日志）
* `config.example.yaml` 或 `.env.example`
* `requirements.txt`
* `docker-compose.yml`（可选：包含 redis + mysql；mysql 密码我会自己改）
* `README.md`（启动方式、示例调用、状态字段解释）

### 验收标准（必须满足）

* 启动 API：`uvicorn app.main:app --reload`
* 启动 worker：`celery -A app.services.crawl_tasks worker --loglevel=info`
* 调用 `POST /crawl` 返回 job_id 不阻塞
* `GET /status/{job_id}` 能看到进度递增直到 DONE/FAILED
* MySQL 中 `site_pages` 表每个 url 行都有 `childrens` JSON 数组
* BFS 层级正确：`depth=0` 是 root；root 的内链为 1；以此类推
* 能通过配置控制 max_depth/max_pages/concurrency

### 实现提示（请按此实现）

* BFS 主循环使用队列（deque）。为了断点续爬，推荐把“待爬队列”存在 Redis（list）并把 seen 存在 Redis set；同时把 depth/parent 保存在 MySQL。
  如果你不想依赖 Redis 结构，也可以在 MySQL 加一张 `crawl_queue` 表（job_id,url,status,depth,parent），但实现要简单稳定。
* `childrens` 字段只存 **站内链接**；外链可写入 edges 表或忽略（按你实现选择，但请在 README 说明）
* 注意 URL 规范化和去重，否则会爆炸式重复

### 分阶段交付（允许）

如果一次生成太复杂，你可以分阶段：

* Phase 1：FastAPI + MySQL + 同步爬取（单线程）跑通写库 + status（伪异步也可）
* Phase 2：Celery+Redis 真异步 + 并发 + 断点续爬
* Phase 3：增强：edges 表、tree API、cancel、canonical、content_hash

但最终输出应包含 Phase 2 的完整实现（异步任务 + status API）。

请开始生成完整代码，确保可运行，并在 README 给出 curl 示例。
