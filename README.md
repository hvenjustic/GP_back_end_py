# Internal URL BFS Crawler Service

FastAPI + Celery + Redis + MySQL service that crawls internal links with BFS using crawl4ai single-page mode.

## Features

- `POST /crawl` returns `job_id` immediately (async crawl)
- `GET /status/{job_id}` for progress
- Results stored in MySQL (`crawl_jobs` / `site_pages`)

## Requirements

- Python 3.11
- MySQL 8.0
- Redis

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Supports `config.yaml` or `.env`:

- copy `config.example.yaml` to `config.yaml`
- or copy `.env.example` to `.env`


Example config.yaml:

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
```

Notes:

- the first address in the list is used
- Redis db index defaults to 0 if omitted

Markdown filter (controls `fit_markdown` generation):

```yaml
crawl:
  markdown_filter:
    threshold: 0.22
    threshold_type: "dynamic"
    min_word_threshold: 0
    ignore_links: true
    ignore_images: true
```

## Run

The worker is started automatically when running `python app/main.py`.
Worker concurrency and log file are configured in `config.yaml` under `runtime`.

Startup will auto-install dependencies and Playwright browsers unless disabled in `config.yaml` runtime section.

Start API:

```bash
uvicorn app.main:app --reload
```

Start worker:

```bash
celery -A app.tasks worker --loglevel=info
```

## API

`POST /crawl`

- request: `{"root_url":"https://example.com/","max_depth":3,"max_pages":5000,"concurrency":5}`
- response: `{"job_id":"uuid","status_url":"/status/uuid"}`

`GET /status/{job_id}`

- returns `progress` (discovered/queued/crawled/failed/current_depth) and `params`

Optional:

- `POST /cancel/{job_id}` cancel a job
- `GET /tree/{job_id}` return a simplified tree (parent_url relations)

## Progress Fields

- `discovered`: unique URLs discovered
- `queued`: pending count (`discovered - crawled - failed`)
- `crawled`: successful pages
- `failed`: failed pages
- `current_depth`: current BFS depth

## Crawl Strategy

- BFS by depth (root is depth 0)
- only internal links (same netloc or `allowed_domains`)
- `childrens` stores internal links only (normalized absolute URLs)
- fragments removed; query/tracking params optional
- static extensions filtered (configurable)

## crawl4ai Version Assumption

The adapter in `app/crawler_adapter.py` assumes crawl4ai provides `AsyncWebCrawler` or `WebCrawler`,
with one of `arun`/`run`/`crawl`, and returns `links` (either a list or internal/external groups).
If your version differs, adjust the adapter accordingly.

## crawl4ai 0.7.x CrawlResult JSON

```json
{
  "url": "https://example.com",
  "html": "<html>...</html>",
  "fit_html": "<html>...</html>",
  "success": true,
  "cleaned_html": "<html>...</html>",
  "media": {
    "images": [
      {
        "src": "https://example.com/logo.png",
        "data": "",
        "alt": "logo",
        "desc": "",
        "score": 0,
        "type": "image",
        "group_id": 0,
        "format": "png",
        "width": 256
      }
    ],
    "videos": [],
    "audios": [],
    "tables": []
  },
  "links": {
    "internal": [
      {
        "href": "https://example.com/about",
        "text": "About",
        "title": "",
        "base_domain": "example.com",
        "head_data": null,
        "head_extraction_status": null,
        "head_extraction_error": null,
        "intrinsic_score": null,
        "contextual_score": null,
        "total_score": null
      }
    ],
    "external": []
  },
  "downloaded_files": [],
  "js_execution_result": {},
  "screenshot": null,
  "pdf": null,
  "mhtml": null,
  "markdown": {
    "raw_markdown": "# Title",
    "markdown_with_citations": "# Title",
    "references_markdown": "",
    "fit_markdown": null,
    "fit_html": null
  },
  "extracted_content": null,
  "metadata": {},
  "error_message": "",
  "session_id": null,
  "response_headers": {},
  "status_code": 200,
  "ssl_certificate": null,
  "dispatch_result": null,
  "redirected_url": "https://example.com",
  "network_requests": [],
  "console_messages": [],
  "tables": [
    {
      "headers": ["H1", "H2"],
      "rows": [["A1", "A2"]],
      "caption": "",
      "summary": ""
    }
  ]
}
```

## Tables

- `crawl_jobs`: job status and progress
- `site_pages`: pages and childrens (internal link array); `url` stores full URL (VARCHAR(1024)) and `url_hash` (SHA-256 hex) is used for equality lookups with `job_id`

