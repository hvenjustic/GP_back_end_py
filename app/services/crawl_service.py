from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
from urllib.parse import parse_qsl, urljoin, urlparse, urlunparse, urlencode, urldefrag

from sqlalchemy.dialects.mysql import insert

from app.config import Settings
from app.services.crawler_adapter import Crawl4AIAdapter
from app.models import CrawlJob, SitePage

TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "gclid",
    "fbclid",
}


logger = logging.getLogger(__name__)


@dataclass
class CrawlParams:
    max_depth: int
    max_pages: int
    concurrency: int
    timeout: int
    retries: int
    strip_query: bool
    strip_tracking_params: bool
    static_extensions: list[str]
    allowed_domains: list[str]

    def to_dict(self) -> dict:
        return {
            "max_depth": self.max_depth,
            "max_pages": self.max_pages,
            "concurrency": self.concurrency,
            "timeout": self.timeout,
            "retries": self.retries,
            "strip_query": self.strip_query,
            "strip_tracking_params": self.strip_tracking_params,
            "static_extensions": self.static_extensions,
            "allowed_domains": self.allowed_domains,
        }

    @classmethod
    def from_dict(cls, data: dict, settings: Settings) -> "CrawlParams":
        return cls(
            max_depth=int(data.get("max_depth", settings.max_depth)),
            max_pages=int(data.get("max_pages", settings.max_pages)),
            concurrency=int(data.get("concurrency", settings.concurrency)),
            timeout=int(data.get("timeout", settings.timeout)),
            retries=int(data.get("retries", settings.retries)),
            strip_query=bool(data.get("strip_query", settings.strip_query)),
            strip_tracking_params=bool(
                data.get("strip_tracking_params", settings.strip_tracking_params)
            ),
            static_extensions=data.get("static_extensions", settings.static_extensions),
            allowed_domains=data.get("allowed_domains", settings.allowed_domains),
        )


@dataclass
class FetchResult:
    url: str
    success: bool
    status_code: int | None
    title: str | None
    final_url: str | None
    links: list[str]
    internal_links: list[str] | None
    external_links: list[str] | None
    canonical_url: str | None
    html: str | None
    fit_markdown: str | None
    error: str | None = None


def build_crawl_params(settings: Settings, request) -> CrawlParams:
    max_depth = request.max_depth if request.max_depth is not None else settings.max_depth
    max_pages = request.max_pages if request.max_pages is not None else settings.max_pages
    concurrency = (
        request.concurrency if request.concurrency is not None else settings.concurrency
    )
    timeout = request.timeout if request.timeout is not None else settings.timeout
    retries = request.retries if request.retries is not None else settings.retries
    strip_query = (
        request.strip_query if request.strip_query is not None else settings.strip_query
    )
    strip_tracking_params = (
        request.strip_tracking_params
        if request.strip_tracking_params is not None
        else settings.strip_tracking_params
    )

    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if timeout < 1:
        raise ValueError("timeout must be >= 1")
    if retries < 0:
        raise ValueError("retries must be >= 0")

    return CrawlParams(
        max_depth=max_depth,
        max_pages=max_pages,
        concurrency=concurrency,
        timeout=timeout,
        retries=retries,
        strip_query=strip_query,
        strip_tracking_params=strip_tracking_params,
        static_extensions=settings.static_extensions,
        allowed_domains=settings.allowed_domains,
    )


def normalize_url(url: str, base_url: str, params: CrawlParams) -> str | None:
    if not url:
        return None
    url = url.strip()
    absolute = urljoin(base_url, url)
    absolute, _fragment = urldefrag(absolute)
    parsed = urlparse(absolute)

    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    query = parsed.query

    if params.strip_query:
        query = ""
    elif params.strip_tracking_params and query:
        filtered = [
            (k, v)
            for k, v in parse_qsl(query, keep_blank_values=True)
            if k.lower() not in TRACKING_PARAMS
        ]
        query = urlencode(filtered, doseq=True)

    return urlunparse((scheme, netloc, path, parsed.params, query, ""))


def is_internal(url: str, root_netloc: str, allowed_domains: list[str]) -> bool:
    netloc = urlparse(url).netloc.lower()
    if netloc == root_netloc:
        return True
    if allowed_domains:
        for domain in allowed_domains:
            domain = domain.strip().lower()
            if not domain:
                continue
            if netloc == domain or netloc.endswith("." + domain):
                return True
    return False


def should_filter(url: str, params: CrawlParams) -> bool:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        return True
    path_lower = parsed.path.lower()
    return any(path_lower.endswith(ext.lower()) for ext in params.static_extensions)


def _dedupe_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _hash_content(html: str | None) -> str | None:
    if not html:
        return None
    return hashlib.sha256(html.encode("utf-8")).hexdigest()


def hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()



async def _fetch_with_retries(
    adapter: Crawl4AIAdapter, url: str, params: CrawlParams
) -> FetchResult:
    last_error: str | None = None
    timeout = max(1, params.timeout)
    for attempt in range(params.retries + 1):
        try:
            result = await asyncio.wait_for(
                adapter.fetch(url, timeout=timeout),
                timeout=timeout,
            )
            if not result.success:
                last_error = result.error_message or "crawl failed"
                if attempt < params.retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                return FetchResult(
                    url=url,
                    success=False,
                    status_code=result.status_code,
                    title=result.title,
                    final_url=result.final_url,
                    links=result.links,
                    internal_links=result.internal_links,
                    external_links=result.external_links,
                    canonical_url=result.canonical_url,
                    html=result.html,
                    fit_markdown=result.fit_markdown,
                    error=last_error,
                )
            return FetchResult(
                url=url,
                success=True,
                status_code=result.status_code,
                title=result.title,
                final_url=result.final_url,
                links=result.links,
                internal_links=result.internal_links,
                external_links=result.external_links,
                canonical_url=result.canonical_url,
                html=result.html,
                fit_markdown=result.fit_markdown,
            )
        except asyncio.TimeoutError:
            last_error = f"timeout after {timeout}s"
            if attempt < params.retries:
                await asyncio.sleep(0.5 * (attempt + 1))
        except Exception as exc:  # pylint: disable=broad-except
            last_error = str(exc)
            if attempt < params.retries:
                await asyncio.sleep(0.5 * (attempt + 1))
    return FetchResult(
        url=url,
        success=False,
        status_code=None,
        title=None,
        final_url=None,
        links=[],
        internal_links=None,
        external_links=None,
        canonical_url=None,
        html=None,
        fit_markdown=None,
        error=last_error,
    )


async def crawl_batch(
    adapter: Crawl4AIAdapter, urls: list[str], params: CrawlParams
) -> list[FetchResult]:
    concurrency = max(1, params.concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async def bound_fetch(target_url: str) -> FetchResult:
        async with semaphore:
            return await _fetch_with_retries(adapter, target_url, params)

    tasks = [asyncio.create_task(bound_fetch(url)) for url in urls]
    return await asyncio.gather(*tasks)


def crawl_job(job_id: str, session_factory, settings: Settings, adapter: Crawl4AIAdapter) -> None:
    db = session_factory()
    job = db.query(CrawlJob).filter_by(job_id=job_id).first()
    if not job:
        db.close()
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def run_async(coro):
        return loop.run_until_complete(coro)

    try:
        params = CrawlParams.from_dict(job.params or {}, settings)
        root_netloc = urlparse(job.root_url).netloc.lower()

        logger.info(
            "crawl start job_id=%s root_url=%s max_depth=%s max_pages=%s concurrency=%s",
            job_id,
            job.root_url,
            params.max_depth,
            params.max_pages,
            params.concurrency,
        )

        db.query(SitePage).filter_by(job_id=job_id, crawl_status="PROCESSING").update(
            {"crawl_status": "PENDING"}
        )
        db.commit()

        seen_hashes = {
            url_hash
            for (url_hash,) in db.query(SitePage.url_hash)
            .filter_by(job_id=job_id)
            .all()
            if url_hash
        }
        child_seen_hashes = set(seen_hashes)
        db.close()

        for depth in range(params.max_depth + 1):
            while True:
                db = session_factory()
                job = db.query(CrawlJob).filter_by(job_id=job_id).first()
                if not job:
                    db.close()
                    return
                if job.status == "CANCELLED":
                    job.finished_at = datetime.utcnow()
                    job.updated_at = datetime.utcnow()
                    db.commit()
                    db.close()
                    logger.info("crawl cancelled job_id=%s", job_id)
                    return

                remaining_to_process = params.max_pages - (job.crawled_count + job.failed_count)
                if remaining_to_process <= 0:
                    logger.info(
                        "crawl limit reached job_id=%s crawled=%s failed=%s discovered=%s",
                        job_id,
                        job.crawled_count,
                        job.failed_count,
                        job.discovered_count,
                    )
                    db.close()
                    return

                batch_size = min(params.concurrency, remaining_to_process)
                pages = (
                    db.query(SitePage)
                    .filter_by(job_id=job_id, depth=depth, crawl_status="PENDING")
                    .limit(batch_size)
                    .all()
                )
                if not pages:
                    db.close()
                    logger.info("no pending pages job_id=%s depth=%s", job_id, depth)
                    break

                for page in pages:
                    page.crawl_status = "PROCESSING"
                db.commit()
                logger.info("crawl batch job_id=%s depth=%s size=%s", job_id, depth, len(pages))
                urls = [page.url for page in pages]
                db.close()

                results = run_async(crawl_batch(adapter, urls, params))

                db = session_factory()
                job = db.query(CrawlJob).filter_by(job_id=job_id).first()
                if not job:
                    db.close()
                    return

                success_count = 0
                failed_count = 0
                for result in results:
                    page = (
                        db.query(SitePage)
                        .filter_by(job_id=job_id, url_hash=hash_url(result.url))
                        .first()
                    )
                    if not page:
                        continue

                    page.last_crawled = datetime.utcnow()
                    page.crawled = True
                    page.status_code = result.status_code
                    page.title = result.title
                    page.canonical_url = result.canonical_url
                    page.content_hash = _hash_content(result.html)
                    page.fit_markdown = result.fit_markdown

                    if result.success:
                        page.crawl_status = "CRAWLED"
                        success_count += 1
                    else:
                        page.crawl_status = "FAILED"
                        page.error_message = result.error
                        failed_count += 1
                        logger.warning(
                            "fetch failed job_id=%s url=%s error=%s",
                            job_id,
                            result.url,
                            result.error,
                        )

                    raw_all_links = result.links
                    if not raw_all_links and (result.internal_links or result.external_links):
                        raw_all_links = (result.internal_links or []) + (
                            result.external_links or []
                        )

                    raw_internal_links = (
                        result.internal_links
                        if result.internal_links is not None
                        else raw_all_links
                    )

                    internal_links: list[str] = []
                    for link in raw_internal_links:
                        normalized = normalize_url(link, page.url, params)
                        if not normalized or should_filter(normalized, params):
                            continue
                        if not is_internal(normalized, root_netloc, params.allowed_domains):
                            continue
                        internal_links.append(normalized)

                    internal_links = _dedupe_preserve(internal_links)

                    filtered_children: list[tuple[str, str]] = []
                    for child_url in internal_links:
                        child_hash = hash_url(child_url)
                        if child_hash in child_seen_hashes:
                            continue
                        child_seen_hashes.add(child_hash)
                        filtered_children.append((child_url, child_hash))

                    page.childrens = [child_url for child_url, _ in filtered_children]

                    new_pages: list[SitePage] = []
                    if depth < params.max_depth and len(seen_hashes) < params.max_pages:
                        for child_url, child_hash in filtered_children:
                            if len(seen_hashes) >= params.max_pages:
                                break
                            seen_hashes.add(child_hash)
                            new_pages.append(
                                SitePage(
                                    job_id=job_id,
                                    url=child_url,
                                    url_hash=child_hash,
                                    childrens=[],
                                    parent_url=page.url,
                                    depth=depth + 1,
                                    crawled=False,
                                    crawl_status="PENDING",
                                )
                            )
                        if new_pages:
                            db.add_all(new_pages)

                    logger.info(
                        "page processed job_id=%s url=%s depth=%s links=%s internal=%s new=%s status=%s",
                        job_id,
                        page.url,
                        depth,
                        len(raw_all_links),
                        len(internal_links),
                        len(new_pages),
                        page.crawl_status,
                    )

                job.crawled_count += success_count
                job.failed_count += failed_count
                job.discovered_count = len(seen_hashes)
                job.queued_count = max(
                    0, job.discovered_count - job.crawled_count - job.failed_count
                )
                job.current_depth = depth
                job.updated_at = datetime.utcnow()
                db.commit()
                db.close()
    finally:
        try:
            run_async(adapter.aclose())
        except Exception:
            logger.warning("failed to close crawler", exc_info=True)
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            logger.warning("failed to shutdown async generators", exc_info=True)
        loop.close()
        asyncio.set_event_loop(None)
