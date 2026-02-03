from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class CrawlRequest(BaseModel):
    root_url: str
    max_depth: int | None = None
    max_pages: int | None = None
    concurrency: int | None = None
    timeout: int | None = None
    retries: int | None = None
    strip_query: bool | None = None
    strip_tracking_params: bool | None = None


class CrawlResponse(BaseModel):
    job_id: str
    status_url: str


class StatusResponse(BaseModel):
    job_id: str
    root_url: str
    status: str
    progress: dict[str, Any]
    timestamps: dict[str, Any]
    params: dict[str, Any]
    message: str | None = None


class GraphBuildRequest(BaseModel):
    task_id: int


class GraphBuildResponse(BaseModel):
    task_id: int
    status: str
    celery_task_id: str | None = None


class SubmitTaskItem(BaseModel):
    url: str
    name: str | None = None
    site_name: str | None = None


class SubmitTasksRequest(BaseModel):
    urls: list[SubmitTaskItem]


class SubmitTasksResponse(BaseModel):
    created: int


class EnqueueTasksRequest(BaseModel):
    ids: list[int]


class GraphBatchRequest(BaseModel):
    ids: list[int]


class QueueStatusResponse(BaseModel):
    pending: int
    queue_key: str


class ClearQueueRequest(BaseModel):
    queue_name: str


class ClearQueueResponse(BaseModel):
    queue_name: str
    removed_keys: int


class IDRequest(BaseModel):
    id: int


class QueueAckResponse(BaseModel):
    queued: int
    queue_key: str
    pending: int


class CrawlJobMeta(BaseModel):
    job_id: str
    status: str
    discovered_count: int
    queued_count: int
    crawled_count: int
    failed_count: int
    current_depth: int
    error_message: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    updated_at: datetime | None = None
    finished_at: datetime | None = None


class ResultItem(BaseModel):
    id: int
    name: str | None = None
    site_name: str | None = None
    url: str
    crawled_at: datetime | None = None
    llm_processed_at: datetime | None = None
    is_crawled: bool
    on_sale: bool
    crawl_count: int
    page_count: int
    graph_json: str | None = None
    crawl_duration_ms: int | None = None
    llm_duration_ms: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    crawl_job: CrawlJobMeta | None = None


class ListResultsResponse(BaseModel):
    items: list[ResultItem]
    total: int
    page: int
    page_size: int


class ProductReviewItem(BaseModel):
    id: int
    name: str | None = None
    site_name: str | None = None
    url: str
    llm_processed_at: datetime | None = None
    updated_at: datetime | None = None
    on_sale: bool


class ProductReviewResponse(BaseModel):
    items: list[ProductReviewItem]
    total: int
    page: int
    page_size: int


class ProductOnSaleRequest(BaseModel):
    ids: list[int]


class ProductOnSaleResponse(BaseModel):
    updated: int
    ids: list[int]


class ResultDetailResponse(BaseModel):
    item: ResultItem | None = None


class GraphLocateItem(BaseModel):
    id: int
    latitude: float
    longitude: float


class GraphLocateResponse(BaseModel):
    items: list[GraphLocateItem] | None = None
    total: int | None = None


class GraphVisualNode(BaseModel):
    id: str
    name: str | None = None
    type: str | None = None
    label: str | None = None
    description: str | None = None
    extra: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    meta: dict[str, str] | None = None


class GraphVisualEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str | None = None
    label: str | None = None
    raw: dict[str, Any] | None = None
    meta: dict[str, str] | None = None


class GraphVisualResponse(BaseModel):
    nodes: list[GraphVisualNode]
    edges: list[GraphVisualEdge]


class AgentSessionCreateRequest(BaseModel):
    title: str | None = None


class AgentSessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class AgentMessageResponse(BaseModel):
    id: int
    role: str
    content: str | None = None
    status: str | None = None
    tool_name: str | None = None
    tool_payload: dict[str, Any] | None = None
    created_at: datetime | None = None


class AgentSessionDetailResponse(BaseModel):
    session: AgentSessionResponse
    messages: list[AgentMessageResponse]


class AgentSessionDeleteResponse(BaseModel):
    session_id: str
    deleted: bool


# ============ Embedding Schemas ============

class EmbeddingComputeRequest(BaseModel):
    """图嵌入计算请求"""
    embedding_method: str = "gnn"  # "gnn" or "node2vec"
    reduction_method: str = "umap"  # "umap" or "tsne"
    embedding_dim: int = 128
    use_gpu: bool = True
    save_node_embeddings: bool = False
    site_ids: list[int] | None = None  # None表示处理所有


class EmbeddingComputeResponse(BaseModel):
    """异步嵌入计算响应"""
    status: str
    message: str


class EmbeddingStatusResponse(BaseModel):
    """嵌入计算状态响应"""
    is_running: bool
    progress: int
    total: int
    message: str
    error: str | None = None


class EmbeddingResultItem(BaseModel):
    """单个嵌入结果项"""
    site_id: int
    site_name: str | None = None
    site_url: str | None = None
    embedding: list[float] | None = None
    coord_3d: list[float] | None = None
    node_count: int | None = None


class EmbeddingListResponse(BaseModel):
    """嵌入列表响应"""
    items: list[EmbeddingResultItem]
    total: int


class EmbeddingCoord3DItem(BaseModel):
    """3D坐标项"""
    site_id: int
    site_name: str | None = None
    site_url: str | None = None
    x: float
    y: float
    z: float


class EmbeddingCoord3DResponse(BaseModel):
    """3D坐标响应"""
    items: list[dict[str, Any]]
    total: int
