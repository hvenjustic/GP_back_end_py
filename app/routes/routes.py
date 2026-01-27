from __future__ import annotations

from fastapi import APIRouter

from app.handlers import agent, api, crawl, graph
from app.schemas import (
    AgentSessionDetailResponse,
    AgentSessionDeleteResponse,
    AgentSessionResponse,
    ClearQueueResponse,
    CrawlResponse,
    GraphBuildResponse,
    GraphLocateResponse,
    GraphVisualResponse,
    ListResultsResponse,
    QueueAckResponse,
    QueueStatusResponse,
    ResultDetailResponse,
    StatusResponse,
    SubmitTasksResponse,
)

router = APIRouter()

# Crawl routes
router.add_api_route(
    "/crawl",
    crawl.create_crawl,
    methods=["POST"],
    response_model=CrawlResponse,
)
router.add_api_route(
    "/status/{job_id}",
    crawl.get_status,
    methods=["GET"],
    response_model=StatusResponse,
)
router.add_api_route(
    "/cancel/{job_id}",
    crawl.cancel_job,
    methods=["POST"],
)
router.add_api_route(
    "/tree/{job_id}",
    crawl.get_tree,
    methods=["GET"],
)

# Graph routes
router.add_api_route(
    "/graph/build",
    graph.build_graph,
    methods=["POST"],
    response_model=GraphBuildResponse,
    status_code=202,
)

# API routes (formerly Go-style endpoints)
api_group = APIRouter(prefix="/api")
api_group.add_api_route(
    "/tasks",
    api.submit_tasks,
    methods=["POST"],
    response_model=SubmitTasksResponse,
)
api_group.add_api_route(
    "/tasks/crawl",
    api.enqueue_tasks,
    methods=["POST"],
    response_model=QueueAckResponse,
)
api_group.add_api_route(
    "/tasks/status",
    api.get_task_status,
    methods=["GET"],
    response_model=QueueStatusResponse,
)
api_group.add_api_route(
    "/graph_locate",
    api.get_graph_locate,
    methods=["GET"],
    response_model=GraphLocateResponse,
)
api_group.add_api_route(
    "/queues/clear",
    api.clear_queue,
    methods=["POST"],
    response_model=ClearQueueResponse,
)
api_group.add_api_route(
    "/results",
    api.list_results,
    methods=["GET"],
    response_model=ListResultsResponse,
)
api_group.add_api_route(
    "/products",
    api.list_products,
    methods=["GET"],
    response_model=ListResultsResponse,
)
api_group.add_api_route(
    "/results/{task_id}",
    api.get_result_detail,
    methods=["GET"],
    response_model=ResultDetailResponse,
)
api_group.add_api_route(
    "/results/{task_id}/graph_view",
    api.get_graph_view,
    methods=["GET"],
    response_model=GraphVisualResponse,
)
api_group.add_api_route(
    "/results/graph",
    api.build_graph,
    methods=["POST"],
    response_model=QueueAckResponse,
)
api_group.add_api_route(
    "/results/graph/batch",
    api.build_graph_batch,
    methods=["POST"],
    response_model=QueueAckResponse,
)
api_group.add_api_route(
    "/results/preprocess/status",
    api.get_preprocess_status,
    methods=["GET"],
    response_model=QueueStatusResponse,
)
api_group.add_api_route(
    "/results/graph/status",
    api.get_graph_status,
    methods=["GET"],
    response_model=QueueStatusResponse,
)
router.include_router(api_group)

# Agent routes
chat_group = APIRouter(prefix="/api/chat")
chat_group.add_api_route(
    "/agent/stream",
    agent.stream_agent,
    methods=["GET"],
)
router.include_router(chat_group)

agent_group = APIRouter(prefix="/api/agent")
agent_group.add_api_route(
    "/sessions",
    agent.create_session,
    methods=["POST"],
    response_model=AgentSessionResponse,
)
agent_group.add_api_route(
    "/sessions",
    agent.list_sessions,
    methods=["GET"],
    response_model=list[AgentSessionResponse],
)
agent_group.add_api_route(
    "/sessions/{session_id}",
    agent.get_session,
    methods=["GET"],
    response_model=AgentSessionDetailResponse,
)
agent_group.add_api_route(
    "/sessions/{session_id}",
    agent.delete_session,
    methods=["DELETE"],
    response_model=AgentSessionDeleteResponse,
)
router.include_router(agent_group)
