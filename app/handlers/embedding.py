"""
图嵌入API处理器

提供API端点来：
1. 异步计算图谱的高维嵌入和3D坐标
2. 获取已计算的嵌入数据
3. 获取嵌入计算状态和计时信息
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db, SessionLocal
from app.schemas import (
    EmbeddingComputeRequest,
    EmbeddingComputeResponse,
    EmbeddingStatusResponse,
    EmbeddingResultItem,
    EmbeddingListResponse,
    EmbeddingCoord3DResponse,
)
from app.services import embedding_service

logger = logging.getLogger(__name__)

# 用于跟踪后台任务状态
_embedding_task_status = {
    "is_running": False,
    "progress": 0,
    "total": 0,
    "message": "",
    "error": None,
    "results": [],  # 存储计算结果
    "total_duration_ms": 0,
}


def _reset_task_status():
    """重置任务状态"""
    global _embedding_task_status
    _embedding_task_status = {
        "is_running": False,
        "progress": 0,
        "total": 0,
        "message": "",
        "error": None,
        "results": [],
        "total_duration_ms": 0,
    }


def _run_embedding_task(
    task_ids: list[int],
    embedding_method: str,
    reduction_method: str,
    embedding_dim: int,
    use_gpu: bool,
):
    """后台运行嵌入计算任务"""
    global _embedding_task_status
    settings = get_settings()
    
    # 为后台任务创建独立的数据库会话
    db = SessionLocal()
    
    try:
        _embedding_task_status["is_running"] = True
        _embedding_task_status["message"] = "正在获取图谱数据..."
        _embedding_task_status["error"] = None
        _embedding_task_status["total"] = len(task_ids)
        
        # 计算嵌入
        _embedding_task_status["message"] = f"正在使用 {embedding_method} 计算嵌入..."
        
        results = embedding_service.compute_embeddings_for_tasks(
            settings=settings,
            db=db,
            task_ids=task_ids,
            embedding_method=embedding_method,
            reduction_method=reduction_method,
            embedding_dim=embedding_dim,
            use_gpu=use_gpu,
            save_to_db=True,
            save_to_neo4j=True,
        )
        
        # 计算总耗时
        total_duration = sum(r.duration_ms for r in results)
        
        # 存储结果摘要
        result_items = []
        for r in results:
            result_items.append({
                "task_id": r.site_id,
                "name": r.site_name,
                "node_count": r.node_count,
                "edge_count": r.edge_count,
                "embedding_dim": len(r.high_dim_embedding),
                "duration_ms": r.duration_ms,
            })
        
        _embedding_task_status["progress"] = len(results)
        _embedding_task_status["results"] = result_items
        _embedding_task_status["total_duration_ms"] = total_duration
        _embedding_task_status["message"] = f"完成！处理了 {len(results)} 个任务，总耗时 {total_duration}ms"
        
    except Exception as e:
        logger.exception("Embedding task failed")
        _embedding_task_status["error"] = str(e)
        _embedding_task_status["message"] = f"任务失败: {str(e)}"
    finally:
        db.close()
        _embedding_task_status["is_running"] = False


async def compute_embeddings(
    request: EmbeddingComputeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> EmbeddingComputeResponse:
    """
    异步计算图谱嵌入向量和3D坐标
    
    POST /api/embeddings/compute
    
    该接口会在后台异步计算嵌入，可以通过 GET /api/embeddings/status 查询进度和结果
    """
    # 检查是否有任务正在运行
    if _embedding_task_status["is_running"]:
        raise HTTPException(status_code=409, detail="已有嵌入计算任务正在运行")
    
    # 验证task_ids
    if not request.site_ids:
        raise HTTPException(status_code=400, detail="site_ids 不能为空")
    
    task_ids = [int(x) for x in request.site_ids if int(x) > 0]
    if not task_ids:
        raise HTTPException(status_code=400, detail="没有有效的任务ID")
    
    # 检查任务是否存在
    from app.models import SiteTask
    tasks = db.query(SiteTask).filter(SiteTask.id.in_(task_ids)).all()
    if not tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 检查哪些任务有图谱数据
    valid_ids = [int(t.id) for t in tasks if t.graph_json]
    if not valid_ids:
        raise HTTPException(status_code=400, detail="所有任务都没有图谱数据，请先构建知识图谱")
    
    # 重置状态
    _reset_task_status()
    _embedding_task_status["is_running"] = True
    _embedding_task_status["message"] = "任务已提交，正在启动..."
    _embedding_task_status["total"] = len(valid_ids)
    
    # 启动后台任务
    background_tasks.add_task(
        _run_embedding_task,
        task_ids=valid_ids,
        embedding_method=request.embedding_method,
        reduction_method=request.reduction_method,
        embedding_dim=request.embedding_dim,
        use_gpu=request.use_gpu,
    )
    
    return EmbeddingComputeResponse(
        status="accepted",
        message=f"嵌入计算任务已提交（{len(valid_ids)}个任务），请通过 /api/embeddings/status 查询进度",
    )


async def get_embedding_status() -> EmbeddingStatusResponse:
    """
    获取嵌入计算任务状态和计时信息
    
    GET /api/embeddings/status
    """
    return EmbeddingStatusResponse(
        is_running=_embedding_task_status["is_running"],
        progress=_embedding_task_status["progress"],
        total=_embedding_task_status["total"],
        message=_embedding_task_status["message"],
        error=_embedding_task_status["error"],
        results=_embedding_task_status.get("results"),
        total_duration_ms=_embedding_task_status.get("total_duration_ms"),
    )


async def list_embeddings(
    site_ids: Optional[str] = None,
    db: Session = Depends(get_db),
) -> EmbeddingListResponse:
    """
    获取已计算的嵌入数据列表（从MySQL读取）
    
    GET /api/embeddings
    
    Args:
        site_ids: 可选，逗号分隔的site_id列表
    """
    # 解析site_ids参数
    parsed_site_ids = None
    if site_ids:
        try:
            parsed_site_ids = [int(x.strip()) for x in site_ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="site_ids格式错误，应为逗号分隔的整数")
    
    try:
        # 从MySQL获取
        data = embedding_service.get_embeddings_from_mysql(db, parsed_site_ids)
        
        items = [
            EmbeddingResultItem(
                site_id=d["site_id"],
                site_name=d.get("site_name") or "",
                site_url=d.get("site_url") or "",
                embedding=list(d["embedding"]) if d.get("embedding") else None,
                coord_3d=list(d["coord_3d"]) if d.get("coord_3d") else None,
            )
            for d in data
        ]
        
        return EmbeddingListResponse(
            items=items,
            total=len(items),
        )
        
    except Exception as e:
        logger.exception("Failed to fetch embeddings")
        raise HTTPException(status_code=500, detail=f"获取嵌入数据失败: {str(e)}")


async def get_3d_coordinates(
    site_ids: Optional[str] = None,
    db: Session = Depends(get_db),
) -> EmbeddingCoord3DResponse:
    """
    获取3D坐标数据（用于可视化，从MySQL读取）
    
    GET /api/embeddings/coords3d
    
    Args:
        site_ids: 可选，逗号分隔的site_id列表
    """
    # 解析site_ids参数
    parsed_site_ids = None
    if site_ids:
        try:
            parsed_site_ids = [int(x.strip()) for x in site_ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="site_ids格式错误，应为逗号分隔的整数")
    
    try:
        # 从MySQL获取
        data = embedding_service.get_embeddings_from_mysql(db, parsed_site_ids)
        
        coords = []
        for d in data:
            coord_3d = d.get("coord_3d")
            if coord_3d and len(coord_3d) >= 3:
                coords.append({
                    "site_id": d["site_id"],
                    "site_name": d.get("site_name") or "",
                    "site_url": d.get("site_url") or "",
                    "x": float(coord_3d[0]),
                    "y": float(coord_3d[1]),
                    "z": float(coord_3d[2]),
                })
        
        return EmbeddingCoord3DResponse(
            items=coords,
            total=len(coords),
        )
        
    except Exception as e:
        logger.exception("Failed to fetch 3D coordinates")
        raise HTTPException(status_code=500, detail=f"获取3D坐标失败: {str(e)}")


async def get_embedding_task_status(
    task_ids: Optional[str] = None,
    db: Session = Depends(get_db),
) -> dict:
    """
    获取指定任务的嵌入状态和计时信息
    
    GET /api/embeddings/tasks/status
    
    Args:
        task_ids: 可选，逗号分隔的task_id列表
    """
    parsed_ids = None
    if task_ids:
        try:
            parsed_ids = [int(x.strip()) for x in task_ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="task_ids格式错误，应为逗号分隔的整数")
    
    try:
        status_list = embedding_service.get_embedding_status_for_tasks(db, parsed_ids)
        
        # 统计信息
        from app.models import SiteTask
        total_with_embedding = db.query(SiteTask).filter(SiteTask.embedding.isnot(None)).count()
        total_with_graph = db.query(SiteTask).filter(SiteTask.graph_json.isnot(None)).count()
        total_buildable = db.query(SiteTask).filter(
            SiteTask.graph_json.isnot(None),
            SiteTask.embedding.is_(None)
        ).count()
        
        return {
            "summary": {
                "total_with_embedding": total_with_embedding,
                "total_with_graph": total_with_graph,
                "total_buildable": total_buildable,
            },
            "count": len(status_list),
            "items": status_list,
        }
        
    except Exception as e:
        logger.exception("Failed to fetch embedding status")
        raise HTTPException(status_code=500, detail=f"获取嵌入状态失败: {str(e)}")
