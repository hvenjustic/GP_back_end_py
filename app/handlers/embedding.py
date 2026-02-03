"""
图嵌入API处理器

提供API端点来：
1. 计算图谱的高维嵌入和3D坐标
2. 获取已计算的嵌入数据
3. 获取嵌入计算状态
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
from app.services.neo4j_service import neo4j_enabled

logger = logging.getLogger(__name__)

# 用于跟踪后台任务状态
_embedding_task_status = {
    "is_running": False,
    "progress": 0,
    "total": 0,
    "message": "",
    "error": None,
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
    }


def _run_embedding_task(
    embedding_method: str,
    reduction_method: str,
    embedding_dim: int,
    use_gpu: bool,
    save_node_embeddings: bool,
    site_ids: Optional[list[int]],
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
        
        # 计算嵌入
        _embedding_task_status["message"] = f"正在使用 {embedding_method} 计算嵌入..."
        
        results = embedding_service.compute_all_embeddings(
            settings=settings,
            db=db,
            embedding_method=embedding_method,
            reduction_method=reduction_method,
            embedding_dim=embedding_dim,
            use_gpu=use_gpu,
            save_to_db=True,
            save_to_neo4j=True,
            save_node_embeddings=save_node_embeddings,
            site_ids=site_ids,
        )
        
        _embedding_task_status["progress"] = len(results)
        _embedding_task_status["total"] = len(results)
        _embedding_task_status["message"] = f"完成！处理了 {len(results)} 个图谱"
        
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
) -> EmbeddingComputeResponse:
    """
    计算图谱嵌入向量和3D坐标
    
    POST /api/embeddings/compute
    
    该接口会在后台异步计算嵌入，可以通过 GET /api/embeddings/status 查询进度
    """
    settings = get_settings()
    
    # 检查Neo4j是否启用
    if not neo4j_enabled(settings):
        raise HTTPException(status_code=503, detail="Neo4j未配置或不可用")
    
    # 检查是否有任务正在运行
    if _embedding_task_status["is_running"]:
        raise HTTPException(status_code=409, detail="已有嵌入计算任务正在运行")
    
    # 重置状态
    _reset_task_status()
    _embedding_task_status["is_running"] = True
    _embedding_task_status["message"] = "任务已提交，正在启动..."
    
    # 启动后台任务
    background_tasks.add_task(
        _run_embedding_task,
        embedding_method=request.embedding_method,
        reduction_method=request.reduction_method,
        embedding_dim=request.embedding_dim,
        use_gpu=request.use_gpu,
        save_node_embeddings=request.save_node_embeddings,
        site_ids=request.site_ids,
    )
    
    return EmbeddingComputeResponse(
        status="accepted",
        message="嵌入计算任务已提交，请通过 /api/embeddings/status 查询进度",
    )


async def compute_embeddings_sync(
    request: EmbeddingComputeRequest,
    db: Session = Depends(get_db),
) -> EmbeddingListResponse:
    """
    同步计算图谱嵌入向量和3D坐标（等待完成后返回结果）
    
    POST /api/embeddings/compute/sync
    
    注意：对于大型图谱，此操作可能需要较长时间
    """
    settings = get_settings()
    
    # 检查Neo4j是否启用
    if not neo4j_enabled(settings):
        raise HTTPException(status_code=503, detail="Neo4j未配置或不可用")
    
    try:
        results = embedding_service.compute_all_embeddings(
            settings=settings,
            db=db,
            embedding_method=request.embedding_method,
            reduction_method=request.reduction_method,
            embedding_dim=request.embedding_dim,
            use_gpu=request.use_gpu,
            save_to_db=True,
            save_to_neo4j=True,
            save_node_embeddings=request.save_node_embeddings,
            site_ids=request.site_ids,
        )
        
        items = [
            EmbeddingResultItem(
                site_id=r.site_id,
                site_name=r.site_name,
                site_url=r.site_url,
                embedding=r.high_dim_embedding,
                coord_3d=r.coord_3d,
                node_count=len(r.node_embeddings) if r.node_embeddings else 0,
            )
            for r in results
        ]
        
        return EmbeddingListResponse(
            items=items,
            total=len(items),
        )
        
    except Exception as e:
        logger.exception("Embedding computation failed")
        raise HTTPException(status_code=500, detail=f"嵌入计算失败: {str(e)}")


async def get_embedding_status() -> EmbeddingStatusResponse:
    """
    获取嵌入计算任务状态
    
    GET /api/embeddings/status
    """
    return EmbeddingStatusResponse(
        is_running=_embedding_task_status["is_running"],
        progress=_embedding_task_status["progress"],
        total=_embedding_task_status["total"],
        message=_embedding_task_status["message"],
        error=_embedding_task_status["error"],
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
        # 优先从MySQL获取
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

