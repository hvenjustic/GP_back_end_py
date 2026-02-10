"""
图嵌入服务：将知识图谱转换为高维向量并降维到3D坐标

支持功能：
1. 从MySQL的graph_json字段或Neo4j获取图谱数据
2. 使用Node2Vec/Graph2Vec进行图嵌入
3. 使用PyTorch MPS后端(Apple M系列芯片)进行GPU加速
4. 使用UMAP进行降维到3D
5. 将嵌入向量和3D坐标存储到MySQL(site_tasks表)和Neo4j
6. 支持计时和异步处理
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib

import numpy as np
from neo4j import GraphDatabase
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import SiteTask

logger = logging.getLogger(__name__)


@dataclass
class GraphData:
    """图数据结构"""
    site_id: int
    site_name: str
    site_url: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    node_id_map: dict[str, int]  # name -> index
    

@dataclass
class EmbeddingResult:
    """嵌入结果"""
    site_id: int
    site_name: str
    site_url: str
    high_dim_embedding: list[float]
    coord_3d: list[float]
    duration_ms: int = 0
    node_count: int = 0
    edge_count: int = 0
    node_embeddings: Optional[dict[str, list[float]]] = None
    node_coords_3d: Optional[dict[str, list[float]]] = None


def _get_device():
    """获取最佳可用设备（MPS/CUDA/CPU）"""
    try:
        import torch
        if torch.backends.mps.is_available():
            logger.info("Using MPS (Metal Performance Shaders) backend for Apple M-series GPU")
            return torch.device("mps")
        elif torch.cuda.is_available():
            logger.info("Using CUDA backend for GPU")
            return torch.device("cuda")
        else:
            logger.info("Using CPU backend")
            return torch.device("cpu")
    except ImportError:
        logger.warning("PyTorch not available, falling back to CPU-only computation")
        return None


def _get_neo4j_driver(settings: Settings):
    """获取Neo4j驱动"""
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def _get_database(settings: Settings) -> str | None:
    """获取数据库名称"""
    value = (settings.neo4j_database or "").strip()
    return value or None


def fetch_graph_from_mysql(db: Session, task_id: int) -> Optional[GraphData]:
    """
    从MySQL的site_tasks表的graph_json字段获取图谱数据
    
    Args:
        db: 数据库会话
        task_id: 任务ID
        
    Returns:
        GraphData | None: 图谱数据，如果不存在返回None
    """
    task = db.query(SiteTask).filter(SiteTask.id == task_id).first()
    if not task:
        logger.warning(f"SiteTask {task_id} not found")
        return None
    
    if not task.graph_json:
        logger.warning(f"SiteTask {task_id} has no graph_json")
        return None
    
    try:
        graph_data = json.loads(task.graph_json) if isinstance(task.graph_json, str) else task.graph_json
    except json.JSONDecodeError:
        logger.error(f"Failed to parse graph_json for task {task_id}")
        return None
    
    # 解析entities和relations
    entities = graph_data.get("entities", [])
    relations = graph_data.get("relations", [])
    
    if not entities:
        logger.warning(f"SiteTask {task_id} has no entities in graph_json")
        return None
    
    # 构建nodes
    nodes = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        name = str(ent.get("name") or "").strip()
        if not name:
            continue
        nodes.append({
            "name": name,
            "type": str(ent.get("type") or ent.get("type_level_2") or ent.get("type_level_1") or "unknown").strip(),
            "description": str(ent.get("description") or "").strip(),
            "type_level_1": str(ent.get("type_level_1") or "").strip(),
            "type_level_2": str(ent.get("type_level_2") or "").strip(),
        })
    
    # 构建node_id_map
    node_id_map = {node["name"]: idx for idx, node in enumerate(nodes)}
    
    # 构建edges
    edges = []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        source = str(rel.get("source") or "").strip()
        target = str(rel.get("target") or "").strip()
        if source and target and source in node_id_map and target in node_id_map:
            edges.append({
                "source": source,
                "target": target,
                "type": str(rel.get("type") or "RELATED_TO").strip(),
            })
    
    return GraphData(
        site_id=task.id,
        site_name=task.site_name or task.name or "",
        site_url=task.url or "",
        nodes=nodes,
        edges=edges,
        node_id_map=node_id_map,
    )


def fetch_graphs_from_mysql(db: Session, task_ids: list[int]) -> list[GraphData]:
    """
    从MySQL批量获取图谱数据
    
    Args:
        db: 数据库会话
        task_ids: 任务ID列表
        
    Returns:
        list[GraphData]: 图谱数据列表
    """
    graphs = []
    for task_id in task_ids:
        graph = fetch_graph_from_mysql(db, task_id)
        if graph:
            graphs.append(graph)
    return graphs


def fetch_all_graphs(settings: Settings) -> list[GraphData]:
    """
    从Neo4j获取所有网站的图谱数据
    
    Returns:
        list[GraphData]: 所有网站图谱数据列表
    """
    driver = _get_neo4j_driver(settings)
    graphs = []
    
    try:
        with driver.session(database=_get_database(settings)) as session:
            # 获取所有Site
            sites_result = session.run("""
                MATCH (s:Site)
                RETURN s.id AS site_id, s.name AS site_name, s.url AS site_url
            """)
            sites = [dict(record) for record in sites_result]
            
            for site in sites:
                site_id = site["site_id"]
                site_name = site.get("site_name") or ""
                site_url = site.get("site_url") or ""
                
                # 获取该Site的所有Entity节点
                nodes_result = session.run("""
                    MATCH (e:Entity {site_id: $site_id})
                    RETURN e.name AS name, e.type AS type, e.description AS description,
                           e.type_level_1 AS type_level_1, e.type_level_2 AS type_level_2
                """, site_id=site_id)
                nodes = [dict(record) for record in nodes_result]
                
                # 构建节点ID映射
                node_id_map = {node["name"]: idx for idx, node in enumerate(nodes)}
                
                # 获取该Site的所有RELATION边
                edges_result = session.run("""
                    MATCH (source:Entity {site_id: $site_id})-[r:RELATION {site_id: $site_id}]->(target:Entity {site_id: $site_id})
                    RETURN source.name AS source, target.name AS target, r.type AS type
                """, site_id=site_id)
                edges = [dict(record) for record in edges_result]
                
                if nodes:  # 只添加有节点的图
                    graphs.append(GraphData(
                        site_id=site_id,
                        site_name=site_name,
                        site_url=site_url,
                        nodes=nodes,
                        edges=edges,
                        node_id_map=node_id_map
                    ))
                    logger.info(f"Fetched graph for site {site_id}: {len(nodes)} nodes, {len(edges)} edges")
                    
    finally:
        driver.close()
    
    return graphs


def fetch_graph_by_site_id(settings: Settings, site_id: int) -> Optional[GraphData]:
    """
    从Neo4j获取指定网站的图谱数据
    
    Args:
        settings: 配置
        site_id: 网站ID
        
    Returns:
        GraphData | None: 图谱数据，如果不存在返回None
    """
    driver = _get_neo4j_driver(settings)
    
    try:
        with driver.session(database=_get_database(settings)) as session:
            # 获取Site信息
            site_result = session.run("""
                MATCH (s:Site {id: $site_id})
                RETURN s.id AS site_id, s.name AS site_name, s.url AS site_url
            """, site_id=site_id)
            site_record = site_result.single()
            
            if not site_record:
                return None
                
            site = dict(site_record)
            site_name = site.get("site_name") or ""
            site_url = site.get("site_url") or ""
            
            # 获取Entity节点
            nodes_result = session.run("""
                MATCH (e:Entity {site_id: $site_id})
                RETURN e.name AS name, e.type AS type, e.description AS description,
                       e.type_level_1 AS type_level_1, e.type_level_2 AS type_level_2
            """, site_id=site_id)
            nodes = [dict(record) for record in nodes_result]
            
            if not nodes:
                return None
                
            # 构建节点ID映射
            node_id_map = {node["name"]: idx for idx, node in enumerate(nodes)}
            
            # 获取RELATION边
            edges_result = session.run("""
                MATCH (source:Entity {site_id: $site_id})-[r:RELATION {site_id: $site_id}]->(target:Entity {site_id: $site_id})
                RETURN source.name AS source, target.name AS target, r.type AS type
            """, site_id=site_id)
            edges = [dict(record) for record in edges_result]
            
            return GraphData(
                site_id=site_id,
                site_name=site_name,
                site_url=site_url,
                nodes=nodes,
                edges=edges,
                node_id_map=node_id_map
            )
    finally:
        driver.close()


def _build_networkx_graph(graph_data: GraphData):
    """将GraphData转换为NetworkX图"""
    import networkx as nx
    
    G = nx.DiGraph()
    
    # 添加节点及其属性
    for node in graph_data.nodes:
        node_attrs = {
            "type": node.get("type") or "unknown",
            "description": node.get("description") or "",
            "type_level_1": node.get("type_level_1") or "",
            "type_level_2": node.get("type_level_2") or "",
        }
        G.add_node(node["name"], **node_attrs)
    
    # 添加边
    for edge in graph_data.edges:
        source = edge.get("source")
        target = edge.get("target")
        if source in graph_data.node_id_map and target in graph_data.node_id_map:
            G.add_edge(source, target, type=edge.get("type") or "RELATED_TO")
    
    return G


def _compute_node_features(graph_data: GraphData, embedding_dim: int = 64) -> np.ndarray:
    """
    计算节点特征矩阵
    
    使用节点类型和描述的哈希作为初始特征
    """
    num_nodes = len(graph_data.nodes)
    features = np.zeros((num_nodes, embedding_dim), dtype=np.float32)
    
    # 获取所有唯一的类型
    types = list(set(node.get("type") or "unknown" for node in graph_data.nodes))
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    for idx, node in enumerate(graph_data.nodes):
        # 类型编码 (one-hot style contribution)
        node_type = node.get("type") or "unknown"
        type_idx = type_to_idx.get(node_type, 0)
        
        # 使用类型索引设置一些特征维度
        features[idx, type_idx % embedding_dim] = 1.0
        
        # 使用描述的哈希来填充其他特征
        desc = node.get("description") or node.get("name") or ""
        if desc:
            hash_bytes = hashlib.md5(desc.encode()).digest()
            for i, b in enumerate(hash_bytes):
                if i < embedding_dim:
                    features[idx, i] += (b / 255.0 - 0.5) * 0.1
        
        # 归一化
        norm = np.linalg.norm(features[idx])
        if norm > 0:
            features[idx] /= norm
    
    return features


def compute_graph_embeddings_graph2vec(
    graphs: list[GraphData],
    dimensions: int = 128,
    wl_iterations: int = 2,
    epochs: int = 10,
    learning_rate: float = 0.025,
    min_count: int = 5,
) -> list[np.ndarray]:
    """
    使用Graph2Vec计算图级嵌入
    
    Graph2Vec通过Weisfeiler-Lehman子图特征来捕捉图的全局结构信息，
    然后使用类似Doc2Vec的方法学习图的嵌入表示。
    
    Args:
        graphs: 图数据列表
        dimensions: 嵌入维度
        wl_iterations: Weisfeiler-Lehman迭代次数，越大捕捉越高阶的结构
        epochs: 训练轮数
        learning_rate: 学习率
        min_count: 最小词频
        
    Returns:
        list[np.ndarray]: 每个图的嵌入向量列表
    """
    from karateclub import Graph2Vec
    import networkx as nx
    
    if not graphs:
        return []
    
    # 将GraphData转换为NetworkX图（Graph2Vec需要整数节点标签）
    nx_graphs = []
    for graph_data in graphs:
        G = nx.Graph()  # Graph2Vec需要无向图
        
        # 创建节点名到整数ID的映射
        node_to_int = {name: idx for idx, name in enumerate(graph_data.node_id_map.keys())}
        
        # 添加节点（使用整数ID）
        for node_name in graph_data.node_id_map.keys():
            G.add_node(node_to_int[node_name])
        
        # 添加边
        for edge in graph_data.edges:
            source = edge.get("source")
            target = edge.get("target")
            if source in node_to_int and target in node_to_int:
                G.add_edge(node_to_int[source], node_to_int[target])
        
        # 如果图为空或只有孤立节点，添加自环确保图有效
        if G.number_of_edges() == 0:
            for node in G.nodes():
                G.add_edge(node, node)
        
        nx_graphs.append(G)
    
    logger.info(f"Training Graph2Vec on {len(nx_graphs)} graphs, dimensions={dimensions}, wl_iterations={wl_iterations}")
    
    # 调整min_count以适应小图
    actual_min_count = min(min_count, max(1, len(nx_graphs) // 2))
    
    # 创建并训练Graph2Vec模型
    model = Graph2Vec(
        dimensions=dimensions,
        wl_iterations=wl_iterations,
        epochs=epochs,
        learning_rate=learning_rate,
        min_count=actual_min_count,
    )
    
    model.fit(nx_graphs)
    
    # 获取所有图的嵌入
    embeddings = model.get_embedding()
    
    logger.info(f"Graph2Vec training completed, embedding shape: {embeddings.shape}")
    
    return [embeddings[i] for i in range(len(graphs))]


def compute_single_graph_embedding_graph2vec(
    graph_data: GraphData,
    dimensions: int = 128,
    wl_iterations: int = 2,
) -> np.ndarray:
    """
    为单个图计算Graph2Vec嵌入
    
    注意：Graph2Vec设计用于多图场景，单图时效果可能不如多图一起训练。
    对于单图，会创建一些随机扰动图来辅助训练。
    
    Args:
        graph_data: 图数据
        wl_iterations: WL迭代次数
        dimensions: 嵌入维度
        
    Returns:
        np.ndarray: 图嵌入向量
    """
    import networkx as nx
    from karateclub import Graph2Vec
    import random
    
    # 构建主图
    G = nx.Graph()
    node_to_int = {name: idx for idx, name in enumerate(graph_data.node_id_map.keys())}
    
    for node_name in graph_data.node_id_map.keys():
        G.add_node(node_to_int[node_name])
    
    for edge in graph_data.edges:
        source = edge.get("source")
        target = edge.get("target")
        if source in node_to_int and target in node_to_int:
            G.add_edge(node_to_int[source], node_to_int[target])
    
    if G.number_of_edges() == 0:
        for node in G.nodes():
            G.add_edge(node, node)
    
    # 创建扰动图来辅助训练（Graph2Vec需要多个图）
    graphs = [G]
    num_augmented = min(9, max(4, G.number_of_nodes() // 2))
    
    for _ in range(num_augmented):
        G_aug = G.copy()
        # 随机添加或删除一些边
        edges = list(G_aug.edges())
        if edges and random.random() > 0.5:
            # 随机删除一条边
            edge_to_remove = random.choice(edges)
            G_aug.remove_edge(*edge_to_remove)
        if G_aug.number_of_nodes() > 1 and random.random() > 0.5:
            # 随机添加一条边
            nodes = list(G_aug.nodes())
            n1, n2 = random.sample(nodes, 2)
            G_aug.add_edge(n1, n2)
        graphs.append(G_aug)
    
    model = Graph2Vec(
        dimensions=dimensions,
        wl_iterations=wl_iterations,
        epochs=10,
        min_count=1,
    )
    model.fit(graphs)
    
    embeddings = model.get_embedding()
    return embeddings[0]  # 返回原始图的嵌入


def compute_node_embeddings_node2vec(
    graph_data: GraphData,
    dimensions: int = 128,
    walk_length: int = 30,
    num_walks: int = 200,
    p: float = 1.0,
    q: float = 1.0,
    workers: int = 4,
) -> dict[str, np.ndarray]:
    """
    使用Node2Vec计算节点嵌入
    
    Args:
        graph_data: 图数据
        dimensions: 嵌入维度
        walk_length: 随机游走长度
        num_walks: 每个节点的随机游走数
        p: 返回参数
        q: 进出参数
        workers: 工作线程数
        
    Returns:
        dict[str, np.ndarray]: 节点名称到嵌入向量的映射
    """
    from node2vec import Node2Vec
    import networkx as nx
    
    G = _build_networkx_graph(graph_data)
    
    if G.number_of_nodes() == 0:
        return {}
    
    # 对于小图，调整参数
    actual_num_walks = min(num_walks, max(10, G.number_of_nodes()))
    actual_walk_length = min(walk_length, max(5, G.number_of_nodes()))
    
    # 创建Node2Vec模型
    # 注意：node2vec需要无向图或将有向图转为无向图
    G_undirected = G.to_undirected()
    
    node2vec = Node2Vec(
        G_undirected,
        dimensions=dimensions,
        walk_length=actual_walk_length,
        num_walks=actual_num_walks,
        p=p,
        q=q,
        workers=workers,
        quiet=True
    )
    
    # 训练模型
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    # 获取节点嵌入
    embeddings = {}
    for node in G.nodes():
        try:
            embeddings[node] = model.wv[node]
        except KeyError:
            # 如果某个节点没有嵌入，使用零向量
            embeddings[node] = np.zeros(dimensions, dtype=np.float32)
    
    return embeddings


def compute_graph_embedding_gnn(
    graph_data: GraphData,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 3,
    use_gpu: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    使用图神经网络(GNN)计算图嵌入和节点嵌入
    
    支持MPS (Apple M系列芯片) GPU加速
    
    Args:
        graph_data: 图数据
        embedding_dim: 最终嵌入维度
        hidden_dim: 隐藏层维度
        num_layers: GNN层数
        use_gpu: 是否使用GPU加速
        
    Returns:
        tuple[np.ndarray, dict[str, np.ndarray]]: (图嵌入, 节点嵌入字典)
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    device = _get_device() if use_gpu else torch.device("cpu")
    if device is None:
        device = torch.device("cpu")
    
    num_nodes = len(graph_data.nodes)
    if num_nodes == 0:
        return np.zeros(embedding_dim, dtype=np.float32), {}
    
    # 计算初始节点特征
    node_features = _compute_node_features(graph_data, embedding_dim=hidden_dim)
    x = torch.tensor(node_features, dtype=torch.float32, device=device)
    
    # 构建边索引
    edge_index = []
    for edge in graph_data.edges:
        source = edge.get("source")
        target = edge.get("target")
        if source in graph_data.node_id_map and target in graph_data.node_id_map:
            src_idx = graph_data.node_id_map[source]
            tgt_idx = graph_data.node_id_map[target]
            edge_index.append([src_idx, tgt_idx])
            # 添加反向边使图变为无向
            edge_index.append([tgt_idx, src_idx])
    
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
    else:
        # 如果没有边，创建自环
        edge_index = torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long, device=device).t().contiguous()
    
    # 简单的消息传递GNN
    class SimpleGNN(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, out_dim))
            
        def forward(self, x, edge_index):
            # 简单的聚合：平均邻居特征
            for i, layer in enumerate(self.layers):
                # 消息传递
                if edge_index.shape[1] > 0:
                    row, col = edge_index
                    # 聚合邻居特征
                    neighbor_sum = torch.zeros_like(x)
                    neighbor_count = torch.zeros(x.shape[0], 1, device=x.device)
                    neighbor_sum.index_add_(0, row, x[col])
                    neighbor_count.index_add_(0, row, torch.ones(col.shape[0], 1, device=x.device))
                    neighbor_count = neighbor_count.clamp(min=1)
                    x = (x + neighbor_sum / neighbor_count) / 2
                
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = F.relu(x)
            return x
    
    model = SimpleGNN(hidden_dim, hidden_dim, embedding_dim, num_layers).to(device)
    
    # 前向传播获取节点嵌入
    with torch.no_grad():
        node_embeddings_tensor = model(x, edge_index)
    
    # 将节点嵌入转换为CPU numpy数组
    node_embeddings_np = node_embeddings_tensor.cpu().numpy()
    
    # 计算图级嵌入（使用均值池化）
    graph_embedding = np.mean(node_embeddings_np, axis=0)
    
    # 构建节点嵌入字典
    node_embeddings = {}
    for node_name, node_idx in graph_data.node_id_map.items():
        node_embeddings[node_name] = node_embeddings_np[node_idx]
    
    return graph_embedding, node_embeddings


def compute_graph_embedding_aggregated(
    node_embeddings: dict[str, np.ndarray],
    method: str = "mean"
) -> np.ndarray:
    """
    通过聚合节点嵌入来计算图级嵌入
    
    Args:
        node_embeddings: 节点嵌入字典
        method: 聚合方法 ("mean", "sum", "max")
        
    Returns:
        np.ndarray: 图级嵌入向量
    """
    if not node_embeddings:
        return np.zeros(128, dtype=np.float32)
    
    embeddings_array = np.array(list(node_embeddings.values()))
    
    if method == "mean":
        return np.mean(embeddings_array, axis=0)
    elif method == "sum":
        return np.sum(embeddings_array, axis=0)
    elif method == "max":
        return np.max(embeddings_array, axis=0)
    else:
        return np.mean(embeddings_array, axis=0)


def reduce_to_3d_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    使用UMAP将高维嵌入降维到3D
    
    Args:
        embeddings: 高维嵌入矩阵 (n_samples, n_features)
        n_neighbors: UMAP邻居数
        min_dist: 最小距离参数
        metric: 距离度量
        random_state: 随机种子
        
    Returns:
        np.ndarray: 3D坐标 (n_samples, 3)
    """
    import umap
    
    n_samples = embeddings.shape[0]
    
    if n_samples < 2:
        # 如果只有一个样本，返回原点
        return np.zeros((n_samples, 3), dtype=np.float32)
    
    # 调整n_neighbors以适应样本数
    actual_n_neighbors = min(n_neighbors, n_samples - 1)
    actual_n_neighbors = max(2, actual_n_neighbors)
    
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=actual_n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    
    coords_3d = reducer.fit_transform(embeddings)
    return coords_3d.astype(np.float32)


def reduce_to_3d_tsne(
    embeddings: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    使用t-SNE将高维嵌入降维到3D
    
    Args:
        embeddings: 高维嵌入矩阵 (n_samples, n_features)
        perplexity: t-SNE困惑度
        random_state: 随机种子
        
    Returns:
        np.ndarray: 3D坐标 (n_samples, 3)
    """
    from sklearn.manifold import TSNE
    
    n_samples = embeddings.shape[0]
    
    if n_samples < 2:
        return np.zeros((n_samples, 3), dtype=np.float32)
    
    # 调整perplexity以适应样本数
    actual_perplexity = min(perplexity, (n_samples - 1) / 3)
    actual_perplexity = max(5.0, actual_perplexity)
    
    reducer = TSNE(
        n_components=3,
        perplexity=actual_perplexity,
        random_state=random_state,
        n_iter=1000,
    )
    
    coords_3d = reducer.fit_transform(embeddings)
    return coords_3d.astype(np.float32)


def save_embeddings_to_mysql(
    db: Session,
    results: list[EmbeddingResult],
) -> None:
    """
    将嵌入向量和3D坐标保存到MySQL的site_tasks表
    
    Args:
        db: 数据库会话
        results: 嵌入结果列表
    """
    for result in results:
        site_task = db.query(SiteTask).filter(SiteTask.id == result.site_id).first()
        if site_task:
            site_task.embedding = result.high_dim_embedding
            site_task.coord_3d = result.coord_3d
            site_task.embedding_duration_ms = result.duration_ms
            site_task.embedding_updated_at = datetime.utcnow()
            site_task.updated_at = datetime.utcnow()
            logger.info(f"Saved embedding to MySQL for site_task {result.site_id}, duration={result.duration_ms}ms")
        else:
            logger.warning(f"SiteTask {result.site_id} not found in MySQL")
    
    db.commit()


def save_embeddings_to_neo4j(
    settings: Settings,
    results: list[EmbeddingResult],
    save_node_embeddings: bool = False,
) -> None:
    """
    将嵌入向量和3D坐标保存到Neo4j
    
    Args:
        settings: 配置
        results: 嵌入结果列表
        save_node_embeddings: 是否保存节点级嵌入
    """
    driver = _get_neo4j_driver(settings)
    
    try:
        with driver.session(database=_get_database(settings)) as session:
            for result in results:
                # 保存Site级嵌入和3D坐标
                session.run("""
                    MATCH (s:Site {id: $site_id})
                    SET s.embedding = $embedding,
                        s.coord_3d = $coord_3d,
                        s.embedding_updated_at = timestamp()
                """, 
                    site_id=result.site_id,
                    embedding=result.high_dim_embedding,
                    coord_3d=result.coord_3d,
                )
                logger.info(f"Saved embedding to Neo4j for site {result.site_id}")
                
                # 可选：保存节点级嵌入
                if save_node_embeddings and result.node_embeddings:
                    for node_name, node_embedding in result.node_embeddings.items():
                        node_coord = result.node_coords_3d.get(node_name, [0.0, 0.0, 0.0]) if result.node_coords_3d else [0.0, 0.0, 0.0]
                        session.run("""
                            MATCH (e:Entity {site_id: $site_id, name: $name})
                            SET e.embedding = $embedding,
                                e.coord_3d = $coord_3d,
                                e.embedding_updated_at = timestamp()
                        """,
                            site_id=result.site_id,
                            name=node_name,
                            embedding=node_embedding,
                            coord_3d=node_coord,
                        )
                        
    finally:
        driver.close()


def compute_all_embeddings(
    settings: Settings,
    db: Optional[Session] = None,
    embedding_method: str = "gnn",  # "gnn" or "node2vec"
    reduction_method: str = "umap",  # "umap" or "tsne"
    embedding_dim: int = 128,
    use_gpu: bool = True,
    save_to_db: bool = True,
    save_to_neo4j: bool = True,
    save_node_embeddings: bool = False,
    site_ids: Optional[list[int]] = None,
) -> list[EmbeddingResult]:
    """
    计算所有图谱的嵌入向量和3D坐标
    
    Args:
        settings: 配置
        db: 数据库会话（用于保存到MySQL）
        embedding_method: 嵌入方法 ("gnn" 或 "node2vec")
        reduction_method: 降维方法 ("umap" 或 "tsne")
        embedding_dim: 嵌入维度
        use_gpu: 是否使用GPU加速
        save_to_db: 是否保存到MySQL
        save_to_neo4j: 是否保存到Neo4j
        save_node_embeddings: 是否保存节点级嵌入
        site_ids: 要处理的site_id列表，None表示处理所有
        
    Returns:
        list[EmbeddingResult]: 嵌入结果列表
    """
    # 获取图数据
    if site_ids:
        graphs = []
        for site_id in site_ids:
            graph = fetch_graph_by_site_id(settings, site_id)
            if graph:
                graphs.append(graph)
    else:
        graphs = fetch_all_graphs(settings)
    
    if not graphs:
        logger.warning("No graphs found to process")
        return []
    
    logger.info(f"Processing {len(graphs)} graphs with {embedding_method} embedding and {reduction_method} reduction")
    
    results = []
    all_graph_embeddings = []
    all_node_embeddings_list = []
    
    # 计算每个图的嵌入
    for graph in graphs:
        logger.info(f"Computing embeddings for site {graph.site_id}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        if embedding_method == "gnn":
            graph_embedding, node_embeddings = compute_graph_embedding_gnn(
                graph,
                embedding_dim=embedding_dim,
                use_gpu=use_gpu,
            )
        elif embedding_method == "node2vec":
            node_embeddings = compute_node_embeddings_node2vec(
                graph,
                dimensions=embedding_dim,
            )
            graph_embedding = compute_graph_embedding_aggregated(node_embeddings, method="mean")
        else:
            raise ValueError(f"Unknown embedding method: {embedding_method}")
        
        all_graph_embeddings.append(graph_embedding)
        all_node_embeddings_list.append(node_embeddings)
    
    # 将图嵌入堆叠成矩阵
    graph_embeddings_matrix = np.array(all_graph_embeddings)
    
    # 降维到3D
    logger.info(f"Reducing {len(graphs)} graph embeddings to 3D using {reduction_method}")
    if reduction_method == "umap":
        graph_coords_3d = reduce_to_3d_umap(graph_embeddings_matrix)
    elif reduction_method == "tsne":
        graph_coords_3d = reduce_to_3d_tsne(graph_embeddings_matrix)
    else:
        raise ValueError(f"Unknown reduction method: {reduction_method}")
    
    # 如果需要，也对节点嵌入进行降维
    for i, graph in enumerate(graphs):
        node_embeddings = all_node_embeddings_list[i]
        node_coords_3d = None
        
        if save_node_embeddings and node_embeddings:
            node_names = list(node_embeddings.keys())
            node_embeddings_matrix = np.array([node_embeddings[name] for name in node_names])
            
            if len(node_names) >= 2:
                if reduction_method == "umap":
                    node_coords_matrix = reduce_to_3d_umap(node_embeddings_matrix)
                else:
                    node_coords_matrix = reduce_to_3d_tsne(node_embeddings_matrix)
                    
                node_coords_3d = {name: node_coords_matrix[j].tolist() for j, name in enumerate(node_names)}
            else:
                node_coords_3d = {name: [0.0, 0.0, 0.0] for name in node_names}
        
        result = EmbeddingResult(
            site_id=graph.site_id,
            site_name=graph.site_name,
            site_url=graph.site_url,
            high_dim_embedding=all_graph_embeddings[i].tolist(),
            coord_3d=graph_coords_3d[i].tolist(),
            node_embeddings={k: v.tolist() for k, v in node_embeddings.items()} if node_embeddings else None,
            node_coords_3d=node_coords_3d,
        )
        results.append(result)
    
    # 保存到MySQL
    if save_to_db and db:
        logger.info("Saving embeddings to MySQL (site_tasks table)")
        save_embeddings_to_mysql(db, results)
    
    # 保存到Neo4j
    if save_to_neo4j:
        logger.info("Saving embeddings to Neo4j")
        save_embeddings_to_neo4j(settings, results, save_node_embeddings=save_node_embeddings)
    
    logger.info(f"Completed embedding computation for {len(results)} graphs")
    return results


def get_embeddings_from_mysql(
    db: Session,
    site_ids: Optional[list[int]] = None,
) -> list[dict[str, Any]]:
    """
    从MySQL的site_tasks表获取已保存的嵌入向量和3D坐标
    
    Args:
        db: 数据库会话
        site_ids: 要获取的site_id列表，None表示获取所有
        
    Returns:
        list[dict]: 嵌入数据列表
    """
    query = db.query(SiteTask).filter(SiteTask.embedding.isnot(None))
    
    if site_ids:
        query = query.filter(SiteTask.id.in_(site_ids))
    
    results = []
    for task in query.all():
        results.append({
            "site_id": task.id,
            "site_name": task.site_name,
            "site_url": task.url,
            "embedding": task.embedding,
            "coord_3d": task.coord_3d,
            "updated_at": task.embedding_updated_at,
        })
    
    return results


def get_embeddings_from_neo4j(
    settings: Settings,
    site_ids: Optional[list[int]] = None,
) -> list[dict[str, Any]]:
    """
    从Neo4j获取已保存的嵌入向量和3D坐标
    
    Args:
        settings: 配置
        site_ids: 要获取的site_id列表，None表示获取所有
        
    Returns:
        list[dict]: 嵌入数据列表
    """
    driver = _get_neo4j_driver(settings)
    
    try:
        with driver.session(database=_get_database(settings)) as session:
            if site_ids:
                result = session.run("""
                    MATCH (s:Site)
                    WHERE s.id IN $site_ids AND s.embedding IS NOT NULL
                    RETURN s.id AS site_id, s.name AS site_name, s.url AS site_url,
                           s.embedding AS embedding, s.coord_3d AS coord_3d,
                           s.embedding_updated_at AS updated_at
                """, site_ids=site_ids)
            else:
                result = session.run("""
                    MATCH (s:Site)
                    WHERE s.embedding IS NOT NULL
                    RETURN s.id AS site_id, s.name AS site_name, s.url AS site_url,
                           s.embedding AS embedding, s.coord_3d AS coord_3d,
                           s.embedding_updated_at AS updated_at
                """)
            
            return [dict(record) for record in result]
    finally:
        driver.close()


def compute_embeddings_for_tasks(
    settings: Settings,
    db: Session,
    task_ids: list[int],
    embedding_method: str = "graph2vec",
    reduction_method: str = "umap",
    embedding_dim: int = 128,
    use_gpu: bool = True,
    save_to_db: bool = True,
    save_to_neo4j: bool = True,
) -> list[EmbeddingResult]:
    """
    为指定的任务计算嵌入向量和3D坐标（从MySQL的graph_json获取数据）
    
    这是主要的接口函数，用于处理指定任务的嵌入计算。
    
    Args:
        settings: 配置
        db: 数据库会话
        task_ids: 任务ID列表
        embedding_method: 嵌入方法 ("graph2vec"[默认], "gnn" 或 "node2vec")
        reduction_method: 降维方法 ("umap" 或 "tsne")
        embedding_dim: 嵌入维度
        use_gpu: 是否使用GPU加速（仅对gnn方法有效）
        save_to_db: 是否保存到MySQL
        save_to_neo4j: 是否保存到Neo4j
        
    Returns:
        list[EmbeddingResult]: 嵌入结果列表（包含计时信息）
    """
    if not task_ids:
        logger.warning("No task_ids provided")
        return []
    
    # 从MySQL获取图数据
    graphs = fetch_graphs_from_mysql(db, task_ids)
    
    if not graphs:
        logger.warning(f"No valid graphs found for task_ids: {task_ids}")
        return []
    
    logger.info(f"Processing {len(graphs)} graphs with {embedding_method} embedding and {reduction_method} reduction")
    
    results = []
    all_graph_embeddings = []
    all_node_embeddings_list = []
    
    # 计算嵌入
    start_time = time.time()
    
    if embedding_method == "graph2vec":
        # Graph2Vec: 一次性处理所有图，捕捉全局结构
        logger.info(f"Using Graph2Vec for {len(graphs)} graphs")
        
        if len(graphs) == 1:
            # 单个图使用特殊处理
            embedding = compute_single_graph_embedding_graph2vec(
                graphs[0],
                dimensions=embedding_dim,
            )
            all_graph_embeddings = [embedding]
        else:
            # 多个图一起训练，效果更好
            all_graph_embeddings = compute_graph_embeddings_graph2vec(
                graphs,
                dimensions=embedding_dim,
            )
        
        # Graph2Vec不产生节点级嵌入
        all_node_embeddings_list = [None] * len(graphs)
        
    elif embedding_method == "gnn":
        # GNN: 逐个处理每个图
        for graph in graphs:
            logger.info(f"Computing GNN embeddings for task {graph.site_id}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
            graph_embedding, node_embeddings = compute_graph_embedding_gnn(
                graph,
                embedding_dim=embedding_dim,
                use_gpu=use_gpu,
            )
            all_graph_embeddings.append(graph_embedding)
            all_node_embeddings_list.append(node_embeddings)
            
    elif embedding_method == "node2vec":
        # Node2Vec: 通过聚合节点嵌入得到图嵌入
        for graph in graphs:
            logger.info(f"Computing Node2Vec embeddings for task {graph.site_id}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
            node_embeddings = compute_node_embeddings_node2vec(
                graph,
                dimensions=embedding_dim,
            )
            graph_embedding = compute_graph_embedding_aggregated(node_embeddings, method="mean")
            all_graph_embeddings.append(graph_embedding)
            all_node_embeddings_list.append(node_embeddings)
    else:
        raise ValueError(f"Unknown embedding method: {embedding_method}. Supported: graph2vec, gnn, node2vec")
    
    embedding_time_ms = int((time.time() - start_time) * 1000)
    logger.info(f"Embedding computation completed in {embedding_time_ms}ms")
    
    # 将图嵌入堆叠成矩阵
    graph_embeddings_matrix = np.array(all_graph_embeddings)
    
    # 降维到3D
    reduction_start = time.time()
    logger.info(f"Reducing {len(graphs)} graph embeddings to 3D using {reduction_method}")
    if reduction_method == "umap":
        graph_coords_3d = reduce_to_3d_umap(graph_embeddings_matrix)
    elif reduction_method == "tsne":
        graph_coords_3d = reduce_to_3d_tsne(graph_embeddings_matrix)
    else:
        raise ValueError(f"Unknown reduction method: {reduction_method}. Supported: umap, tsne")
    
    reduction_time_ms = int((time.time() - reduction_start) * 1000)
    logger.info(f"Dimension reduction completed in {reduction_time_ms}ms")
    
    total_time_ms = embedding_time_ms + reduction_time_ms
    # 将时间平均分配到每个图
    time_per_graph = total_time_ms // len(graphs) if graphs else 0
    
    # 构建结果
    for i, graph in enumerate(graphs):
        node_embeddings = all_node_embeddings_list[i] if i < len(all_node_embeddings_list) else None
        
        result = EmbeddingResult(
            site_id=graph.site_id,
            site_name=graph.site_name,
            site_url=graph.site_url,
            high_dim_embedding=all_graph_embeddings[i].tolist(),
            coord_3d=graph_coords_3d[i].tolist(),
            duration_ms=time_per_graph,
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
            node_embeddings={k: v.tolist() for k, v in node_embeddings.items()} if node_embeddings else None,
        )
        results.append(result)
    
    # 保存到MySQL
    if save_to_db:
        logger.info("Saving embeddings to MySQL (site_tasks table)")
        save_embeddings_to_mysql(db, results)
    
    # 保存到Neo4j
    if save_to_neo4j:
        logger.info("Saving embeddings to Neo4j")
        save_embeddings_to_neo4j(settings, results, save_node_embeddings=False)
    
    logger.info(f"Completed embedding computation for {len(results)} tasks, total time: {total_time_ms}ms (embedding: {embedding_time_ms}ms, reduction: {reduction_time_ms}ms)")
    
    return results


def get_embedding_status_for_tasks(
    db: Session,
    task_ids: Optional[list[int]] = None,
) -> list[dict[str, Any]]:
    """
    获取任务的嵌入状态信息
    
    Args:
        db: 数据库会话
        task_ids: 任务ID列表，None表示获取所有有嵌入的任务
        
    Returns:
        list[dict]: 状态信息列表
    """
    query = db.query(SiteTask)
    
    if task_ids:
        query = query.filter(SiteTask.id.in_(task_ids))
    else:
        query = query.filter(SiteTask.embedding.isnot(None))
    
    results = []
    for task in query.all():
        has_embedding = task.embedding is not None
        has_coord_3d = task.coord_3d is not None
        
        results.append({
            "task_id": task.id,
            "name": task.name or task.site_name,
            "site_name": task.site_name,
            "url": task.url,
            "has_graph": bool(task.graph_json),
            "has_embedding": has_embedding,
            "has_coord_3d": has_coord_3d,
            "embedding_dim": len(task.embedding) if has_embedding else 0,
            "embedding_duration_ms": task.embedding_duration_ms or 0,
            "embedding_updated_at": task.embedding_updated_at,
        })
    
    return results

