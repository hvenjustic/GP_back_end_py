import os
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from dotenv import load_dotenv
import langextract as lx


# -----------------------
# 0) 小工具
# -----------------------
def stable_id(*parts: str) -> str:
    h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def normalize_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def iter_md_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.md") if p.is_file()]


def read_text(p: Path) -> str:
    # 你的 md 可能含特殊字符，errors=ignore 更稳
    return p.read_text(encoding="utf-8", errors="ignore")


def safe_char_interval(extraction: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    LangExtract 的 Extraction 里有 char_interval=CharInterval(start_pos, end_pos)
    有些情况下可能为 None（例如未对齐）。:contentReference[oaicite:6]{index=6}
    """
    ci = getattr(extraction, "char_interval", None)
    if ci is None:
        return None, None
    return getattr(ci, "start_pos", None), getattr(ci, "end_pos", None)


# -----------------------
# 1) 定义“知识图谱抽取任务”的 Prompt + Few-shot 示例
# -----------------------
PROMPT_DESCRIPTION = r"""
你是信息抽取系统。你的任务是从 Markdown 文本中抽取可用于构建知识图谱的结构化信息。

抽取两类信息（按出现顺序输出）：

1) entity（实体）
- extraction_text：必须是原文中出现的实体文本（不要改写、不要翻译、不要补全）
- attributes：
  - type: 实体类型（Organization | Person | Product | Technology | Location | Other）
  - canonical_name: 实体规范名（允许做轻微归一化，如去掉多余空格；但不要凭空发明）

2) relation（关系）
- extraction_text：必须是原文中直接表达该关系的“证据片段”（一句话或短语，必须原文出现）
- attributes：
  - subject: 主体实体（尽量与某个 entity 的 canonical_name 对齐）
  - predicate: 关系谓词（例如: located_in / produces / partners_with / founded_by / develops / supplies / offers / acquired_by）
  - object: 客体实体（尽量与某个 entity 的 canonical_name 对齐）
  - subject_type: 主体类型（同上）
  - object_type: 客体类型（同上）

严格要求：
- 不要输出重复实体（同一文档内尽量去重）
- 不要输出与知识图谱无关的泛化描述
- entity/relation 的 extraction_text 必须来自原文（不允许 paraphrase），否则会影响对齐定位
""".strip()

EXAMPLES = [
    lx.data.ExampleData(
        text=(
            "## About\n"
            "Acme Bio Inc. is headquartered in San Diego, California. "
            "Acme Bio develops cell-free protein synthesis platforms. "
            "CEO Jane Doe previously worked at Example University.\n"
        ),
        extractions=[
            # 实体
            lx.data.Extraction(
                extraction_class="entity",
                extraction_text="Acme Bio Inc.",
                attributes={"type": "Organization", "canonical_name": "Acme Bio Inc."},
            ),
            lx.data.Extraction(
                extraction_class="entity",
                extraction_text="San Diego, California",
                attributes={"type": "Location", "canonical_name": "San Diego, California"},
            ),
            lx.data.Extraction(
                extraction_class="entity",
                extraction_text="cell-free protein synthesis",
                attributes={"type": "Technology", "canonical_name": "cell-free protein synthesis"},
            ),
            lx.data.Extraction(
                extraction_class="entity",
                extraction_text="Jane Doe",
                attributes={"type": "Person", "canonical_name": "Jane Doe"},
            ),
            lx.data.Extraction(
                extraction_class="entity",
                extraction_text="Example University",
                attributes={"type": "Organization", "canonical_name": "Example University"},
            ),

            # 关系（注意 extraction_text 是原文证据片段）
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="is headquartered in San Diego, California",
                attributes={
                    "subject": "Acme Bio Inc.",
                    "predicate": "located_in",
                    "object": "San Diego, California",
                    "subject_type": "Organization",
                    "object_type": "Location",
                },
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="develops cell-free protein synthesis platforms",
                attributes={
                    "subject": "Acme Bio Inc.",
                    "predicate": "develops",
                    "object": "cell-free protein synthesis",
                    "subject_type": "Organization",
                    "object_type": "Technology",
                },
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="worked at Example University",
                attributes={
                    "subject": "Jane Doe",
                    "predicate": "works_at",
                    "object": "Example University",
                    "subject_type": "Person",
                    "object_type": "Organization",
                },
            ),
        ],
    )
]


# -----------------------
# 2) 选择模型 + 完整配置（含智能分块/长文档参数）
# -----------------------
def build_extract_kwargs(model_id: str) -> Dict[str, Any]:
    """
    LangExtract 针对长文档的优化参数（chunking + 并行 + 多次抽取）：
    extraction_passes / max_workers / max_char_buffer :contentReference[oaicite:7]{index=7}
    """
    common = dict(
        prompt_description=PROMPT_DESCRIPTION,
        examples=EXAMPLES,
        model_id=model_id,

        # --- 智能分块/长文档优化 ---
        extraction_passes=2,     # 长文档可提高到 3（召回更高，但更慢更贵）:contentReference[oaicite:8]{index=8}
        max_workers=12,          # 并发（受模型限流影响）:contentReference[oaicite:9]{index=9}
        max_char_buffer=1200,    # 单块上下文大小：越小越“敏感”，但块数会更多 :contentReference[oaicite:10]{index=10}

        # 你也可以加 debug=True 看更详细日志（版本不同可能有差异）
        # debug=True,
    )

    # OpenAI 模型的特殊要求：fence_output=True 且 use_schema_constraints=False :contentReference[oaicite:11]{index=11}
    if model_id.lower().startswith(("gpt-", "o1", "o3", "gpt4", "gpt-4")):
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        common.update(
            api_key=os.getenv("OPENAI_API_KEY"),
            fence_output=True,
            use_schema_constraints=False,
        )
        # base_url：用于 OpenAI-compatible 中转/厂商（通过 language_model_params 传入）:contentReference[oaicite:12]{index=12}
        if base_url:
            common["language_model_params"] = {"base_url": base_url}
        return common

    # Gemini：README 用 LANGEXTRACT_API_KEY（也可直接 api_key=...）:contentReference[oaicite:13]{index=13}
    common.update(
        api_key=os.getenv("LANGEXTRACT_API_KEY")
    )
    return common


# -----------------------
# 3) 抽取 + 组装 KG（nodes / edges）
# -----------------------
def extract_one_doc(text: str, extract_kwargs: Dict[str, Any]) -> Any:
    return lx.extract(text_or_documents=text, **extract_kwargs)


def build_graph_from_result(
    annotated_doc: Any,
    doc_id: str,
    source_path: str,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    annotated_doc 是 AnnotatedDocument，extractions 是 Extraction 列表；其中包含 char_interval 定位信息。:contentReference[oaicite:14]{index=14}
    """
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    extractions = getattr(annotated_doc, "extractions", []) or []
    for ex in extractions:
        ex_class = getattr(ex, "extraction_class", None)
        ex_text = getattr(ex, "extraction_text", "") or ""
        attrs = getattr(ex, "attributes", {}) or {}
        start_pos, end_pos = safe_char_interval(ex)

        if ex_class == "entity":
            etype = (attrs.get("type") or "Other").strip()
            canonical = normalize_name(attrs.get("canonical_name") or ex_text)
            nid = stable_id("entity", etype, canonical)

            if nid not in nodes:
                nodes[nid] = {
                    "id": nid,
                    "label": etype,
                    "name": canonical,
                    "aliases": sorted({normalize_name(ex_text)} - {canonical}),
                    "provenance": [{
                        "doc_id": doc_id,
                        "source_path": source_path,
                        "evidence_text": ex_text,
                        "char_start": start_pos,
                        "char_end": end_pos,
                    }],
                }
            else:
                # 追加别名/溯源
                nodes[nid]["aliases"] = sorted(set(nodes[nid]["aliases"]) | {normalize_name(ex_text)})
                nodes[nid]["provenance"].append({
                    "doc_id": doc_id,
                    "source_path": source_path,
                    "evidence_text": ex_text,
                    "char_start": start_pos,
                    "char_end": end_pos,
                })

        elif ex_class == "relation":
            subj = normalize_name(attrs.get("subject", ""))
            pred = normalize_name(attrs.get("predicate", "related_to"))
            obj = normalize_name(attrs.get("object", ""))

            subj_type = (attrs.get("subject_type") or "Other").strip()
            obj_type = (attrs.get("object_type") or "Other").strip()

            # 为关系两端生成（或复用）node id（即使 entity 未抽到，也兜底创建）
            subj_id = stable_id("entity", subj_type, subj or "UNKNOWN_SUBJECT")
            obj_id = stable_id("entity", obj_type, obj or "UNKNOWN_OBJECT")

            if subj and subj_id not in nodes:
                nodes[subj_id] = {"id": subj_id, "label": subj_type, "name": subj, "aliases": [], "provenance": []}
            if obj and obj_id not in nodes:
                nodes[obj_id] = {"id": obj_id, "label": obj_type, "name": obj, "aliases": [], "provenance": []}

            eid = stable_id("edge", subj_id, pred, obj_id, doc_id, str(start_pos), str(end_pos))
            edges.append({
                "id": eid,
                "type": pred,
                "source": subj_id,
                "target": obj_id,
                "provenance": {
                    "doc_id": doc_id,
                    "source_path": source_path,
                    "evidence_text": ex_text,   # 关系证据片段（原文）——便于审计/可视化
                    "char_start": start_pos,
                    "char_end": end_pos,
                }
            })

    return nodes, edges


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    load_dotenv()

    md_root = Path(os.getenv("MD_ROOT", "./md_dump")).resolve()
    out_dir = Path(os.getenv("OUT_DIR", "./out")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = os.getenv("MODEL_ID", "gemini-2.5-flash").strip()
    extract_kwargs = build_extract_kwargs(model_id)

    md_files = iter_md_files(md_root)
    if not md_files:
        raise RuntimeError(f"No .md files found under: {md_root}")

    all_nodes: Dict[str, Dict[str, Any]] = {}
    all_edges: List[Dict[str, Any]] = []
    annotated_docs: List[Any] = []

    for i, p in enumerate(md_files, 1):
        text = read_text(p)
        doc_id = stable_id("doc", str(p.relative_to(md_root)))

        print(f"[{i}/{len(md_files)}] extracting: {p}")

        result = extract_one_doc(text, extract_kwargs)
        annotated_docs.append(result)

        nodes, edges = build_graph_from_result(
            annotated_doc=result,
            doc_id=doc_id,
            source_path=str(p.relative_to(md_root)),
        )

        # merge nodes
        for nid, n in nodes.items():
            if nid not in all_nodes:
                all_nodes[nid] = n
            else:
                # 合并别名和 provenance
                all_nodes[nid]["aliases"] = sorted(set(all_nodes[nid].get("aliases", [])) | set(n.get("aliases", [])))
                all_nodes[nid]["provenance"] = (all_nodes[nid].get("provenance", []) + n.get("provenance", []))

        all_edges.extend(edges)

    # 1) 保存 LangExtract 的可审计结果（jsonl）并可视化 :contentReference[oaicite:15]{index=15}
    lx.io.save_annotated_documents(annotated_docs, output_name="annotated_documents.jsonl", output_dir=str(out_dir))
    html_content = lx.visualize(str(out_dir / "annotated_documents.jsonl"))
    html_path = out_dir / "visualization.html"
    with html_path.open("w", encoding="utf-8") as f:
        if hasattr(html_content, "data"):
            f.write(html_content.data)
        else:
            f.write(html_content)

    # 2) 输出知识图谱 nodes/edges（jsonl）
    write_jsonl(out_dir / "kg_nodes.jsonl", list(all_nodes.values()))
    write_jsonl(out_dir / "kg_edges.jsonl", all_edges)

    print("\nDone.")
    print(f"- Annotated JSONL: {out_dir/'annotated_documents.jsonl'}")
    print(f"- Visualization : {html_path}")
    print(f"- KG nodes      : {out_dir/'kg_nodes.jsonl'}")
    print(f"- KG edges      : {out_dir/'kg_edges.jsonl'}")


if __name__ == "__main__":
    main()
