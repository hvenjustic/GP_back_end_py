from __future__ import annotations

import logging
from typing import Any

from neo4j import GraphDatabase

from app.config import Settings

logger = logging.getLogger(__name__)


def neo4j_enabled(settings: Settings) -> bool:
    return bool(settings.neo4j_uri and settings.neo4j_user and settings.neo4j_password)


def reset_site_graph(settings: Settings, site_id: int) -> None:
    if not neo4j_enabled(settings):
        return
    driver = _get_driver(settings)
    try:
        with driver.session(database=_database(settings)) as session:
            session.execute_write(_delete_site_tx, int(site_id))
    finally:
        driver.close()


def sync_graph_to_neo4j(
    settings: Settings,
    site: dict[str, Any],
    items: dict[str, list[Any]],
) -> None:
    if not neo4j_enabled(settings):
        return
    site_id = int(site.get("id") or 0)
    if site_id <= 0:
        logger.warning("neo4j sync skipped: invalid site_id=%s", site.get("id"))
        return

    site_url = str(site.get("url") or "").strip()
    site_name = str(site.get("name") or "").strip()

    entities = _normalize_entities(items.get("entities", []))
    relations = _normalize_relations(items.get("relations", []), entities)

    driver = _get_driver(settings)
    try:
        with driver.session(database=_database(settings)) as session:
            session.execute_write(_merge_site_tx, site_id, site_url, site_name)
            if entities:
                session.execute_write(
                    _upsert_entities_tx, site_id, site_url, site_name, entities
                )
            if relations:
                session.execute_write(_upsert_relations_tx, site_id, relations)
    finally:
        driver.close()


def _database(settings: Settings) -> str | None:
    value = (settings.neo4j_database or "").strip()
    return value or None


def _get_driver(settings: Settings):
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def _delete_site_tx(tx, site_id: int) -> None:
    tx.run(
        """
        MATCH (e:Entity {site_id: $site_id})
        DETACH DELETE e
        """,
        site_id=site_id,
    )
    tx.run(
        """
        MATCH (s:Site {id: $site_id})
        DETACH DELETE s
        """,
        site_id=site_id,
    )


def _merge_site_tx(tx, site_id: int, site_url: str, site_name: str) -> None:
    tx.run(
        """
        MERGE (s:Site {id: $site_id})
        SET s.url = $site_url,
            s.name = $site_name,
            s.updated_at = timestamp()
        """,
        site_id=site_id,
        site_url=site_url,
        site_name=site_name,
    )


def _upsert_entities_tx(
    tx, site_id: int, site_url: str, site_name: str, entities: list[dict[str, Any]]
) -> None:
    tx.run(
        """
        MATCH (s:Site {id: $site_id})
        UNWIND $entities AS ent
        MERGE (e:Entity {site_id: $site_id, name: ent.name, type: ent.type})
        SET e.description = ent.description,
            e.site_url = $site_url,
            e.site_name = $site_name,
            e.updated_at = timestamp()
        SET e += ent.extra
        MERGE (s)-[:HAS_ENTITY]->(e)
        """,
        site_id=site_id,
        site_url=site_url,
        site_name=site_name,
        entities=entities,
    )


def _upsert_relations_tx(tx, site_id: int, relations: list[dict[str, Any]]) -> None:
    tx.run(
        """
        UNWIND $relations AS rel
        MATCH (source:Entity {site_id: $site_id, name: rel.source})
        WHERE rel.source_type IS NULL OR source.type = rel.source_type
        MATCH (target:Entity {site_id: $site_id, name: rel.target})
        WHERE rel.target_type IS NULL OR target.type = rel.target_type
        MERGE (source)-[r:RELATION {site_id: $site_id, type: rel.type}]->(target)
        SET r.updated_at = timestamp()
        SET r += rel.props
        """,
        site_id=site_id,
        relations=relations,
    )


def _normalize_entities(raw: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return items
    for ent in raw:
        if not isinstance(ent, dict):
            continue
        name = str(ent.get("name") or "").strip()
        if not name:
            continue
        type_level_1 = str(ent.get("type_level_1") or "").strip()
        type_level_2 = str(ent.get("type_level_2") or "").strip()
        ent_type = str(ent.get("type") or "").strip()
        if not ent_type:
            ent_type = type_level_2 or type_level_1
        description = str(ent.get("description") or "").strip()
        extra_raw = ent.get("extra")
        extra: dict[str, Any] = {}
        if isinstance(extra_raw, dict):
            for key, value in extra_raw.items():
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    extra[str(key)] = text
        if type_level_1 and "type_level_1" not in extra:
            extra["type_level_1"] = type_level_1
        if type_level_2 and "type_level_2" not in extra:
            extra["type_level_2"] = type_level_2
        items.append(
            {
                "name": name,
                "type": ent_type,
                "description": description,
                "extra": extra,
            }
        )
    return items


def _normalize_relations(
    raw: Any, entities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return items

    name_to_types: dict[str, set[str]] = {}
    for ent in entities:
        name = ent.get("name")
        ent_type = ent.get("type") or ""
        if not name:
            continue
        name_to_types.setdefault(name, set()).add(ent_type)

    for rel in raw:
        if not isinstance(rel, dict):
            continue
        source = str(rel.get("source") or "").strip()
        target = str(rel.get("target") or "").strip()
        rel_type = str(rel.get("type") or "").strip() or "RELATED_TO"
        if not source or not target:
            continue

        source_types = sorted(name_to_types.get(source, []))
        target_types = sorted(name_to_types.get(target, []))
        source_type = source_types[0] if len(source_types) == 1 else None
        target_type = target_types[0] if len(target_types) == 1 else None

        props: dict[str, Any] = {"type": rel_type}
        if len(source_types) > 1:
            props["source_types"] = source_types
        if len(target_types) > 1:
            props["target_types"] = target_types

        items.append(
            {
                "source": source,
                "target": target,
                "type": rel_type,
                "source_type": source_type,
                "target_type": target_type,
                "props": props,
            }
        )
    return items
