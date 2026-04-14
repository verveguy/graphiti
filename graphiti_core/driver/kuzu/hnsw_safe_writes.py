"""HNSW-safe write helpers for KuzuDB/LadybugDB.

KuzuDB does not allow SET on columns that have HNSW vector indexes.
These helpers replace the MERGE+SET pattern with DELETE+INSERT,
preserving relationships across the delete-create cycle.
"""

import logging
from typing import Any

from graphiti_core.driver.driver import GraphDriverSession
from graphiti_core.driver.query_executor import QueryExecutor, Transaction

TxLike = Transaction | GraphDriverSession | None

logger = logging.getLogger(__name__)


async def _query(
    executor: QueryExecutor,
    query: str,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    # Always uses executor (not tx) because KuzuDB tx.run() doesn't return results.
    records, _, _ = await executor.execute_query(query, **kwargs)
    return records  # type: ignore


async def _write(
    executor: QueryExecutor,
    tx: TxLike,
    query: str,
    **kwargs: Any,
) -> None:
    if tx is not None:
        await tx.run(query, **kwargs)  # type: ignore[union-attr]
    else:
        await executor.execute_query(query, **kwargs)


# ---------------------------------------------------------------------------
# Entity Edge (RelatesToNode_)
# ---------------------------------------------------------------------------
# Simplest case: source_uuid and target_uuid are already in save params,
# so no relationship backup is needed.


async def hnsw_safe_save_entity_edge(
    executor: QueryExecutor,
    tx: TxLike,
    params: dict[str, Any],
) -> None:
    uuid = params['uuid']
    source_uuid = params.get('source_uuid') or params['source_node_uuid']
    target_uuid = params.get('target_uuid') or params['target_node_uuid']

    await _write(
        executor,
        tx,
        'MATCH (e:RelatesToNode_ {uuid: $uuid}) DETACH DELETE e',
        uuid=uuid,
    )

    await _write(
        executor,
        tx,
        """
        CREATE (e:RelatesToNode_ {
            uuid: $uuid,
            group_id: $group_id,
            created_at: $created_at,
            name: $name,
            fact: $fact,
            fact_embedding: $fact_embedding,
            episodes: $episodes,
            expired_at: $expired_at,
            valid_at: $valid_at,
            invalid_at: $invalid_at,
            attributes: $attributes
        })
        """,
        **params,
    )

    await _write(
        executor,
        tx,
        """
        MATCH (source:Entity {uuid: $source_uuid}), (e:RelatesToNode_ {uuid: $uuid}),
              (target:Entity {uuid: $target_uuid})
        CREATE (source)-[:RELATES_TO]->(e), (e)-[:RELATES_TO]->(target)
        """,
        source_uuid=source_uuid,
        uuid=uuid,
        target_uuid=target_uuid,
    )


# ---------------------------------------------------------------------------
# Entity Node
# ---------------------------------------------------------------------------


async def hnsw_safe_save_entity_node(
    executor: QueryExecutor,
    tx: TxLike,
    params: dict[str, Any],
) -> None:
    uuid = params['uuid']

    existing = await _query(
        executor,
        'MATCH (n:Entity {uuid: $uuid}) RETURN n.uuid AS uuid',
        uuid=uuid,
    )

    if existing:
        mentions = await _query(
            executor,
            """
            MATCH (ep:Episodic)-[m:MENTIONS]->(n:Entity {uuid: $uuid})
            RETURN ep.uuid AS episode_uuid, m.uuid AS mention_uuid,
                   m.group_id AS group_id, m.created_at AS created_at
            """,
            uuid=uuid,
        )

        relates_to_outgoing = await _query(
            executor,
            """
            MATCH (n:Entity {uuid: $uuid})-[:RELATES_TO]->(r:RelatesToNode_)
            RETURN r.uuid AS relates_to_node_uuid
            """,
            uuid=uuid,
        )

        relates_to_incoming = await _query(
            executor,
            """
            MATCH (r:RelatesToNode_)-[:RELATES_TO]->(n:Entity {uuid: $uuid})
            RETURN r.uuid AS relates_to_node_uuid
            """,
            uuid=uuid,
        )

        has_member = await _query(
            executor,
            """
            MATCH (c:Community)-[h:HAS_MEMBER]->(n:Entity {uuid: $uuid})
            RETURN c.uuid AS community_uuid, h.uuid AS member_uuid,
                   h.group_id AS group_id, h.created_at AS created_at
            """,
            uuid=uuid,
        )

        await _write(
            executor,
            tx,
            'MATCH (n:Entity {uuid: $uuid}) DETACH DELETE n',
            uuid=uuid,
        )
    else:
        mentions = []
        relates_to_outgoing = []
        relates_to_incoming = []
        has_member = []

    await _write(
        executor,
        tx,
        """
        CREATE (n:Entity {
            uuid: $uuid,
            name: $name,
            group_id: $group_id,
            labels: $labels,
            created_at: $created_at,
            name_embedding: $name_embedding,
            summary: $summary,
            attributes: $attributes
        })
        """,
        **params,
    )

    for m in mentions:
        await _write(
            executor,
            tx,
            """
            MATCH (ep:Episodic {uuid: $episode_uuid}), (n:Entity {uuid: $uuid})
            CREATE (ep)-[:MENTIONS {uuid: $mention_uuid, group_id: $group_id, created_at: $created_at}]->(n)
            """,
            uuid=uuid,
            episode_uuid=m['episode_uuid'],
            mention_uuid=m['mention_uuid'],
            group_id=m['group_id'],
            created_at=m['created_at'],
        )

    for r in relates_to_outgoing:
        await _write(
            executor,
            tx,
            """
            MATCH (n:Entity {uuid: $uuid}), (r:RelatesToNode_ {uuid: $relates_to_node_uuid})
            CREATE (n)-[:RELATES_TO]->(r)
            """,
            uuid=uuid,
            relates_to_node_uuid=r['relates_to_node_uuid'],
        )

    for r in relates_to_incoming:
        await _write(
            executor,
            tx,
            """
            MATCH (r:RelatesToNode_ {uuid: $relates_to_node_uuid}), (n:Entity {uuid: $uuid})
            CREATE (r)-[:RELATES_TO]->(n)
            """,
            uuid=uuid,
            relates_to_node_uuid=r['relates_to_node_uuid'],
        )

    for h in has_member:
        await _write(
            executor,
            tx,
            """
            MATCH (c:Community {uuid: $community_uuid}), (n:Entity {uuid: $uuid})
            CREATE (c)-[:HAS_MEMBER {uuid: $member_uuid, group_id: $group_id, created_at: $created_at}]->(n)
            """,
            uuid=uuid,
            community_uuid=h['community_uuid'],
            member_uuid=h['member_uuid'],
            group_id=h['group_id'],
            created_at=h['created_at'],
        )


# ---------------------------------------------------------------------------
# Community Node
# ---------------------------------------------------------------------------


async def hnsw_safe_save_community_node(
    executor: QueryExecutor,
    tx: TxLike,
    params: dict[str, Any],
) -> None:
    uuid = params['uuid']

    existing = await _query(
        executor,
        'MATCH (c:Community {uuid: $uuid}) RETURN c.uuid AS uuid',
        uuid=uuid,
    )

    if existing:
        has_member_to_entity = await _query(
            executor,
            """
            MATCH (c:Community {uuid: $uuid})-[h:HAS_MEMBER]->(n:Entity)
            RETURN n.uuid AS entity_uuid, h.uuid AS member_uuid,
                   h.group_id AS group_id, h.created_at AS created_at
            """,
            uuid=uuid,
        )

        has_member_to_community = await _query(
            executor,
            """
            MATCH (c:Community {uuid: $uuid})-[h:HAS_MEMBER]->(child:Community)
            WHERE child.uuid <> $uuid
            RETURN child.uuid AS child_uuid, h.uuid AS member_uuid,
                   h.group_id AS group_id, h.created_at AS created_at
            """,
            uuid=uuid,
        )

        has_member_from_community = await _query(
            executor,
            """
            MATCH (parent:Community)-[h:HAS_MEMBER]->(c:Community {uuid: $uuid})
            WHERE parent.uuid <> $uuid
            RETURN parent.uuid AS parent_uuid, h.uuid AS member_uuid,
                   h.group_id AS group_id, h.created_at AS created_at
            """,
            uuid=uuid,
        )

        await _write(
            executor,
            tx,
            'MATCH (c:Community {uuid: $uuid}) DETACH DELETE c',
            uuid=uuid,
        )
    else:
        has_member_to_entity = []
        has_member_to_community = []
        has_member_from_community = []

    await _write(
        executor,
        tx,
        """
        CREATE (c:Community {
            uuid: $uuid,
            name: $name,
            group_id: $group_id,
            created_at: $created_at,
            name_embedding: $name_embedding,
            summary: $summary
        })
        """,
        **params,
    )

    for h in has_member_to_entity:
        await _write(
            executor,
            tx,
            """
            MATCH (c:Community {uuid: $uuid}), (n:Entity {uuid: $entity_uuid})
            CREATE (c)-[:HAS_MEMBER {uuid: $member_uuid, group_id: $group_id, created_at: $created_at}]->(n)
            """,
            uuid=uuid,
            entity_uuid=h['entity_uuid'],
            member_uuid=h['member_uuid'],
            group_id=h['group_id'],
            created_at=h['created_at'],
        )

    for h in has_member_to_community:
        await _write(
            executor,
            tx,
            """
            MATCH (c:Community {uuid: $uuid}), (child:Community {uuid: $child_uuid})
            CREATE (c)-[:HAS_MEMBER {uuid: $member_uuid, group_id: $group_id, created_at: $created_at}]->(child)
            """,
            uuid=uuid,
            child_uuid=h['child_uuid'],
            member_uuid=h['member_uuid'],
            group_id=h['group_id'],
            created_at=h['created_at'],
        )

    for h in has_member_from_community:
        await _write(
            executor,
            tx,
            """
            MATCH (parent:Community {uuid: $parent_uuid}), (c:Community {uuid: $uuid})
            CREATE (parent)-[:HAS_MEMBER {uuid: $member_uuid, group_id: $group_id, created_at: $created_at}]->(c)
            """,
            uuid=uuid,
            parent_uuid=h['parent_uuid'],
            member_uuid=h['member_uuid'],
            group_id=h['group_id'],
            created_at=h['created_at'],
        )


# ---------------------------------------------------------------------------
# Episodic Node
# ---------------------------------------------------------------------------
# Episodic.content_embedding is HNSW-indexed, so MERGE+SET fails on it.
# Outgoing: MENTIONS->Entity, NEXT_EPISODE->Episodic
# Incoming: NEXT_EPISODE from Episodic, HAS_EPISODE from Saga


async def hnsw_safe_save_episode_node(
    executor: QueryExecutor,
    tx: TxLike,
    params: dict[str, Any],
) -> None:
    uuid = params['uuid']

    existing = await _query(
        executor,
        'MATCH (n:Episodic {uuid: $uuid}) RETURN n.uuid AS uuid',
        uuid=uuid,
    )

    if existing:
        mentions = await _query(
            executor,
            """
            MATCH (ep:Episodic {uuid: $uuid})-[m:MENTIONS]->(n:Entity)
            RETURN n.uuid AS entity_uuid, m.uuid AS mention_uuid,
                   m.group_id AS group_id, m.created_at AS created_at
            """,
            uuid=uuid,
        )

        next_episode_outgoing = await _query(
            executor,
            """
            MATCH (ep:Episodic {uuid: $uuid})-[e:NEXT_EPISODE]->(target:Episodic)
            RETURN target.uuid AS target_uuid, e.uuid AS edge_uuid,
                   e.group_id AS group_id, e.created_at AS created_at
            """,
            uuid=uuid,
        )

        next_episode_incoming = await _query(
            executor,
            """
            MATCH (source:Episodic)-[e:NEXT_EPISODE]->(ep:Episodic {uuid: $uuid})
            WHERE source.uuid <> $uuid
            RETURN source.uuid AS source_uuid, e.uuid AS edge_uuid,
                   e.group_id AS group_id, e.created_at AS created_at
            """,
            uuid=uuid,
        )

        has_episode = await _query(
            executor,
            """
            MATCH (s:Saga)-[e:HAS_EPISODE]->(ep:Episodic {uuid: $uuid})
            RETURN s.uuid AS saga_uuid, e.uuid AS edge_uuid,
                   e.group_id AS group_id, e.created_at AS created_at
            """,
            uuid=uuid,
        )

        await _write(
            executor,
            tx,
            'MATCH (n:Episodic {uuid: $uuid}) DETACH DELETE n',
            uuid=uuid,
        )
    else:
        mentions = []
        next_episode_outgoing = []
        next_episode_incoming = []
        has_episode = []

    await _write(
        executor,
        tx,
        """
        CREATE (n:Episodic {
            uuid: $uuid,
            name: $name,
            group_id: $group_id,
            created_at: $created_at,
            source: $source,
            source_description: $source_description,
            content: $content,
            content_embedding: $content_embedding,
            valid_at: $valid_at,
            entity_edges: $entity_edges
        })
        """,
        **params,
    )

    for m in mentions:
        await _write(
            executor,
            tx,
            """
            MATCH (ep:Episodic {uuid: $uuid}), (n:Entity {uuid: $entity_uuid})
            CREATE (ep)-[:MENTIONS {uuid: $mention_uuid, group_id: $group_id, created_at: $created_at}]->(n)
            """,
            uuid=uuid,
            entity_uuid=m['entity_uuid'],
            mention_uuid=m['mention_uuid'],
            group_id=m['group_id'],
            created_at=m['created_at'],
        )

    for e in next_episode_outgoing:
        await _write(
            executor,
            tx,
            """
            MATCH (ep:Episodic {uuid: $uuid}), (target:Episodic {uuid: $target_uuid})
            CREATE (ep)-[:NEXT_EPISODE {uuid: $edge_uuid, group_id: $group_id, created_at: $created_at}]->(target)
            """,
            uuid=uuid,
            target_uuid=e['target_uuid'],
            edge_uuid=e['edge_uuid'],
            group_id=e['group_id'],
            created_at=e['created_at'],
        )

    for e in next_episode_incoming:
        await _write(
            executor,
            tx,
            """
            MATCH (source:Episodic {uuid: $source_uuid}), (ep:Episodic {uuid: $uuid})
            CREATE (source)-[:NEXT_EPISODE {uuid: $edge_uuid, group_id: $group_id, created_at: $created_at}]->(ep)
            """,
            uuid=uuid,
            source_uuid=e['source_uuid'],
            edge_uuid=e['edge_uuid'],
            group_id=e['group_id'],
            created_at=e['created_at'],
        )

    for e in has_episode:
        await _write(
            executor,
            tx,
            """
            MATCH (s:Saga {uuid: $saga_uuid}), (ep:Episodic {uuid: $uuid})
            CREATE (s)-[:HAS_EPISODE {uuid: $edge_uuid, group_id: $group_id, created_at: $created_at}]->(ep)
            """,
            uuid=uuid,
            saga_uuid=e['saga_uuid'],
            edge_uuid=e['edge_uuid'],
            group_id=e['group_id'],
            created_at=e['created_at'],
        )
