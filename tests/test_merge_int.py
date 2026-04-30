"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from datetime import datetime
from uuid import uuid4

import pytest

from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.graphiti import Graphiti, MergeEntitiesResult, MergeRequest
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from tests.helpers_test import get_node_count, group_id, group_id_2

pytest_plugins = ('pytest_asyncio',)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 384


def _node(name: str, labels: list[str], gid: str = group_id, summary: str = '') -> EntityNode:
    return EntityNode(
        uuid=str(uuid4()),
        name=name,
        group_id=gid,
        labels=labels,
        created_at=datetime.now(),
        name_embedding=[0.1] * EMBEDDING_DIM,
        summary=summary,
        attributes={},
    )


def _episode(gid: str = group_id) -> EpisodicNode:
    return EpisodicNode(
        uuid=str(uuid4()),
        name='test_episode',
        group_id=gid,
        created_at=datetime.now(),
        source=EpisodeType.text,
        source_description='test',
        content='test content',
        valid_at=datetime.now(),
        entity_edges=[],
    )


def _entity_edge(
    src_uuid: str,
    tgt_uuid: str,
    gid: str = group_id,
    name: str = 'RELATES_TO',
    fact: str = 'default fact',
) -> EntityEdge:
    return EntityEdge(
        uuid=str(uuid4()),
        source_node_uuid=src_uuid,
        target_node_uuid=tgt_uuid,
        group_id=gid,
        name=name,
        fact=fact,
        fact_embedding=[0.2] * EMBEDDING_DIM,
        created_at=datetime.now(),
        episodes=[],
    )


def _mentions(episode_uuid: str, entity_uuid: str, gid: str = group_id) -> EpisodicEdge:
    return EpisodicEdge(
        uuid=str(uuid4()),
        source_node_uuid=episode_uuid,
        target_node_uuid=entity_uuid,
        group_id=gid,
        created_at=datetime.now(),
    )


async def _graphiti(graph_driver):
    g = Graphiti(graph_driver=graph_driver)
    return g


async def _get_mentions_count(driver, entity_uuid: str) -> int:
    results, _, _ = await driver.execute_query(
        'MATCH (n:Episodic)-[e:MENTIONS]->(m:Entity {uuid: $uuid}) RETURN COUNT(e) AS cnt',
        uuid=entity_uuid,
        routing_='r',
    )
    return int(results[0]['cnt'])


# ---------------------------------------------------------------------------
# Happy path: two duplicate entities merged into survivor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_entities_happy_path(graph_driver, mock_embedder):
    """Merge two duplicate entity nodes; verify edges/MENTIONS are rewired."""
    g = await _graphiti(graph_driver)

    survivor = _node('Visualization Data Service', ['Entity', 'Service'])
    dup = _node('Visualization Data Service', ['Entity', 'DataService'])
    third = _node('Consumer', ['Entity'])

    for node in [survivor, dup, third]:
        await node.save(graph_driver)

    # Edges: third -> dup, and dup -> third
    edge_in = _entity_edge(third.uuid, dup.uuid, fact='Consumer uses VDS')
    edge_out = _entity_edge(dup.uuid, third.uuid, fact='VDS serves Consumer')
    await edge_in.save(graph_driver)
    await edge_out.save(graph_driver)

    # Episode MENTIONS dup
    ep = _episode()
    await ep.save(graph_driver)
    mention = _mentions(ep.uuid, dup.uuid)
    await mention.save(graph_driver)

    result = await g.merge_entities(
        survivor_uuid=survivor.uuid,
        duplicate_uuids=[dup.uuid],
    )

    # dup node is gone
    dup_count = await get_node_count(graph_driver, [dup.uuid])
    assert dup_count == 0, 'duplicate node should be deleted'

    # survivor still exists
    survivor_count = await get_node_count(graph_driver, [survivor.uuid])
    assert survivor_count == 1, 'survivor node should still exist'

    # labels are unioned
    assert sorted(result.labels_after) == sorted({'Entity', 'Service', 'DataService'})

    # edge endpoints rewired to survivor
    assert result.edges_rewired == 2

    # episode now mentions survivor, not dup
    assert result.episodes_relinked == 1
    dup_mentions = await _get_mentions_count(graph_driver, dup.uuid)
    assert dup_mentions == 0
    survivor_mentions = await _get_mentions_count(graph_driver, survivor.uuid)
    assert survivor_mentions == 1

    # result counters
    assert result.merged_count == 1
    assert result.survivor_uuid == survivor.uuid
    assert result.error is None

    await g.close()


# ---------------------------------------------------------------------------
# Summary line-deduplication
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_entities_summary_dedup(graph_driver, mock_embedder):
    """Duplicate summary lines are not repeated; unique lines are appended."""
    g = await _graphiti(graph_driver)

    survivor = _node('Alpha', ['Entity'], summary='Line A\nLine B')
    dup = _node('Alpha', ['Entity'], summary='Line B\nLine C')

    for node in [survivor, dup]:
        await node.save(graph_driver)

    result = await g.merge_entities(survivor_uuid=survivor.uuid, duplicate_uuids=[dup.uuid])

    lines = set(result.summary_after.splitlines())
    assert 'Line A' in lines
    assert 'Line B' in lines
    assert 'Line C' in lines
    # Line B should appear exactly once
    assert result.summary_after.count('Line B') == 1

    await g.close()


# ---------------------------------------------------------------------------
# Dry-run: graph is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_entities_dry_run(graph_driver, mock_embedder):
    """dry_run=True returns a result but makes no writes."""
    g = await _graphiti(graph_driver)

    survivor = _node('SvcA', ['Entity', 'ServiceA'])
    dup = _node('SvcA', ['Entity', 'ServiceB'])

    for node in [survivor, dup]:
        await node.save(graph_driver)

    ep = _episode()
    await ep.save(graph_driver)
    mention = _mentions(ep.uuid, dup.uuid)
    await mention.save(graph_driver)

    edge = _entity_edge(dup.uuid, survivor.uuid, fact='SvcA fact')
    await edge.save(graph_driver)

    node_count_before = await get_node_count(graph_driver, [survivor.uuid, dup.uuid])

    result = await g.merge_entities(
        survivor_uuid=survivor.uuid,
        duplicate_uuids=[dup.uuid],
        dry_run=True,
    )

    # Result shape is valid with accurate counts (dry-run still computes what would change)
    assert isinstance(result, MergeEntitiesResult)
    assert result.survivor_uuid == survivor.uuid
    assert result.merged_count == 1
    assert result.edges_rewired == 1
    assert result.episodes_relinked == 1
    assert result.error is None

    # Graph is unchanged
    node_count_after = await get_node_count(graph_driver, [survivor.uuid, dup.uuid])
    assert node_count_after == node_count_before == 2

    # dup still has its MENTIONS edge
    dup_mentions = await _get_mentions_count(graph_driver, dup.uuid)
    assert dup_mentions == 1

    await g.close()


# ---------------------------------------------------------------------------
# Batch merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_entities_batch(graph_driver, mock_embedder):
    """merge_entities_batch processes each pair and returns one result each."""
    g = await _graphiti(graph_driver)

    s1 = _node('Service1', ['Entity', 'S1'])
    d1 = _node('Service1Dup', ['Entity', 'S1Dup'])
    s2 = _node('Service2', ['Entity', 'S2'])
    d2 = _node('Service2Dup', ['Entity', 'S2Dup'])

    for node in [s1, d1, s2, d2]:
        await node.save(graph_driver)

    merges = [
        MergeRequest(survivor_uuid=s1.uuid, duplicate_uuids=[d1.uuid]),
        MergeRequest(survivor_uuid=s2.uuid, duplicate_uuids=[d2.uuid]),
    ]

    results = await g.merge_entities_batch(merges=merges)

    assert len(results) == 2
    assert results[0].survivor_uuid == s1.uuid
    assert results[0].merged_count == 1
    assert results[0].error is None
    assert results[1].survivor_uuid == s2.uuid
    assert results[1].merged_count == 1
    assert results[1].error is None

    # Both duplicates are deleted
    assert await get_node_count(graph_driver, [d1.uuid]) == 0
    assert await get_node_count(graph_driver, [d2.uuid]) == 0

    await g.close()


# ---------------------------------------------------------------------------
# Missing duplicate UUID: skip with warning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_entities_missing_duplicate(graph_driver, mock_embedder, caplog):
    """Missing duplicate UUID is skipped with a warning; operation succeeds."""
    g = await _graphiti(graph_driver)

    survivor = _node('MySvc', ['Entity'])
    await survivor.save(graph_driver)

    nonexistent_uuid = str(uuid4())

    with caplog.at_level(logging.WARNING, logger='graphiti_core.graphiti'):
        result = await g.merge_entities(
            survivor_uuid=survivor.uuid,
            duplicate_uuids=[nonexistent_uuid],
        )

    assert result.merged_count == 0
    assert result.error is None
    assert any(nonexistent_uuid in record.message for record in caplog.records)

    await g.close()


# ---------------------------------------------------------------------------
# group_id mismatch: raises ValueError before any write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_entities_group_id_mismatch(graph_driver, mock_embedder):
    """Merging entities across group_id boundaries raises ValueError."""
    g = await _graphiti(graph_driver)

    survivor = _node('SvcA', ['Entity'], gid=group_id)
    dup = _node('SvcA', ['Entity'], gid=group_id_2)

    for node in [survivor, dup]:
        await node.save(graph_driver)

    node_count_before = await get_node_count(graph_driver, [survivor.uuid, dup.uuid])

    with pytest.raises(ValueError, match='group_id'):
        await g.merge_entities(
            survivor_uuid=survivor.uuid,
            duplicate_uuids=[dup.uuid],
        )

    # No writes happened
    node_count_after = await get_node_count(graph_driver, [survivor.uuid, dup.uuid])
    assert node_count_after == node_count_before

    await g.close()


# ---------------------------------------------------------------------------
# Survivor in duplicate_uuids raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_entities_survivor_in_duplicates(graph_driver, mock_embedder):
    """Passing survivor_uuid inside duplicate_uuids raises ValueError."""
    g = await _graphiti(graph_driver)

    node = _node('SelfMerge', ['Entity'])
    await node.save(graph_driver)

    with pytest.raises(ValueError):
        await g.merge_entities(
            survivor_uuid=node.uuid,
            duplicate_uuids=[node.uuid],
        )

    await g.close()


# ---------------------------------------------------------------------------
# Survivor not found raises NodeNotFoundError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_entities_survivor_not_found(graph_driver, mock_embedder):
    """Non-existent survivor UUID raises NodeNotFoundError."""
    g = await _graphiti(graph_driver)

    with pytest.raises(NodeNotFoundError):
        await g.merge_entities(
            survivor_uuid=str(uuid4()),
            duplicate_uuids=[str(uuid4())],
        )

    await g.close()


# ---------------------------------------------------------------------------
# Edge deduplication: colliding edges are removed, not duplicated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_entities_edge_dedup(graph_driver, mock_embedder):
    """Edges with identical (src, tgt, name, fact) after rewiring are deduplicated."""
    g = await _graphiti(graph_driver)

    survivor = _node('Alpha', ['Entity'])
    dup = _node('AlphaDup', ['Entity'])
    other = _node('Beta', ['Entity'])

    for node in [survivor, dup, other]:
        await node.save(graph_driver)

    # Both survivor and dup have an edge to 'other' with the same fact
    edge_survivor = _entity_edge(survivor.uuid, other.uuid, name='KNOWS', fact='shared fact')
    edge_dup = _entity_edge(dup.uuid, other.uuid, name='KNOWS', fact='shared fact')
    await edge_survivor.save(graph_driver)
    await edge_dup.save(graph_driver)

    result = await g.merge_entities(survivor_uuid=survivor.uuid, duplicate_uuids=[dup.uuid])

    # The colliding edge was dropped, not rewired
    assert result.edges_rewired == 0

    # Only one edge remains from survivor to other
    edges_after, _, _ = await graph_driver.execute_query(
        """
        MATCH (n:Entity {uuid: $survivor})-[e:RELATES_TO]-(m:Entity {uuid: $other})
        RETURN COUNT(e) AS cnt
        """,
        survivor=survivor.uuid,
        other=other.uuid,
        routing_='r',
    )
    assert int(edges_after[0]['cnt']) == 1

    await g.close()


# ---------------------------------------------------------------------------
# Batch partial failure: one bad merge doesn't block the others
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_entities_batch_partial_failure(graph_driver, mock_embedder):
    """A failed merge in a batch populates error; other merges still succeed."""
    g = await _graphiti(graph_driver)

    survivor = _node('GoodSvc', ['Entity'])
    dup = _node('GoodSvcDup', ['Entity'])
    await survivor.save(graph_driver)
    await dup.save(graph_driver)

    bad_survivor_uuid = str(uuid4())

    merges = [
        MergeRequest(survivor_uuid=bad_survivor_uuid, duplicate_uuids=[str(uuid4())]),
        MergeRequest(survivor_uuid=survivor.uuid, duplicate_uuids=[dup.uuid]),
    ]

    results = await g.merge_entities_batch(merges=merges)

    assert len(results) == 2
    # First merge failed (bad survivor)
    assert results[0].error is not None
    assert results[0].merged_count == 0
    # Second merge succeeded
    assert results[1].error is None
    assert results[1].merged_count == 1

    await g.close()
