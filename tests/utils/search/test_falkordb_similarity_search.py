"""Tests for FalkorDB similarity search branches in search_utils.py.

Note: FalkorDB node/edge similarity use brute-force cosine ranking (FalkorDB's
HNSW vector index is unreliable — see FalkorDB#716). Only community similarity
still uses the HNSW `db.idx.vector.queryNodes` path.
"""

from unittest.mock import AsyncMock, PropertyMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    _edge_embedding_cache,
    _node_embedding_cache,
    community_similarity_search,
    edge_similarity_search,
    node_similarity_search,
)


@pytest.fixture(autouse=True)
def _reset_bruteforce_caches():
    """Invalidate module-level embedding caches between tests."""
    _node_embedding_cache.invalidate()
    _edge_embedding_cache.invalidate()
    yield
    _node_embedding_cache.invalidate()
    _edge_embedding_cache.invalidate()


def _make_falkordb_driver():
    """Create a mock GraphDriver configured as FalkorDB."""
    driver = AsyncMock()
    type(driver).provider = PropertyMock(return_value=GraphProvider.FALKORDB)
    driver.search_interface = None
    driver.fulltext_syntax = '@'
    return driver


def _make_edge_record():
    """Create a mock edge record matching get_entity_edge_return_query output."""
    return {
        'uuid': 'edge-1',
        'group_id': 'group-1',
        'source_node_uuid': 'node-1',
        'target_node_uuid': 'node-2',
        'source_node_name': 'Alice',
        'target_node_name': 'Bob',
        'created_at': '2024-01-01T00:00:00',
        'expired_at': None,
        'valid_at': None,
        'invalid_at': None,
        'name': 'knows',
        'fact': 'Alice knows Bob',
        'fact_embedding': [0.1] * 768,
        'episodes': ['ep-1'],
        'source_name': 'test',
        'source_description': 'test source',
        'attributes': {},
    }


def _make_node_record():
    """Create a mock node record matching get_entity_node_return_query output."""
    return {
        'uuid': 'node-1',
        'group_id': 'group-1',
        'name': 'Alice',
        'name_embedding': [0.1] * 768,
        'labels': ['Entity'],
        'created_at': '2024-01-01T00:00:00',
        'summary': 'A person named Alice',
        'attributes': {},
    }


def _make_community_record():
    """Create a mock community record matching COMMUNITY_NODE_RETURN output."""
    return {
        'uuid': 'comm-1',
        'group_id': 'group-1',
        'name': 'Test Community',
        'name_embedding': [0.1] * 768,
        'created_at': '2024-01-01T00:00:00',
        'summary': 'A test community',
    }


class TestFalkorDBEdgeSimilaritySearch:
    """Tests for FalkorDB brute-force branch in edge_similarity_search.

    FalkorDB no longer uses HNSW for edge vector search — see the comment in
    search_utils.py referencing FalkorDB#716. The driver loads all edge
    embeddings into an in-process cache and ranks with vectorized cosine.
    """

    @pytest.mark.asyncio
    async def test_uses_brute_force_match_query(self):
        """Test that FalkorDB loads edges via MATCH and does NOT use HNSW index."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([_make_edge_record()], ['uuid'], None)

        search_vector = [0.1] * 768
        await edge_similarity_search(
            driver,
            search_vector,
            source_node_uuid=None,
            target_node_uuid=None,
            search_filter=SearchFilters(),
            group_ids=['group-1'],
        )

        driver.execute_query.assert_called_once()
        query = driver.execute_query.call_args[0][0]
        assert 'MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)' in query
        assert 'fact_embedding' in query
        assert 'db.idx.vector' not in query  # no HNSW

    @pytest.mark.asyncio
    async def test_caches_all_edges_in_unscoped_search(self):
        """Unscoped search loads edges once into the module-level cache and reuses them."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([_make_edge_record()], ['uuid'], None)

        search_vector = [0.1] * 768
        for _ in range(2):
            await edge_similarity_search(
                driver,
                search_vector,
                source_node_uuid=None,
                target_node_uuid=None,
                search_filter=SearchFilters(),
                group_ids=['group-1'],
            )

        # The second call reuses the cache — DB should be hit only once.
        assert driver.execute_query.call_count == 1

    @pytest.mark.asyncio
    async def test_scoped_search_filters_by_edge_uuids(self):
        """Scoped search (edge_uuids set) applies the uuid filter in the query."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([_make_edge_record()], ['uuid'], None)

        search_vector = [0.1] * 768
        edge_uuids = ['edge-1', 'edge-2']
        await edge_similarity_search(
            driver,
            search_vector,
            source_node_uuid=None,
            target_node_uuid=None,
            search_filter=SearchFilters(edge_uuids=edge_uuids),
            group_ids=['group-1'],
        )

        query = driver.execute_query.call_args[0][0]
        assert 'e.uuid IN $edge_uuids' in query
        call_kwargs = driver.execute_query.call_args[1]
        assert call_kwargs['edge_uuids'] == edge_uuids


class TestFalkorDBNodeSimilaritySearch:
    """Tests for FalkorDB brute-force branch in node_similarity_search.

    Like edges, FalkorDB node similarity is brute-force cosine ranked from an
    in-process cache (FalkorDB#716 — the HNSW index is unreliable for nodes too).
    """

    @pytest.mark.asyncio
    async def test_uses_brute_force_match_query(self):
        """Test that FalkorDB loads nodes via MATCH and does NOT use HNSW index."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([_make_node_record()], ['uuid'], None)

        search_vector = [0.1] * 768
        await node_similarity_search(
            driver,
            search_vector,
            search_filter=SearchFilters(),
            group_ids=['group-1'],
        )

        driver.execute_query.assert_called_once()
        query = driver.execute_query.call_args[0][0]
        assert 'MATCH (n:Entity)' in query
        assert 'name_embedding' in query
        assert 'db.idx.vector' not in query  # no HNSW

    @pytest.mark.asyncio
    async def test_caches_all_nodes_between_calls(self):
        """Repeated calls reuse the module-level node embedding cache."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([_make_node_record()], ['uuid'], None)

        search_vector = [0.1] * 768
        for _ in range(2):
            await node_similarity_search(
                driver,
                search_vector,
                search_filter=SearchFilters(),
                group_ids=['group-1'],
            )

        # The second call reuses the cache — DB should be hit only once.
        assert driver.execute_query.call_count == 1

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_no_results(self):
        """Test that empty results are handled correctly."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([], [], None)

        search_vector = [0.1] * 768
        results = await node_similarity_search(
            driver,
            search_vector,
            search_filter=SearchFilters(),
            group_ids=['group-1'],
        )

        assert results == []


class TestFalkorDBCommunitySimilaritySearch:
    """Tests for FalkorDB HNSW branch in community_similarity_search."""

    @pytest.mark.asyncio
    async def test_uses_hnsw_index_query(self):
        """Test that FalkorDB branch uses db.idx.vector.queryNodes for Community."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([_make_community_record()], ['uuid'], None)

        search_vector = [0.1] * 768
        await community_similarity_search(
            driver,
            search_vector,
            group_ids=['group-1'],
        )

        driver.execute_query.assert_called_once()
        query = driver.execute_query.call_args[0][0]
        assert 'db.idx.vector.queryNodes' in query
        assert "'Community'" in query
        assert 'name_embedding' in query

    @pytest.mark.asyncio
    async def test_applies_group_id_filter(self):
        """Test that group_id filter is applied."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([], [], None)

        search_vector = [0.1] * 768
        await community_similarity_search(
            driver,
            search_vector,
            group_ids=['group-1'],
        )

        query = driver.execute_query.call_args[0][0]
        assert 'group_id' in query

    @pytest.mark.asyncio
    async def test_over_fetches_for_post_filtering(self):
        """Test that the over-fetch limit is passed."""
        driver = _make_falkordb_driver()
        driver.execute_query.return_value = ([], [], None)

        search_vector = [0.1] * 768
        limit = 8
        await community_similarity_search(
            driver,
            search_vector,
            group_ids=['group-1'],
            limit=limit,
        )

        call_kwargs = driver.execute_query.call_args[1]
        assert call_kwargs['over_fetch_limit'] == limit * 10
