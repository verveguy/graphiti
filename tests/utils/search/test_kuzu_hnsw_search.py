"""Tests for Kuzu HNSW vector index search branches in search_utils.py."""

from unittest.mock import AsyncMock, PropertyMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import node_similarity_search


def _make_kuzu_driver():
    """Create a mock GraphDriver configured as Kuzu."""
    driver = AsyncMock()
    type(driver).provider = PropertyMock(return_value=GraphProvider.KUZU)
    driver.search_interface = None
    return driver


def _make_node_record():
    """Create a mock node record matching get_entity_node_return_query (Kuzu) output."""
    return {
        'uuid': 'node-1',
        'group_id': 'group-1',
        'name': 'Alice',
        'labels': ['Entity'],
        'created_at': '2024-01-01T00:00:00',
        'summary': 'A person named Alice',
        'attributes': '{}',
    }


class TestKuzuNodeSimilaritySearch:
    """Tests for Kuzu HNSW branch in node_similarity_search."""

    @pytest.mark.asyncio
    async def test_uses_hnsw_vector_index_query(self):
        """Test that Kuzu branch uses QUERY_VECTOR_INDEX for Entity."""
        driver = _make_kuzu_driver()
        driver.execute_query.return_value = ([_make_node_record()], ['uuid'], None)

        search_vector = [0.1] * 768
        results = await node_similarity_search(
            driver,
            search_vector,
            search_filter=SearchFilters(),
            group_ids=['group-1'],
        )

        driver.execute_query.assert_called_once()
        query = driver.execute_query.call_args[0][0]
        assert 'QUERY_VECTOR_INDEX' in query
        assert "'Entity'" in query
        assert 'entity_name_embedding_idx' in query
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_converts_distance_to_similarity(self):
        """Test that the query converts distance to similarity via (1.0 - distance)."""
        driver = _make_kuzu_driver()
        driver.execute_query.return_value = ([], [], None)

        search_vector = [0.1] * 768
        await node_similarity_search(
            driver,
            search_vector,
            search_filter=SearchFilters(),
        )

        query = driver.execute_query.call_args[0][0]
        assert '1.0 - distance' in query

    @pytest.mark.asyncio
    async def test_over_fetches_for_post_filtering(self):
        """Test that the over-fetch limit (10x) is passed."""
        driver = _make_kuzu_driver()
        driver.execute_query.return_value = ([], [], None)

        search_vector = [0.1] * 768
        limit = 5
        await node_similarity_search(
            driver,
            search_vector,
            search_filter=SearchFilters(),
            limit=limit,
        )

        call_kwargs = driver.execute_query.call_args[1]
        assert call_kwargs['over_fetch_limit'] == limit * 10

    @pytest.mark.asyncio
    async def test_applies_group_id_filter(self):
        """Test that group_id filter is applied as a post-filter."""
        driver = _make_kuzu_driver()
        driver.execute_query.return_value = ([], [], None)

        search_vector = [0.1] * 768
        await node_similarity_search(
            driver,
            search_vector,
            search_filter=SearchFilters(),
            group_ids=['group-1'],
        )

        query = driver.execute_query.call_args[0][0]
        assert 'group_id' in query

    @pytest.mark.asyncio
    async def test_applies_min_score_filter(self):
        """Test that min_score filter is included in the query."""
        driver = _make_kuzu_driver()
        driver.execute_query.return_value = ([], [], None)

        search_vector = [0.1] * 768
        await node_similarity_search(
            driver,
            search_vector,
            search_filter=SearchFilters(),
            min_score=0.7,
        )

        query = driver.execute_query.call_args[0][0]
        assert 'score > $min_score' in query
        call_kwargs = driver.execute_query.call_args[1]
        assert call_kwargs['min_score'] == 0.7

    @pytest.mark.asyncio
    async def test_casts_search_vector_with_dimension(self):
        """Test that the search vector is CAST to the correct FLOAT[N] dimension."""
        driver = _make_kuzu_driver()
        driver.execute_query.return_value = ([], [], None)

        search_vector = [0.1] * 768
        await node_similarity_search(
            driver,
            search_vector,
            search_filter=SearchFilters(),
        )

        query = driver.execute_query.call_args[0][0]
        assert 'FLOAT[768]' in query

    @pytest.mark.asyncio
    async def test_falls_back_to_brute_force_on_error(self):
        """Test that HNSW failure falls back to array_cosine_similarity brute-force."""
        driver = _make_kuzu_driver()
        # First call (HNSW) raises, second call (brute-force) succeeds
        driver.execute_query.side_effect = [
            RuntimeError('index not found'),
            ([_make_node_record()], ['uuid'], None),
        ]

        search_vector = [0.1] * 768
        results = await node_similarity_search(
            driver,
            search_vector,
            search_filter=SearchFilters(),
            group_ids=['group-1'],
        )

        assert driver.execute_query.call_count == 2
        # Second call should be brute-force (array_cosine_similarity)
        fallback_query = driver.execute_query.call_args_list[1][0][0]
        assert 'array_cosine_similarity' in fallback_query
        assert 'MATCH (n:Entity)' in fallback_query
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_no_results(self):
        """Test that empty results are handled correctly."""
        driver = _make_kuzu_driver()
        driver.execute_query.return_value = ([], [], None)

        search_vector = [0.1] * 768
        results = await node_similarity_search(
            driver,
            search_vector,
            search_filter=SearchFilters(),
        )

        assert results == []
