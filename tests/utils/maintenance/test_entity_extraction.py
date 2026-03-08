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
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.node_operations import (
    _build_entity_types_context,
    _extract_entity_summaries_batch,
    _sanitize_label,
    extract_nodes,
    reclassify_entity,
    reprocess_entity_types,
)


def _make_clients():
    """Create mock GraphitiClients for testing."""
    driver = MagicMock()
    embedder = MagicMock()
    cross_encoder = MagicMock()
    llm_client = MagicMock()
    llm_generate = AsyncMock()
    llm_client.generate_response = llm_generate

    clients = GraphitiClients.model_construct(  # bypass validation to allow test doubles
        driver=driver,
        embedder=embedder,
        cross_encoder=cross_encoder,
        llm_client=llm_client,
    )

    return clients, llm_generate


def _make_episode(
    content: str = 'Test content',
    source: EpisodeType = EpisodeType.text,
    group_id: str = 'group',
) -> EpisodicNode:
    """Create a test episode node."""
    return EpisodicNode(
        name='test_episode',
        group_id=group_id,
        source=source,
        source_description='test',
        content=content,
        valid_at=utc_now(),
    )


class TestExtractNodesSmallInput:
    @pytest.mark.asyncio
    async def test_small_input_single_llm_call(self, monkeypatch):
        """Small inputs should use a single LLM call without chunking.

        When no entity_types are provided, freeform mode is used,
        so mock responses use entity_type (str) instead of entity_type_id (int).
        """
        clients, llm_generate = _make_clients()

        # Mock LLM response (freeform mode — no entity_types provided)
        llm_generate.return_value = {
            'extracted_entities': [
                {'name': 'Alice', 'entity_type': 'Person'},
                {'name': 'Bob', 'entity_type': 'Person'},
            ]
        }

        # Small content (below threshold)
        episode = _make_episode(content='Alice talked to Bob.')

        nodes = await extract_nodes(
            clients,
            episode,
            previous_episodes=[],
        )

        # Verify results
        assert len(nodes) == 2
        assert {n.name for n in nodes} == {'Alice', 'Bob'}
        # Both should have Person label from freeform classification
        for node in nodes:
            assert 'Person' in node.labels
            assert 'Entity' in node.labels

        # LLM should be called exactly once
        llm_generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_extracts_entity_types(self, monkeypatch):
        """Entity type classification should work correctly."""
        clients, llm_generate = _make_clients()

        from pydantic import BaseModel

        class Person(BaseModel):
            """A human person."""

            pass

        llm_generate.return_value = {
            'extracted_entities': [
                {'name': 'Alice', 'entity_type_id': 1},  # Person
                {'name': 'Acme Corp', 'entity_type_id': 0},  # Default Entity
            ]
        }

        episode = _make_episode(content='Alice works at Acme Corp.')

        nodes = await extract_nodes(
            clients,
            episode,
            previous_episodes=[],
            entity_types={'Person': Person},
        )

        # Alice should have Person label
        alice = next(n for n in nodes if n.name == 'Alice')
        assert 'Person' in alice.labels

        # Acme should have Entity label
        acme = next(n for n in nodes if n.name == 'Acme Corp')
        assert 'Entity' in acme.labels

    @pytest.mark.asyncio
    async def test_excludes_entity_types(self, monkeypatch):
        """Excluded entity types should not appear in results."""
        clients, llm_generate = _make_clients()

        from pydantic import BaseModel

        class User(BaseModel):
            """A user of the system."""

            pass

        llm_generate.return_value = {
            'extracted_entities': [
                {'name': 'Alice', 'entity_type_id': 1},  # User (excluded)
                {'name': 'Project X', 'entity_type_id': 0},  # Entity
            ]
        }

        episode = _make_episode(content='Alice created Project X.')

        nodes = await extract_nodes(
            clients,
            episode,
            previous_episodes=[],
            entity_types={'User': User},
            excluded_entity_types=['User'],
        )

        # Alice should be excluded
        assert len(nodes) == 1
        assert nodes[0].name == 'Project X'

    @pytest.mark.asyncio
    async def test_filters_empty_names(self, monkeypatch):
        """Entities with empty names should be filtered out."""
        clients, llm_generate = _make_clients()

        # Freeform mode (no entity_types provided)
        llm_generate.return_value = {
            'extracted_entities': [
                {'name': 'Alice', 'entity_type': 'Person'},
                {'name': '', 'entity_type': 'Person'},
                {'name': '   ', 'entity_type': 'Concept'},
            ]
        }

        episode = _make_episode(content='Alice is here.')

        nodes = await extract_nodes(
            clients,
            episode,
            previous_episodes=[],
        )

        assert len(nodes) == 1
        assert nodes[0].name == 'Alice'


class TestExtractNodesPromptSelection:
    @pytest.mark.asyncio
    async def test_uses_text_prompt_for_text_episodes(self, monkeypatch):
        """Text episodes should use extract_text prompt."""
        clients, llm_generate = _make_clients()
        llm_generate.return_value = {'extracted_entities': []}

        episode = _make_episode(source=EpisodeType.text)

        await extract_nodes(clients, episode, previous_episodes=[])

        # Check prompt_name parameter
        call_kwargs = llm_generate.call_args[1]
        assert call_kwargs.get('prompt_name') == 'extract_nodes.extract_text'

    @pytest.mark.asyncio
    async def test_uses_json_prompt_for_json_episodes(self, monkeypatch):
        """JSON episodes should use extract_json prompt."""
        clients, llm_generate = _make_clients()
        llm_generate.return_value = {'extracted_entities': []}

        episode = _make_episode(content='{}', source=EpisodeType.json)

        await extract_nodes(clients, episode, previous_episodes=[])

        call_kwargs = llm_generate.call_args[1]
        assert call_kwargs.get('prompt_name') == 'extract_nodes.extract_json'

    @pytest.mark.asyncio
    async def test_uses_message_prompt_for_message_episodes(self, monkeypatch):
        """Message episodes should use extract_message prompt."""
        clients, llm_generate = _make_clients()
        llm_generate.return_value = {'extracted_entities': []}

        episode = _make_episode(source=EpisodeType.message)

        await extract_nodes(clients, episode, previous_episodes=[])

        call_kwargs = llm_generate.call_args[1]
        assert call_kwargs.get('prompt_name') == 'extract_nodes.extract_message'

    @pytest.mark.asyncio
    async def test_freeform_uses_correct_response_model(self, monkeypatch):
        """Freeform mode should use ExtractedEntitiesFreeform response model."""
        from graphiti_core.prompts.extract_nodes import ExtractedEntitiesFreeform

        clients, llm_generate = _make_clients()
        llm_generate.return_value = {'extracted_entities': []}

        episode = _make_episode(source=EpisodeType.text)

        # No entity_types = freeform mode
        await extract_nodes(clients, episode, previous_episodes=[])

        call_kwargs = llm_generate.call_args[1]
        assert call_kwargs.get('response_model') is ExtractedEntitiesFreeform

    @pytest.mark.asyncio
    async def test_constrained_uses_correct_response_model(self, monkeypatch):
        """Constrained mode should use ExtractedEntities response model."""
        from pydantic import BaseModel

        from graphiti_core.prompts.extract_nodes import ExtractedEntities

        class Person(BaseModel):
            """A human person."""

            pass

        clients, llm_generate = _make_clients()
        llm_generate.return_value = {'extracted_entities': []}

        episode = _make_episode(source=EpisodeType.text)

        await extract_nodes(
            clients, episode, previous_episodes=[], entity_types={'Person': Person}
        )

        call_kwargs = llm_generate.call_args[1]
        assert call_kwargs.get('response_model') is ExtractedEntities


class TestBuildEntityTypesContext:
    def test_default_entity_type_always_included(self):
        """Default Entity type should always be at index 0."""
        context = _build_entity_types_context(None)

        assert len(context) == 1
        assert context[0]['entity_type_id'] == 0
        assert context[0]['entity_type_name'] == 'Entity'

    def test_custom_types_added_after_default(self):
        """Custom entity types should be added with sequential IDs."""
        from pydantic import BaseModel

        class Person(BaseModel):
            """A human person."""

            pass

        class Organization(BaseModel):
            """A business or organization."""

            pass

        context = _build_entity_types_context(
            {
                'Person': Person,
                'Organization': Organization,
            }
        )

        assert len(context) == 3
        assert context[0]['entity_type_name'] == 'Entity'
        assert context[1]['entity_type_name'] == 'Person'
        assert context[1]['entity_type_id'] == 1
        assert context[2]['entity_type_name'] == 'Organization'
        assert context[2]['entity_type_id'] == 2


def _make_entity_node(
    name: str,
    summary: str = '',
    group_id: str = 'group',
    uuid: str | None = None,
) -> EntityNode:
    """Create a test entity node."""
    node = EntityNode(
        name=name,
        group_id=group_id,
        labels=['Entity'],
        summary=summary,
        created_at=utc_now(),
    )
    if uuid is not None:
        node.uuid = uuid
    return node


def _make_entity_edge(
    source_uuid: str,
    target_uuid: str,
    fact: str,
) -> EntityEdge:
    """Create a test entity edge."""
    return EntityEdge(
        source_node_uuid=source_uuid,
        target_node_uuid=target_uuid,
        name='TEST_RELATION',
        fact=fact,
        group_id='group',
        created_at=utc_now(),
    )


class TestExtractEntitySummariesBatch:
    @pytest.mark.asyncio
    async def test_no_nodes_needing_summarization(self):
        """When no nodes need summarization, no LLM call should be made."""
        llm_client = MagicMock()
        llm_generate = AsyncMock()
        llm_client.generate_response = llm_generate

        # Node with short summary that doesn't need LLM
        node = _make_entity_node('Alice', summary='Alice is a person.')
        nodes = [node]

        await _extract_entity_summaries_batch(
            llm_client,
            nodes,
            episode=None,
            previous_episodes=None,
            should_summarize_node=None,
            edges_by_node={},
        )

        # LLM should not be called
        llm_generate.assert_not_awaited()
        # Summary should remain unchanged
        assert nodes[0].summary == 'Alice is a person.'

    @pytest.mark.asyncio
    async def test_short_summary_with_edge_facts(self):
        """Nodes with short summaries should have edge facts appended without LLM."""
        llm_client = MagicMock()
        llm_generate = AsyncMock()
        llm_client.generate_response = llm_generate

        node = _make_entity_node('Alice', summary='Alice is a person.', uuid='alice-uuid')
        edge = _make_entity_edge('alice-uuid', 'bob-uuid', 'Alice works with Bob.')

        edges_by_node = {
            'alice-uuid': [edge],
        }

        await _extract_entity_summaries_batch(
            llm_client,
            [node],
            episode=None,
            previous_episodes=None,
            should_summarize_node=None,
            edges_by_node=edges_by_node,
        )

        # LLM should not be called
        llm_generate.assert_not_awaited()
        # Summary should include edge fact
        assert 'Alice is a person.' in node.summary
        assert 'Alice works with Bob.' in node.summary

    @pytest.mark.asyncio
    async def test_long_summary_needs_llm(self):
        """Nodes with long summaries should trigger LLM summarization."""
        llm_client = MagicMock()
        llm_generate = AsyncMock()
        llm_generate.return_value = {
            'summaries': [
                {'name': 'Alice', 'summary': 'Alice is a software engineer at Acme Corp.'}
            ]
        }
        llm_client.generate_response = llm_generate

        # Create a node with a very long summary (over MAX_SUMMARY_CHARS * 4)
        long_summary = 'Alice is a person. ' * 200  # ~3800 chars
        node = _make_entity_node('Alice', summary=long_summary)

        await _extract_entity_summaries_batch(
            llm_client,
            [node],
            episode=_make_episode(),
            previous_episodes=[],
            should_summarize_node=None,
            edges_by_node={},
        )

        # LLM should be called
        llm_generate.assert_awaited_once()
        # Summary should be updated from LLM response
        assert node.summary == 'Alice is a software engineer at Acme Corp.'

    @pytest.mark.asyncio
    async def test_should_summarize_filter(self):
        """Nodes filtered by should_summarize_node should be skipped."""
        llm_client = MagicMock()
        llm_generate = AsyncMock()
        llm_client.generate_response = llm_generate

        node = _make_entity_node('Alice', summary='')

        # Filter that rejects all nodes
        async def reject_all(n):
            return False

        await _extract_entity_summaries_batch(
            llm_client,
            [node],
            episode=_make_episode(),
            previous_episodes=[],
            should_summarize_node=reject_all,
            edges_by_node={},
        )

        # LLM should not be called
        llm_generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_batch_multiple_nodes(self):
        """Multiple nodes needing summarization should be batched into one call."""
        llm_client = MagicMock()
        llm_generate = AsyncMock()
        llm_generate.return_value = {
            'summaries': [
                {'name': 'Alice', 'summary': 'Alice summary.'},
                {'name': 'Bob', 'summary': 'Bob summary.'},
            ]
        }
        llm_client.generate_response = llm_generate

        # Create nodes with long summaries
        long_summary = 'X ' * 1500  # Long enough to need LLM
        alice = _make_entity_node('Alice', summary=long_summary)
        bob = _make_entity_node('Bob', summary=long_summary)

        await _extract_entity_summaries_batch(
            llm_client,
            [alice, bob],
            episode=_make_episode(),
            previous_episodes=[],
            should_summarize_node=None,
            edges_by_node={},
        )

        # LLM should be called exactly once (batch call)
        llm_generate.assert_awaited_once()
        # Both nodes should have updated summaries
        assert alice.summary == 'Alice summary.'
        assert bob.summary == 'Bob summary.'

    @pytest.mark.asyncio
    async def test_unknown_entity_in_response(self):
        """LLM returning unknown entity names should be logged but not crash."""
        llm_client = MagicMock()
        llm_generate = AsyncMock()
        llm_generate.return_value = {
            'summaries': [
                {'name': 'UnknownEntity', 'summary': 'Should be ignored.'},
                {'name': 'Alice', 'summary': 'Alice summary.'},
            ]
        }
        llm_client.generate_response = llm_generate

        long_summary = 'X ' * 1500
        alice = _make_entity_node('Alice', summary=long_summary)

        await _extract_entity_summaries_batch(
            llm_client,
            [alice],
            episode=_make_episode(),
            previous_episodes=[],
            should_summarize_node=None,
            edges_by_node={},
        )

        # Alice should have updated summary
        assert alice.summary == 'Alice summary.'

    @pytest.mark.asyncio
    async def test_no_episode_and_no_summary(self):
        """Nodes with no summary and no episode should be skipped."""
        llm_client = MagicMock()
        llm_generate = AsyncMock()
        llm_client.generate_response = llm_generate

        node = _make_entity_node('Alice', summary='')

        await _extract_entity_summaries_batch(
            llm_client,
            [node],
            episode=None,
            previous_episodes=None,
            should_summarize_node=None,
            edges_by_node={},
        )

        # LLM should not be called - no content to summarize
        llm_generate.assert_not_awaited()
        assert node.summary == ''

    @pytest.mark.asyncio
    async def test_flight_partitioning(self, monkeypatch):
        """Nodes should be partitioned into flights of MAX_NODES."""
        # Set MAX_NODES to a small value for testing
        monkeypatch.setattr('graphiti_core.utils.maintenance.node_operations.MAX_NODES', 2)

        llm_client = MagicMock()
        call_count = 0
        call_args_list = []

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Extract entity names from the context
            context = args[0][1].content if args else ''
            call_args_list.append(context)
            return {'summaries': []}

        llm_client.generate_response = mock_generate

        # Create 5 nodes with long summaries (need LLM)
        long_summary = 'X ' * 1500
        nodes = [_make_entity_node(f'Entity{i}', summary=long_summary) for i in range(5)]

        await _extract_entity_summaries_batch(
            llm_client,
            nodes,
            episode=_make_episode(),
            previous_episodes=[],
            should_summarize_node=None,
            edges_by_node={},
        )

        # With MAX_NODES=2 and 5 nodes, we should have 3 flights (2+2+1)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_case_insensitive_name_matching(self):
        """LLM response names should match case-insensitively."""
        llm_client = MagicMock()
        llm_generate = AsyncMock()
        # LLM returns name with different casing
        llm_generate.return_value = {
            'summaries': [
                {'name': 'ALICE', 'summary': 'Alice summary from LLM.'},
            ]
        }
        llm_client.generate_response = llm_generate

        # Node has lowercase name
        long_summary = 'X ' * 1500
        node = _make_entity_node('alice', summary=long_summary)

        await _extract_entity_summaries_batch(
            llm_client,
            [node],
            episode=_make_episode(),
            previous_episodes=[],
            should_summarize_node=None,
            edges_by_node={},
        )

        # Should match despite case difference
        assert node.summary == 'Alice summary from LLM.'


class TestFreeformEntityExtraction:
    @pytest.mark.asyncio
    async def test_freeform_assigns_semantic_labels(self):
        """Freeform mode should assign LLM-provided type labels to entities."""
        clients, llm_generate = _make_clients()

        llm_generate.return_value = {
            'extracted_entities': [
                {'name': 'Alice', 'entity_type': 'Person'},
                {'name': 'Acme Corp', 'entity_type': 'Organization'},
                {'name': 'Python', 'entity_type': 'Software'},
            ]
        }

        episode = _make_episode(content='Alice works at Acme Corp using Python.')

        nodes = await extract_nodes(clients, episode, previous_episodes=[])

        assert len(nodes) == 3
        alice = next(n for n in nodes if n.name == 'Alice')
        assert 'Person' in alice.labels
        assert 'Entity' in alice.labels

        acme = next(n for n in nodes if n.name == 'Acme Corp')
        assert 'Organization' in acme.labels

        python = next(n for n in nodes if n.name == 'Python')
        assert 'Software' in python.labels

    @pytest.mark.asyncio
    async def test_freeform_excludes_entity_types(self):
        """Freeform mode should support excluded_entity_types."""
        clients, llm_generate = _make_clients()

        llm_generate.return_value = {
            'extracted_entities': [
                {'name': 'Alice', 'entity_type': 'Person'},
                {'name': 'Python', 'entity_type': 'Software'},
            ]
        }

        episode = _make_episode(content='Alice uses Python.')

        nodes = await extract_nodes(
            clients,
            episode,
            previous_episodes=[],
            excluded_entity_types=['Person'],
        )

        assert len(nodes) == 1
        assert nodes[0].name == 'Python'

    @pytest.mark.asyncio
    async def test_freeform_empty_type_falls_back_to_entity(self):
        """Empty or whitespace entity_type should fall back to 'Entity'."""
        clients, llm_generate = _make_clients()

        llm_generate.return_value = {
            'extracted_entities': [
                {'name': 'Alice', 'entity_type': ''},
                {'name': 'Bob', 'entity_type': '   '},
            ]
        }

        episode = _make_episode(content='Alice and Bob.')

        nodes = await extract_nodes(clients, episode, previous_episodes=[])

        assert len(nodes) == 2
        for node in nodes:
            assert node.labels == ['Entity']


class TestSanitizeLabel:
    def test_basic_pascal_case(self):
        assert _sanitize_label('Person') == 'Person'

    def test_lowercase_capitalizes_first(self):
        assert _sanitize_label('person') == 'Person'

    def test_strips_special_characters(self):
        assert _sanitize_label('My-Type!') == 'Mytype'

    def test_strips_spaces(self):
        assert _sanitize_label('My Type') == 'Mytype'

    def test_digit_prefix_gets_label_prepended(self):
        assert _sanitize_label('123Type') == 'Label123type'

    def test_empty_string_returns_entity(self):
        assert _sanitize_label('') == 'Entity'

    def test_all_special_chars_returns_entity(self):
        assert _sanitize_label('!@#$%') == 'Entity'

    def test_normalizes_underscored_parts_to_pascal_case(self):
        assert _sanitize_label('My_Type') == 'MyType'

    def test_normalizes_all_caps(self):
        assert _sanitize_label('PERSON') == 'Person'

    def test_normalizes_all_caps_with_underscores(self):
        assert _sanitize_label('PERSON_NAME') == 'PersonName'

    def test_underscore_only_returns_entity(self):
        assert _sanitize_label('_') == 'Entity'
        assert _sanitize_label('__') == 'Entity'


class TestReclassifyEntity:
    @pytest.mark.asyncio
    async def test_returns_sanitized_type(self):
        """reclassify_entity should return the sanitized entity type from LLM."""
        llm_client = MagicMock()
        llm_generate = AsyncMock(return_value={'entity_type': 'Person'})
        llm_client.generate_response = llm_generate

        entity = _make_entity_node('Alice', summary='Alice is a software engineer.')

        result = await reclassify_entity(llm_client, entity)

        assert result == 'Person'
        llm_generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sanitizes_llm_response(self):
        """reclassify_entity should sanitize unusual LLM responses."""
        llm_client = MagicMock()
        llm_generate = AsyncMock(return_value={'entity_type': 'my-custom type!'})
        llm_client.generate_response = llm_generate

        entity = _make_entity_node('Test', summary='A test entity.')

        result = await reclassify_entity(llm_client, entity)

        assert result == 'Mycustomtype'

    @pytest.mark.asyncio
    async def test_passes_name_and_summary_to_prompt(self):
        """reclassify_entity should pass entity name and summary as context."""
        llm_client = MagicMock()
        llm_generate = AsyncMock(return_value={'entity_type': 'Person'})
        llm_client.generate_response = llm_generate

        entity = _make_entity_node('Alice', summary='Alice is the CEO of Acme Corp.')

        await reclassify_entity(llm_client, entity)

        # Verify the prompt was called (the first positional arg is the prompt messages)
        call_args = llm_generate.call_args
        prompt_messages = call_args[0][0]
        # The user message should contain the entity name and summary
        user_content = prompt_messages[1].content
        assert 'Alice' in user_content
        assert 'Alice is the CEO of Acme Corp.' in user_content


class TestReprocessEntityTypes:
    @pytest.mark.asyncio
    async def test_reclassifies_untyped_entities(self):
        """Should reclassify entities with only ['Entity'] labels."""
        clients, llm_generate = _make_clients()
        llm_generate.return_value = {'entity_type': 'Person'}

        entity = _make_entity_node('Alice', summary='Alice is a person.')
        mock_save = AsyncMock()

        with pytest.MonkeyPatch.context() as m:
            m.setattr(EntityNode, 'get_by_group_ids', AsyncMock(return_value=[entity]))
            m.setattr(EntityNode, 'save', mock_save)

            result = await reprocess_entity_types(clients, 'group')

        assert len(result) == 1
        assert 'Person' in result[0].labels
        assert 'Entity' in result[0].labels
        mock_save.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_already_typed_entities(self):
        """Entities with semantic type labels should not be re-processed."""
        clients, llm_generate = _make_clients()

        typed_entity = _make_entity_node('Alice', summary='Alice is a person.')
        typed_entity.labels = ['Entity', 'Person']

        with pytest.MonkeyPatch.context() as m:
            m.setattr(EntityNode, 'get_by_group_ids', AsyncMock(return_value=[typed_entity]))

            result = await reprocess_entity_types(clients, 'group')

        assert len(result) == 0
        # LLM should not have been called
        llm_generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_entity_type_fallback(self):
        """Entities where LLM returns 'Entity' should not have labels updated."""
        clients, llm_generate = _make_clients()
        llm_generate.return_value = {'entity_type': 'Entity'}

        entity = _make_entity_node('Unknown Thing', summary='Something generic.')
        mock_save = AsyncMock()

        with pytest.MonkeyPatch.context() as m:
            m.setattr(EntityNode, 'get_by_group_ids', AsyncMock(return_value=[entity]))
            m.setattr(EntityNode, 'save', mock_save)

            result = await reprocess_entity_types(clients, 'group')

        assert len(result) == 0
        mock_save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_group_returns_empty(self):
        """Empty group should return empty list without LLM calls."""
        clients, llm_generate = _make_clients()

        with pytest.MonkeyPatch.context() as m:
            m.setattr(EntityNode, 'get_by_group_ids', AsyncMock(return_value=[]))

            result = await reprocess_entity_types(clients, 'group')

        assert result == []
        llm_generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_logs_progress(self, caplog):
        """Should log progress during reclassification."""
        clients, llm_generate = _make_clients()
        llm_generate.return_value = {'entity_type': 'Person'}

        entity = _make_entity_node('Alice', summary='Alice is a person.')
        mock_save = AsyncMock()

        with (
            pytest.MonkeyPatch.context() as m,
            caplog.at_level(logging.INFO),
        ):
            m.setattr(EntityNode, 'get_by_group_ids', AsyncMock(return_value=[entity]))
            m.setattr(EntityNode, 'save', mock_save)

            await reprocess_entity_types(clients, 'group')

        log_messages = [r.message for r in caplog.records]
        assert any('Reclassified 1/1' in msg for msg in log_messages)
        assert any('Alice' in msg for msg in log_messages)
