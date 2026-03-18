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
import re
from collections.abc import Awaitable, Callable
from time import time
from typing import Any, cast

from pydantic import BaseModel

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.driver.record_parsers import entity_node_from_record
from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.models.nodes.node_db_queries import get_entity_node_return_query
from graphiti_core.nodes import (
    EntityNode,
    EpisodeType,
    EpisodicNode,
    create_entity_node_embeddings,
)
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_nodes import NodeDuplicate, NodeResolutions
from graphiti_core.prompts.extract_nodes import (
    ExtractedEntities,
    ExtractedEntitiesFreeform,
    ExtractedEntity,
    ExtractedEntityFreeform,
    ReclassifiedEntity,
    SummarizedEntities,
)
from graphiti_core.search.search import search
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF_DEDUP
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupCandidateIndexes,
    DedupResolutionState,
    _build_candidate_indexes,
    _resolve_with_similarity,
)
from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS, truncate_at_sentence

logger = logging.getLogger(__name__)

# Maximum number of nodes to summarize in a single LLM call
MAX_NODES = 30

NodeSummaryFilter = Callable[[EntityNode], Awaitable[bool]]


async def extract_nodes(
    clients: GraphitiClients,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    custom_extraction_instructions: str | None = None,
) -> list[EntityNode]:
    """Extract entity nodes from an episode."""
    start = time()
    llm_client = clients.llm_client

    # Determine if we should use freeform entity type classification.
    # When no custom entity_types are provided, we use freeform mode where
    # the LLM assigns semantic type labels (e.g., Person, Software, Concept)
    # rather than being constrained to a single generic "Entity" type.
    use_freeform = entity_types is None

    # Build entity types context (only used in non-freeform mode)
    entity_types_context = _build_entity_types_context(entity_types)

    # Build base context
    context: dict[str, Any] = {
        'episode_content': episode.content,
        'episode_timestamp': episode.valid_at.isoformat(),
        'previous_episodes': [ep.content for ep in previous_episodes],
        'custom_extraction_instructions': custom_extraction_instructions or '',
        'entity_types': entity_types_context,
        'source_description': episode.source_description,
        'freeform_entity_types': use_freeform,
    }

    # Extract entities
    if use_freeform:
        raw_entities = await _extract_nodes_single_freeform(llm_client, episode, context)
        log_suffix = ' (freeform)'
    else:
        raw_entities = await _extract_nodes_single(llm_client, episode, context)
        log_suffix = ''

    # Filter empty names and file-path entities (e.g. "ingestion/foo.docx").
    # File paths contain a slash followed by a filename with a dot-extension.
    # URLs (http://..., autodesk.com) are kept since they can be legitimate entities.
    _FILE_PATH_RE = re.compile(r'[\\/][^/\\]+\.\w{1,5}$')

    def _is_file_path(name: str) -> bool:
        if '://' in name:
            return False  # URL, not a file path
        return bool(_FILE_PATH_RE.search(name))

    filtered_entities = [
        e for e in raw_entities
        if e.name.strip() and not _is_file_path(e.name)
    ]

    end = time()
    logger.debug(
        f'Extracted {len(filtered_entities)} entities{log_suffix} '
        f'in {(end - start) * 1000:.0f} ms'
    )

    # Convert to EntityNode objects
    if use_freeform:
        extracted_nodes = _create_entity_nodes_freeform(
            cast(list[ExtractedEntityFreeform], filtered_entities),
            excluded_entity_types,
            episode,
        )
    else:
        extracted_nodes = _create_entity_nodes(
            cast(list[ExtractedEntity], filtered_entities),
            entity_types_context,
            excluded_entity_types,
            episode,
        )

    logger.debug(f'Extracted nodes: {[n.uuid for n in extracted_nodes]}')
    return extracted_nodes


def _build_entity_types_context(
    entity_types: dict[str, type[BaseModel]] | None,
) -> list[dict]:
    """Build entity types context with ID mappings."""
    entity_types_context = [
        {
            'entity_type_id': 0,
            'entity_type_name': 'Entity',
            'entity_type_description': (
                'Default entity classification. Use this entity type '
                'if the entity is not one of the other listed types.'
            ),
        }
    ]

    if entity_types is not None:
        entity_types_context += [
            {
                'entity_type_id': i + 1,
                'entity_type_name': type_name,
                'entity_type_description': type_model.__doc__,
            }
            for i, (type_name, type_model) in enumerate(entity_types.items())
        ]

    return entity_types_context


async def _extract_nodes_single(
    llm_client: LLMClient,
    episode: EpisodicNode,
    context: dict,
) -> list[ExtractedEntity]:
    """Extract entities using a single LLM call with predefined entity types."""
    llm_response = await _call_extraction_llm(llm_client, episode, context, freeform=False)
    response_object = ExtractedEntities(**llm_response)
    return response_object.extracted_entities


async def _extract_nodes_single_freeform(
    llm_client: LLMClient,
    episode: EpisodicNode,
    context: dict,
) -> list[ExtractedEntityFreeform]:
    """Extract entities using a single LLM call with freeform type classification."""
    llm_response = await _call_extraction_llm(llm_client, episode, context, freeform=True)
    response_object = ExtractedEntitiesFreeform(**llm_response)
    return response_object.extracted_entities


async def _call_extraction_llm(
    llm_client: LLMClient,
    episode: EpisodicNode,
    context: dict,
    freeform: bool = False,
) -> dict:
    """Call the appropriate extraction prompt based on episode type."""
    if episode.source == EpisodeType.message:
        prompt = prompt_library.extract_nodes.extract_message(context)
        prompt_name = 'extract_nodes.extract_message'
    elif episode.source == EpisodeType.text:
        prompt = prompt_library.extract_nodes.extract_text(context)
        prompt_name = 'extract_nodes.extract_text'
    elif episode.source == EpisodeType.json:
        prompt = prompt_library.extract_nodes.extract_json(context)
        prompt_name = 'extract_nodes.extract_json'
    else:
        # Fallback to text extraction
        prompt = prompt_library.extract_nodes.extract_text(context)
        prompt_name = 'extract_nodes.extract_text'

    response_model = ExtractedEntitiesFreeform if freeform else ExtractedEntities

    return await llm_client.generate_response(
        prompt,
        response_model=response_model,
        group_id=episode.group_id,
        prompt_name=prompt_name,
    )


def _create_entity_nodes(
    extracted_entities: list[ExtractedEntity],
    entity_types_context: list[dict],
    excluded_entity_types: list[str] | None,
    episode: EpisodicNode,
) -> list[EntityNode]:
    """Convert ExtractedEntity objects to EntityNode objects (predefined types mode)."""
    extracted_nodes = []

    for extracted_entity in extracted_entities:
        type_id = extracted_entity.entity_type_id
        if 0 <= type_id < len(entity_types_context):
            entity_type_name = entity_types_context[type_id].get('entity_type_name')
        else:
            entity_type_name = 'Entity'

        # Check if this entity type should be excluded
        if excluded_entity_types and entity_type_name in excluded_entity_types:
            logger.debug(f'Excluding entity of type "{entity_type_name}"')
            continue

        labels: list[str] = list({'Entity', str(entity_type_name)})

        new_node = EntityNode(
            name=extracted_entity.name,
            group_id=episode.group_id,
            labels=labels,
            summary='',
            created_at=utc_now(),
        )
        extracted_nodes.append(new_node)
        logger.debug(f'Created new node: {new_node.uuid}')

    return extracted_nodes


def _create_entity_nodes_freeform(
    extracted_entities: list[ExtractedEntityFreeform],
    excluded_entity_types: list[str] | None,
    episode: EpisodicNode,
) -> list[EntityNode]:
    """Convert ExtractedEntityFreeform objects to EntityNode objects (freeform types mode)."""
    extracted_nodes = []

    for extracted_entity in extracted_entities:
        entity_type_name = _sanitize_label(extracted_entity.entity_type)

        # Check if this entity type should be excluded
        if excluded_entity_types and entity_type_name in excluded_entity_types:
            logger.debug(f'Excluding entity of type "{entity_type_name}"')
            continue

        labels: list[str] = list({'Entity', entity_type_name})

        new_node = EntityNode(
            name=extracted_entity.name,
            group_id=episode.group_id,
            labels=labels,
            summary='',
            created_at=utc_now(),
        )
        extracted_nodes.append(new_node)
        logger.debug(f'Created new node: {new_node.uuid} (type: {entity_type_name})')

    return extracted_nodes


async def _collect_candidate_nodes(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
    existing_nodes_override: list[EntityNode] | None,
) -> list[EntityNode]:
    """Search per extracted name and return unique candidates with overrides honored in order."""
    search_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients=clients,
                query=node.name,
                group_ids=[node.group_id],
                search_filter=SearchFilters(),
                config=NODE_HYBRID_SEARCH_RRF_DEDUP,
            )
            for node in extracted_nodes
        ]
    )

    # Log per-entity search results for dedup debugging
    for node, result in zip(extracted_nodes, search_results):
        candidate_names = [c.name for c in result.nodes]
        logger.debug(
            'DEDUP_CANDIDATES for %r: %d candidates found: %s',
            node.name,
            len(result.nodes),
            candidate_names,
        )

    candidate_nodes: list[EntityNode] = [node for result in search_results for node in result.nodes]

    if existing_nodes_override is not None:
        candidate_nodes.extend(existing_nodes_override)

    seen_candidate_uuids: set[str] = set()
    ordered_candidates: list[EntityNode] = []
    for candidate in candidate_nodes:
        if candidate.uuid in seen_candidate_uuids:
            continue
        seen_candidate_uuids.add(candidate.uuid)
        ordered_candidates.append(candidate)

    logger.debug(
        'DEDUP_CANDIDATE_POOL: %d extracted entities, %d total unique candidates (from %d raw results)',
        len(extracted_nodes),
        len(ordered_candidates),
        len(candidate_nodes),
    )

    return ordered_candidates


async def _lookup_node_by_name(
    driver: GraphDriver,
    name: str,
    group_id: str,
) -> EntityNode | None:
    """Direct DB lookup for an entity node by exact name within a group.

    Used as a fallback when the LLM identifies a duplicate that wasn't
    in the candidate pool returned by the top-N hybrid search.
    """
    return_clause = get_entity_node_return_query(driver.provider)
    query = f"""
        MATCH (n:Entity {{name: $name, group_id: $group_id}})
        RETURN {return_clause}
        LIMIT 1
    """
    records, _, _ = await driver.execute_query(query, name=name, group_id=group_id)
    if records:
        return entity_node_from_record(records[0])

    # Try case-insensitive match as a second attempt
    if driver.provider == GraphProvider.FALKORDB:
        query_ci = f"""
            MATCH (n:Entity {{group_id: $group_id}})
            WHERE toLower(n.name) = toLower($name)
            RETURN {return_clause}
            LIMIT 1
        """
    else:
        query_ci = f"""
            MATCH (n:Entity {{group_id: $group_id}})
            WHERE toLower(n.name) = toLower($name)
            RETURN {return_clause}
            LIMIT 1
        """
    records, _, _ = await driver.execute_query(query_ci, name=name, group_id=group_id)
    if records:
        node = entity_node_from_record(records[0])
        logger.debug(
            'DEDUP_DB_FALLBACK: case-insensitive match for %r -> found %r (%s)',
            name,
            node.name,
            node.uuid[:8],
        )
        return node

    return None


async def _resolve_with_llm(
    llm_client: LLMClient,
    driver: GraphDriver,
    extracted_nodes: list[EntityNode],
    indexes: DedupCandidateIndexes,
    state: DedupResolutionState,
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    entity_types: dict[str, type[BaseModel]] | None,
) -> None:
    """Escalate unresolved nodes to the dedupe prompt so the LLM can select or reject duplicates.

    The guardrails below defensively ignore malformed or duplicate LLM responses so the
    ingestion workflow remains deterministic even when the model misbehaves.
    """
    if not state.unresolved_indices:
        return

    entity_types_dict: dict[str, type[BaseModel]] = entity_types if entity_types is not None else {}

    llm_extracted_nodes = [extracted_nodes[i] for i in state.unresolved_indices]

    extracted_nodes_context = [
        {
            'id': i,
            'name': node.name,
            'entity_type': node.labels,
            'entity_type_description': entity_types_dict.get(
                next((item for item in node.labels if item != 'Entity'), '')
            ).__doc__
            or 'Default Entity Type',
        }
        for i, node in enumerate(llm_extracted_nodes)
    ]

    sent_ids = [ctx['id'] for ctx in extracted_nodes_context]
    logger.debug(
        'Sending %d entities to LLM for deduplication with IDs 0-%d (actual IDs sent: %s)',
        len(llm_extracted_nodes),
        len(llm_extracted_nodes) - 1,
        sent_ids if len(sent_ids) < 20 else f'{sent_ids[:10]}...{sent_ids[-10:]}',
    )
    if llm_extracted_nodes:
        sample_size = min(3, len(extracted_nodes_context))
        logger.debug(
            'First %d entity IDs: %s',
            sample_size,
            [ctx['id'] for ctx in extracted_nodes_context[:sample_size]],
        )
        if len(extracted_nodes_context) > 3:
            logger.debug(
                'Last %d entity IDs: %s',
                sample_size,
                [ctx['id'] for ctx in extracted_nodes_context[-sample_size:]],
            )

    existing_nodes_context = [
        {
            **{
                'name': candidate.name,
                'entity_types': candidate.labels,
            },
            **candidate.attributes,
        }
        for candidate in indexes.existing_nodes
    ]

    # Build name -> node mapping for resolving duplicates by name
    existing_nodes_by_name: dict[str, EntityNode] = {
        node.name: node for node in indexes.existing_nodes
    }

    logger.debug(
        'DEDUP_LLM_CONTEXT: %d unresolved entities, %d existing candidates, '
        'existing_names=%s',
        len(llm_extracted_nodes),
        len(indexes.existing_nodes),
        sorted(existing_nodes_by_name.keys()),
    )

    context = {
        'extracted_nodes': extracted_nodes_context,
        'existing_nodes': existing_nodes_context,
        'episode_content': episode.content if episode is not None else '',
        'previous_episodes': (
            [ep.content for ep in previous_episodes] if previous_episodes is not None else []
        ),
    }

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_nodes.nodes(context),
        response_model=NodeResolutions,
        model_size=ModelSize.small,
        prompt_name='dedupe_nodes.nodes',
    )

    node_resolutions: list[NodeDuplicate] = NodeResolutions(**llm_response).entity_resolutions

    valid_relative_range = range(len(state.unresolved_indices))
    processed_relative_ids: set[int] = set()

    received_ids = {r.id for r in node_resolutions}
    expected_ids = set(valid_relative_range)
    missing_ids = expected_ids - received_ids
    extra_ids = received_ids - expected_ids

    logger.debug(
        'Received %d resolutions for %d entities',
        len(node_resolutions),
        len(state.unresolved_indices),
    )

    if missing_ids:
        logger.warning('LLM did not return resolutions for IDs: %s', sorted(missing_ids))

    if extra_ids:
        logger.warning(
            'LLM returned invalid IDs outside valid range 0-%d: %s (all returned IDs: %s)',
            len(state.unresolved_indices) - 1,
            sorted(extra_ids),
            sorted(received_ids),
        )

    for resolution in node_resolutions:
        relative_id: int = resolution.id
        duplicate_name: str = resolution.duplicate_name

        if relative_id not in valid_relative_range:
            logger.warning(
                'Skipping invalid LLM dedupe id %d (valid range: 0-%d, received %d resolutions)',
                relative_id,
                len(state.unresolved_indices) - 1,
                len(node_resolutions),
            )
            continue

        if relative_id in processed_relative_ids:
            logger.warning('Duplicate LLM dedupe id %s received; ignoring.', relative_id)
            continue
        processed_relative_ids.add(relative_id)

        original_index = state.unresolved_indices[relative_id]
        extracted_node = extracted_nodes[original_index]

        resolved_node: EntityNode
        if not duplicate_name:
            logger.debug(
                'DEDUP_LLM_RESULT: %r (id=%d) -> NEW (no duplicate found)',
                extracted_node.name,
                relative_id,
            )
            resolved_node = extracted_node
        elif duplicate_name in existing_nodes_by_name:
            resolved_node = existing_nodes_by_name[duplicate_name]
            logger.debug(
                'DEDUP_LLM_RESULT: %r (id=%d) -> MERGED with existing %r (%s)',
                extracted_node.name,
                relative_id,
                duplicate_name,
                resolved_node.uuid[:8],
            )
        else:
            # LLM returned a name not in the candidate pool — try direct DB lookup
            # before giving up. The candidate search (top-10 per entity) may have
            # missed the correct node, but the LLM recognized it from episode context.
            db_node = await _lookup_node_by_name(driver, duplicate_name, extracted_node.group_id)
            if db_node is not None:
                resolved_node = db_node
                logger.info(
                    'DEDUP_LLM_DB_FALLBACK: %r (id=%d) -> MERGED with %r (%s) '
                    'via direct DB lookup (not in candidate pool of %d)',
                    extracted_node.name,
                    relative_id,
                    duplicate_name,
                    db_node.uuid[:8],
                    len(existing_nodes_by_name),
                )
            else:
                # Check for case-insensitive near-matches for diagnostics
                near_matches = [
                    name for name in existing_nodes_by_name
                    if name.lower() == duplicate_name.lower()
                ]
                logger.warning(
                    'Invalid duplicate_name for extracted node %s (%r); treating as no duplicate. '
                    'duplicate_name was: %r. Case-insensitive near-matches in candidate pool: %s. '
                    'DB lookup also found nothing. Total candidates: %d',
                    extracted_node.uuid,
                    extracted_node.name,
                    duplicate_name[:50] + '...' if len(duplicate_name) > 50 else duplicate_name,
                    near_matches if near_matches else 'NONE',
                    len(existing_nodes_by_name),
                )
                resolved_node = extracted_node

        state.resolved_nodes[original_index] = resolved_node
        state.uuid_map[extracted_node.uuid] = resolved_node.uuid
        if resolved_node.uuid != extracted_node.uuid:
            state.duplicate_pairs.append((extracted_node, resolved_node))


async def resolve_extracted_nodes(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, type[BaseModel]] | None = None,
    existing_nodes_override: list[EntityNode] | None = None,
) -> tuple[list[EntityNode], dict[str, str], list[tuple[EntityNode, EntityNode]]]:
    """Search for existing nodes, resolve deterministic matches, then escalate holdouts to the LLM dedupe prompt."""
    llm_client = clients.llm_client
    existing_nodes = await _collect_candidate_nodes(
        clients,
        extracted_nodes,
        existing_nodes_override,
    )

    indexes: DedupCandidateIndexes = _build_candidate_indexes(existing_nodes)

    state = DedupResolutionState(
        resolved_nodes=[None] * len(extracted_nodes),
        uuid_map={},
        unresolved_indices=[],
    )

    _resolve_with_similarity(extracted_nodes, indexes, state)

    await _resolve_with_llm(
        llm_client,
        clients.driver,
        extracted_nodes,
        indexes,
        state,
        episode,
        previous_episodes,
        entity_types,
    )

    fallback_count = 0
    for idx, node in enumerate(extracted_nodes):
        if state.resolved_nodes[idx] is None:
            state.resolved_nodes[idx] = node
            state.uuid_map[node.uuid] = node.uuid
            fallback_count += 1

    # Summary: how many were deduped vs kept as new
    deduped_count = sum(
        1 for e, r in zip(extracted_nodes, state.resolved_nodes)
        if r is not None and r.uuid != e.uuid
    )
    new_count = len(extracted_nodes) - deduped_count
    logger.debug(
        'DEDUP_RESOLVE_SUMMARY: %d extracted -> %d deduped to existing, %d kept as new '
        '(%d resolved by fallback)',
        len(extracted_nodes),
        deduped_count,
        new_count,
        fallback_count,
    )

    logger.debug(
        'Resolved nodes: %s',
        [node.uuid for node in state.resolved_nodes if node is not None],
    )

    return (
        [node for node in state.resolved_nodes if node is not None],
        state.uuid_map,
        state.duplicate_pairs,
    )


def _build_edges_by_node(edges: list[EntityEdge] | None) -> dict[str, list[EntityEdge]]:
    """Build a dictionary mapping node UUIDs to their connected edges."""
    edges_by_node: dict[str, list[EntityEdge]] = {}
    if not edges:
        return edges_by_node
    for edge in edges:
        if edge.source_node_uuid not in edges_by_node:
            edges_by_node[edge.source_node_uuid] = []
        if edge.target_node_uuid not in edges_by_node:
            edges_by_node[edge.target_node_uuid] = []
        edges_by_node[edge.source_node_uuid].append(edge)
        edges_by_node[edge.target_node_uuid].append(edge)
    return edges_by_node


async def extract_attributes_from_nodes(
    clients: GraphitiClients,
    nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, type[BaseModel]] | None = None,
    should_summarize_node: NodeSummaryFilter | None = None,
    edges: list[EntityEdge] | None = None,
) -> list[EntityNode]:
    llm_client = clients.llm_client
    embedder = clients.embedder

    # Pre-build edges lookup for O(E + N) instead of O(N * E)
    edges_by_node = _build_edges_by_node(edges)

    # Extract attributes in parallel (per-entity calls)
    attribute_results: list[dict[str, Any]] = await semaphore_gather(
        *[
            _extract_entity_attributes(
                llm_client,
                node,
                episode,
                previous_episodes,
                (
                    entity_types.get(next((item for item in node.labels if item != 'Entity'), ''))
                    if entity_types is not None
                    else None
                ),
            )
            for node in nodes
        ]
    )

    # Apply attributes to nodes
    for node, attributes in zip(nodes, attribute_results, strict=True):
        node.attributes.update(attributes)

    # Extract summaries in batch
    await _extract_entity_summaries_batch(
        llm_client,
        nodes,
        episode,
        previous_episodes,
        should_summarize_node,
        edges_by_node,
    )

    await create_entity_node_embeddings(embedder, nodes)

    return nodes


async def _extract_entity_attributes(
    llm_client: LLMClient,
    node: EntityNode,
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    entity_type: type[BaseModel] | None,
) -> dict[str, Any]:
    if entity_type is None or len(entity_type.model_fields) == 0:
        return {}

    attributes_context = _build_episode_context(
        # should not include summary
        node_data={
            'name': node.name,
            'entity_types': node.labels,
            'attributes': node.attributes,
        },
        episode=episode,
        previous_episodes=previous_episodes,
    )

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_attributes(attributes_context),
        response_model=entity_type,
        model_size=ModelSize.small,
        group_id=node.group_id,
        prompt_name='extract_nodes.extract_attributes',
    )

    # validate response
    entity_type(**llm_response)

    return llm_response


async def _extract_entity_summaries_batch(
    llm_client: LLMClient,
    nodes: list[EntityNode],
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    should_summarize_node: NodeSummaryFilter | None,
    edges_by_node: dict[str, list[EntityEdge]],
) -> None:
    """Extract summaries for multiple entities in batched LLM calls.

    Nodes that don't need LLM summarization (short enough with edge facts appended)
    are handled directly without an LLM call. Nodes needing summarization are
    partitioned into flights of MAX_NODES and processed with separate LLM calls.
    """
    # Determine which nodes need LLM summarization vs direct edge fact appending
    nodes_needing_llm: list[EntityNode] = []

    for node in nodes:
        # Check if node should be summarized at all
        if should_summarize_node is not None and not await should_summarize_node(node):
            continue

        node_edges = edges_by_node.get(node.uuid, [])

        # Build summary with edge facts appended
        summary_with_edges = node.summary
        if node_edges:
            edge_facts = '\n'.join(edge.fact for edge in node_edges if edge.fact)
            summary_with_edges = f'{summary_with_edges}\n{edge_facts}'.strip()

        # If summary is short enough, use it directly (append edge facts, no LLM call)
        if summary_with_edges and len(summary_with_edges) <= MAX_SUMMARY_CHARS * 4:
            node.summary = summary_with_edges
            continue

        # Skip if no summary content and no episode to generate from
        if not summary_with_edges and episode is None:
            continue

        # This node needs LLM summarization
        nodes_needing_llm.append(node)

    # If no nodes need LLM summarization, return early
    if not nodes_needing_llm:
        return

    # Partition nodes into flights of MAX_NODES
    node_flights = [
        nodes_needing_llm[i : i + MAX_NODES] for i in range(0, len(nodes_needing_llm), MAX_NODES)
    ]

    # Process flights in parallel
    await semaphore_gather(
        *[
            _process_summary_flight(llm_client, flight, episode, previous_episodes)
            for flight in node_flights
        ]
    )


async def _process_summary_flight(
    llm_client: LLMClient,
    nodes: list[EntityNode],
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
) -> None:
    """Process a single flight of nodes for batch summarization."""
    # Build context for batch summarization
    entities_context = [
        {
            'name': node.name,
            'summary': node.summary,
            'entity_types': node.labels,
            'attributes': node.attributes,
        }
        for node in nodes
    ]

    batch_context = {
        'entities': entities_context,
        'episode_content': episode.content if episode is not None else '',
        'previous_episodes': (
            [ep.content for ep in previous_episodes] if previous_episodes is not None else []
        ),
    }

    # Get group_id from the first node (all nodes in a batch should have same group_id)
    group_id = nodes[0].group_id if nodes else None

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_summaries_batch(batch_context),
        response_model=SummarizedEntities,
        model_size=ModelSize.small,
        group_id=group_id,
        prompt_name='extract_nodes.extract_summaries_batch',
    )

    # Build case-insensitive name -> nodes mapping (handles duplicates)
    name_to_nodes: dict[str, list[EntityNode]] = {}
    for node in nodes:
        key = node.name.lower()
        if key not in name_to_nodes:
            name_to_nodes[key] = []
        name_to_nodes[key].append(node)

    # Apply summaries from LLM response
    summaries_response = SummarizedEntities(**llm_response)
    for summarized_entity in summaries_response.summaries:
        matching_nodes = name_to_nodes.get(summarized_entity.name.lower(), [])
        if matching_nodes:
            truncated_summary = truncate_at_sentence(summarized_entity.summary, MAX_SUMMARY_CHARS)
            for node in matching_nodes:
                node.summary = truncated_summary
        else:
            logger.warning(
                'LLM returned summary for unknown entity (first 30 chars): %.30s',
                summarized_entity.name,
            )


def _build_episode_context(
    node_data: dict[str, Any],
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
) -> dict[str, Any]:
    return {
        'node': node_data,
        'episode_content': episode.content if episode is not None else '',
        'previous_episodes': (
            [ep.content for ep in previous_episodes] if previous_episodes is not None else []
        ),
    }


def _sanitize_label(label: str) -> str:
    """Sanitize a label string to be safe for use as a Cypher node label.

    Strips non-alphanumeric/underscore characters, normalizes to PascalCase
    (splitting on underscores), and ensures the result starts with a letter.
    Falls back to 'Entity' if no valid characters remain.
    """
    # Remove any characters that aren't alphanumeric or underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', label)

    # Split on underscores to find word boundaries, filtering empty parts
    parts = [p for p in sanitized.split('_') if p]

    if not parts:
        return 'Entity'

    # Normalize to PascalCase
    normalized = ''.join(
        p[0].upper() + p[1:].lower() if len(p) > 1 else p.upper() for p in parts
    )

    # Ensure it starts with a letter (prepend 'Label' if it starts with a digit)
    if normalized and normalized[0].isdigit():
        normalized = 'Label' + normalized

    return normalized or 'Entity'


async def reclassify_entity(
    llm_client: LLMClient,
    entity: EntityNode,
) -> str:
    """Classify an entity's type using LLM based on its name and summary."""
    context = {
        'entity_name': entity.name,
        'entity_summary': entity.summary,
    }
    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.reclassify_entity(context),
        response_model=ReclassifiedEntity,
        model_size=ModelSize.small,
        group_id=entity.group_id,
        prompt_name='extract_nodes.reclassify_entity',
    )
    result = ReclassifiedEntity(**llm_response)
    return _sanitize_label(result.entity_type)


async def reprocess_entity_types(
    clients: GraphitiClients,
    group_id: str,
) -> list[EntityNode]:
    """Retroactively classify entities that only have the generic 'Entity' label.

    Retrieves all entities for the given group, filters for untyped entities
    (those with only ['Entity'] as labels), uses the LLM to classify each one,
    and updates their labels in the graph.
    """
    driver = clients.driver
    llm_client = clients.llm_client

    # Retrieve all entities for the group, then filter in Python
    all_entities = await EntityNode.get_by_group_ids(driver, [group_id])
    entities = [e for e in all_entities if set(e.labels) == {'Entity'}]

    if not entities:
        logger.info(f'No untyped entities found for group_id={group_id}')
        return []

    logger.info(f'Found {len(entities)} untyped entities to reclassify for group_id={group_id}')

    # Reclassify all entities with concurrency control
    new_types = await semaphore_gather(
        *[reclassify_entity(llm_client, entity) for entity in entities]
    )

    # Update labels and save
    updated = []
    for i, (entity, new_type) in enumerate(zip(entities, new_types, strict=True)):
        if new_type != 'Entity':
            entity.labels = list({'Entity', new_type})
            await entity.save(driver)
            updated.append(entity)
            logger.info(f'Reclassified {i + 1}/{len(entities)}: {entity.name} → {new_type}')
        else:
            logger.info(f'Skipped {i + 1}/{len(entities)}: {entity.name} (no better type found)')

    logger.info(f'Reclassified {len(updated)}/{len(entities)} entities')
    return updated
