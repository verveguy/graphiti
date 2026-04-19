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
from time import time

from pydantic import BaseModel
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    create_entity_edge_embeddings,
)
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_edges import EdgeBatchResolutions, EdgeDuplicate
from graphiti_core.prompts.extract_edges import Edge as ExtractedEdge
from graphiti_core.prompts.extract_edges import ExtractedEdges
from graphiti_core.search.search import search
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact

logger = logging.getLogger(__name__)

INVALIDATION_SIM_THRESHOLD = 0.3  # cosine similarity floor for invalidation candidates (stage 2)
DEDUP_SIM_THRESHOLD = 0.5  # cosine similarity floor for dedup candidates (stage 1)

# Maximum number of edges to include in a single batched Stage 1 LLM call.
# At typical fact lengths (~50 tokens each) and 10 candidates per edge, 20 edges
# keeps the prompt well within context limits (~10k tokens).
EDGE_DEDUP_BATCH_SIZE = 20


def _cosine_sim(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _extract_edge_attributes(
    llm_client: LLMClient,
    edge: EntityEdge,
    episode: EpisodicNode | None,
    edge_type_candidates: dict[str, type[BaseModel]] | None,
) -> None:
    """Extract and set custom attributes on an edge using the appropriate edge type model."""
    edge_model = edge_type_candidates.get(edge.name) if edge_type_candidates else None
    if edge_model is not None and len(edge_model.model_fields) != 0:
        edge_attributes_context = {
            'fact': edge.fact,
            'reference_time': episode.valid_at if episode is not None else None,
            'existing_attributes': edge.attributes,
        }
        edge_attributes_response = await llm_client.generate_response(
            prompt_library.extract_edges.extract_attributes(edge_attributes_context),
            response_model=edge_model,  # type: ignore
            model_size=ModelSize.small,
            prompt_name='extract_edges.extract_attributes',
        )
        edge.attributes = edge_attributes_response
    else:
        edge.attributes = {}


def build_episodic_edges(
    entity_nodes: list[EntityNode],
    episode_uuid: str,
    created_at: datetime,
) -> list[EpisodicEdge]:
    episodic_edges: list[EpisodicEdge] = [
        EpisodicEdge(
            source_node_uuid=episode_uuid,
            target_node_uuid=node.uuid,
            created_at=created_at,
            group_id=node.group_id,
        )
        for node in entity_nodes
    ]

    logger.debug(f'Built {len(episodic_edges)} episodic edges')

    return episodic_edges


def build_community_edges(
    entity_nodes: list[EntityNode],
    community_node: CommunityNode,
    created_at: datetime,
) -> list[CommunityEdge]:
    edges: list[CommunityEdge] = [
        CommunityEdge(
            source_node_uuid=community_node.uuid,
            target_node_uuid=node.uuid,
            created_at=created_at,
            group_id=community_node.group_id,
        )
        for node in entity_nodes
    ]

    return edges


async def extract_edges(
    clients: GraphitiClients,
    episode: EpisodicNode,
    nodes: list[EntityNode],
    previous_episodes: list[EpisodicNode],
    edge_type_map: dict[tuple[str, str], list[str]],
    group_id: str = '',
    edge_types: dict[str, type[BaseModel]] | None = None,
    custom_extraction_instructions: str | None = None,
) -> list[EntityEdge]:
    start = time()

    extract_edges_max_tokens = 16384
    llm_client = clients.llm_client

    # Build mapping from edge type name to list of valid signatures
    edge_type_signatures_map: dict[str, list[tuple[str, str]]] = {}
    for signature, edge_type_names in edge_type_map.items():
        for edge_type in edge_type_names:
            if edge_type not in edge_type_signatures_map:
                edge_type_signatures_map[edge_type] = []
            edge_type_signatures_map[edge_type].append(signature)

    edge_types_context = (
        [
            {
                'fact_type_name': type_name,
                'fact_type_signatures': edge_type_signatures_map.get(
                    type_name, [('Entity', 'Entity')]
                ),
                'fact_type_description': type_model.__doc__,
            }
            for type_name, type_model in edge_types.items()
        ]
        if edge_types is not None
        else []
    )

    # Build name-to-node mapping for validation
    name_to_node: dict[str, EntityNode] = {node.name: node for node in nodes}

    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'nodes': [{'name': node.name, 'entity_types': node.labels} for node in nodes],
        'previous_episodes': [ep.content for ep in previous_episodes],
        'reference_time': episode.valid_at,
        'edge_types': edge_types_context,
        'custom_extraction_instructions': custom_extraction_instructions or '',
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_edges.edge(context),
        response_model=ExtractedEdges,
        max_tokens=extract_edges_max_tokens,
        group_id=group_id,
        prompt_name='extract_edges.edge',
    )
    all_edges_data = ExtractedEdges(**llm_response).edges

    # Validate entity names
    edges_data: list[ExtractedEdge] = []
    for edge_data in all_edges_data:
        source_name = edge_data.source_entity_name
        target_name = edge_data.target_entity_name

        # Validate LLM-returned names exist in the nodes list
        if source_name not in name_to_node:
            logger.warning(
                'Source entity not found in nodes for edge relation: %s',
                edge_data.relation_type,
            )
            continue

        if target_name not in name_to_node:
            logger.warning(
                'Target entity not found in nodes for edge relation: %s',
                edge_data.relation_type,
            )
            continue

        # Drop self-edges where source and target resolve to the same node
        source_node = name_to_node[source_name]
        target_node = name_to_node[target_name]
        if source_node.uuid == target_node.uuid:
            logger.info(
                'Dropping self-edge for node %s (source and target resolve to same node)',
                source_node.uuid,
            )
            continue

        edges_data.append(edge_data)

    end = time()
    logger.debug(f'Extracted {len(edges_data)} new edges in {(end - start) * 1000:.0f} ms')

    if len(edges_data) == 0:
        return []

    # Convert the extracted data into EntityEdge objects
    edges = []
    for edge_data in edges_data:
        # Validate Edge Date information
        valid_at = edge_data.valid_at
        invalid_at = edge_data.invalid_at
        valid_at_datetime = None
        invalid_at_datetime = None

        # Filter out empty edges
        if not edge_data.fact.strip():
            continue

        # Names already validated above
        source_node = name_to_node.get(edge_data.source_entity_name)
        target_node = name_to_node.get(edge_data.target_entity_name)

        if source_node is None or target_node is None:
            logger.warning('Could not find source or target node for extracted edge')
            continue

        source_node_uuid = source_node.uuid
        target_node_uuid = target_node.uuid

        if valid_at:
            try:
                valid_at_datetime = ensure_utc(
                    datetime.fromisoformat(valid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                logger.warning(f'WARNING: Error parsing valid_at date: {e}. Input: {valid_at}')

        if invalid_at:
            try:
                invalid_at_datetime = ensure_utc(
                    datetime.fromisoformat(invalid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                logger.warning(f'WARNING: Error parsing invalid_at date: {e}. Input: {invalid_at}')
        edge = EntityEdge(
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
            name=edge_data.relation_type,
            group_id=group_id,
            fact=edge_data.fact,
            episodes=[episode.uuid],
            created_at=utc_now(),
            valid_at=valid_at_datetime,
            invalid_at=invalid_at_datetime,
            reference_time=episode.valid_at,
        )
        edges.append(edge)
        logger.debug(
            f'Created new edge {edge.uuid} from {edge.source_node_uuid} to {edge.target_node_uuid}'
        )

    logger.debug(f'Extracted edges: {[e.uuid for e in edges]}')

    return edges


async def resolve_extracted_edges(
    clients: GraphitiClients,
    extracted_edges: list[EntityEdge],
    episode: EpisodicNode,
    entities: list[EntityNode],
    edge_types: dict[str, type[BaseModel]],
    edge_type_map: dict[tuple[str, str], list[str]],
    existing_edges_override: list[EntityEdge] | None = None,
) -> tuple[list[EntityEdge], list[EntityEdge], list[EntityEdge]]:
    """Resolve extracted edges against existing graph context.

    Returns
    -------
    tuple[list[EntityEdge], list[EntityEdge], list[EntityEdge]]
        A tuple of (resolved_edges, invalidated_edges, new_edges) where:
        - resolved_edges: All edges after resolution (may include existing edges if duplicates found)
        - invalidated_edges: Edges that were invalidated/contradicted by new information
        - new_edges: Only edges that are new to the graph (not duplicates of existing edges)
    """
    # Fast path: deduplicate exact matches within the extracted edges before parallel processing
    seen: dict[tuple[str, str, str], EntityEdge] = {}
    deduplicated_edges: list[EntityEdge] = []

    for edge in extracted_edges:
        key = (
            edge.source_node_uuid,
            edge.target_node_uuid,
            _normalize_string_exact(edge.fact),
        )
        if key not in seen:
            seen[key] = edge
            deduplicated_edges.append(edge)

    extracted_edges = deduplicated_edges

    driver = clients.driver
    llm_client = clients.llm_client
    embedder = clients.embedder
    await create_entity_edge_embeddings(embedder, extracted_edges)

    t_between = time()
    valid_edges_list: list[list[EntityEdge]] = await semaphore_gather(
        *[
            EntityEdge.get_between_nodes(driver, edge.source_node_uuid, edge.target_node_uuid)
            for edge in extracted_edges
        ]
    )
    logger.debug(
        'EDGE_RESOLVE_TIMING: get_between_nodes for %d edges in %.0f ms',
        len(extracted_edges),
        (time() - t_between) * 1000,
    )

    # Merge override edges (e.g. from the recent Redis dedup cache) into
    # the per-extracted-edge candidate lists so that recently resolved edges
    # that are not yet visible in the graph-service indexes are still
    # considered during deduplication.
    if existing_edges_override:
        override_by_pair: dict[tuple[str, str], list[EntityEdge]] = {}
        for oe in existing_edges_override:
            key = (oe.source_node_uuid, oe.target_node_uuid)
            override_by_pair.setdefault(key, []).append(oe)

        for i, extracted_edge in enumerate(extracted_edges):
            pair_key = (extracted_edge.source_node_uuid, extracted_edge.target_node_uuid)
            overrides = override_by_pair.get(pair_key, [])
            if overrides:
                existing_uuids = {e.uuid for e in valid_edges_list[i]}
                for oe in overrides:
                    if oe.uuid not in existing_uuids:
                        valid_edges_list[i].append(oe)
                        existing_uuids.add(oe.uuid)

    t_related = time()
    related_edges_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients,
                extracted_edge.fact,
                group_ids=[extracted_edge.group_id],
                config=EDGE_HYBRID_SEARCH_RRF,
                search_filter=SearchFilters(edge_uuids=[edge.uuid for edge in valid_edges]),
            )
            for extracted_edge, valid_edges in zip(extracted_edges, valid_edges_list, strict=True)
        ]
    )
    logger.debug(
        'EDGE_RESOLVE_TIMING: related_edges search for %d edges in %.0f ms',
        len(extracted_edges),
        (time() - t_related) * 1000,
    )

    related_edges_lists: list[list[EntityEdge]] = [result.edges for result in related_edges_results]

    # Fetch invalidation candidates scoped to each edge's endpoints.
    # Instead of searching the entire graph (expensive, returns irrelevant results),
    # we fetch edges connected to either the source or target entity. This catches
    # real contradictions (e.g. "Brett → VP" vs "Sanjeev → VP" share the target)
    # without the cost of a full HNSW+FTS search per edge.
    t_invalidation = time()

    # Collect unique node UUIDs to fetch edges for
    endpoint_uuids: set[str] = set()
    for edge in extracted_edges:
        endpoint_uuids.add(edge.source_node_uuid)
        endpoint_uuids.add(edge.target_node_uuid)

    # Fetch all edges for all endpoints in parallel
    endpoint_edges_lists: list[list[EntityEdge]] = await semaphore_gather(
        *[EntityEdge.get_by_node_uuid(driver, uuid) for uuid in endpoint_uuids]
    )

    # Build lookup: node_uuid -> set of edges connected to that node
    endpoint_edges_map: dict[str, list[EntityEdge]] = {}
    for uuid, edges_for_node in zip(endpoint_uuids, endpoint_edges_lists, strict=True):
        endpoint_edges_map[uuid] = edges_for_node

    logger.debug(
        'EDGE_RESOLVE_TIMING: invalidation_candidate fetch for %d endpoints in %.0f ms',
        len(endpoint_uuids),
        (time() - t_invalidation) * 1000,
    )

    # Build per-edge invalidation candidates from endpoint edges,
    # filtering by embedding similarity to avoid sending irrelevant edges to the LLM.
    # "Brett attended meeting X" should not be checked against "Brett manages Sanjeev".
    MAX_INVALIDATION_CANDIDATES = 10  # cap per edge to bound LLM prompt size

    edge_invalidation_candidates: list[list[EntityEdge]] = []
    for extracted_edge, related_edges in zip(extracted_edges, related_edges_lists, strict=True):
        # Union edges from both endpoints
        source_edges = endpoint_edges_map.get(extracted_edge.source_node_uuid, [])
        target_edges = endpoint_edges_map.get(extracted_edge.target_node_uuid, [])
        all_endpoint_edges = {e.uuid: e for e in source_edges + target_edges}

        # Remove the extracted edge itself and any already in related_edges
        related_uuids = {edge.uuid for edge in related_edges}
        candidates = [
            e
            for e in all_endpoint_edges.values()
            if e.uuid not in related_uuids and e.uuid != extracted_edge.uuid
        ]

        # Filter by embedding similarity and take top N
        scored = [
            (e, _cosine_sim(extracted_edge.fact_embedding, e.fact_embedding)) for e in candidates
        ]
        filtered = sorted(
            [(e, s) for e, s in scored if s >= INVALIDATION_SIM_THRESHOLD],
            key=lambda x: x[1],
            reverse=True,
        )[:MAX_INVALIDATION_CANDIDATES]

        result = [e for e, _ in filtered]
        edge_invalidation_candidates.append(result)

        if len(candidates) != len(result):
            logger.debug(
                'EDGE_INVALIDATION_FILTER: edge "%s" — %d endpoint edges -> %d after sim filter (threshold=%.1f)',
                extracted_edge.fact[:60],
                len(candidates),
                len(result),
                INVALIDATION_SIM_THRESHOLD,
            )

    logger.debug(
        'EDGE_RESOLVE_TIMING: invalidation candidates built: %s per edge',
        [len(c) for c in edge_invalidation_candidates],
    )

    logger.debug(
        f'Related edges: {[e.uuid for edges_lst in related_edges_lists for e in edges_lst]}'
    )

    # Build entity hash table
    uuid_entity_map: dict[str, EntityNode] = {entity.uuid: entity for entity in entities}

    # Collect all node UUIDs referenced by edges that are not in the entities list
    referenced_node_uuids = set()
    for extracted_edge in extracted_edges:
        if extracted_edge.source_node_uuid not in uuid_entity_map:
            referenced_node_uuids.add(extracted_edge.source_node_uuid)
        if extracted_edge.target_node_uuid not in uuid_entity_map:
            referenced_node_uuids.add(extracted_edge.target_node_uuid)

    # Fetch missing nodes from the database
    if referenced_node_uuids:
        missing_nodes = await EntityNode.get_by_uuids(driver, list(referenced_node_uuids))
        for node in missing_nodes:
            uuid_entity_map[node.uuid] = node

    # Determine which edge types are relevant for each edge based on node signatures.
    # `edge_types_lst` stores the subset of custom edge definitions whose
    # node signature matches each extracted edge.
    edge_types_lst: list[dict[str, type[BaseModel]]] = []
    for extracted_edge in extracted_edges:
        source_node = uuid_entity_map.get(extracted_edge.source_node_uuid)
        target_node = uuid_entity_map.get(extracted_edge.target_node_uuid)
        source_node_labels = (
            source_node.labels + ['Entity'] if source_node is not None else ['Entity']
        )
        target_node_labels = (
            target_node.labels + ['Entity'] if target_node is not None else ['Entity']
        )
        label_tuples = [
            (source_label, target_label)
            for source_label in source_node_labels
            for target_label in target_node_labels
        ]

        extracted_edge_types = {}
        for label_tuple in label_tuples:
            type_names = edge_type_map.get(label_tuple, [])
            for type_name in type_names:
                type_model = edge_types.get(type_name)
                if type_model is None:
                    continue

                extracted_edge_types[type_name] = type_model

        edge_types_lst.append(extracted_edge_types)

    # =========================================================================
    # FAST-PATH CLASSIFICATION + BATCHED STAGE 1 DEDUP
    # =========================================================================
    t_llm_resolve = time()

    final_results: dict[int, tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]] = {}
    no_candidates_indices: list[int] = []   # both related and invalidation candidates empty
    direct_stage2_indices: list[int] = []   # no dedup candidates but has invalidation candidates
    batch_indices: list[int] = []           # need batched Stage 1
    fast_path_exact_count = 0

    for i, (extracted_edge, related_edges, inv_candidates) in enumerate(zip(
        extracted_edges, related_edges_lists, edge_invalidation_candidates, strict=True
    )):
        # Fast path 1: no candidates at all — skip all LLM calls
        if len(related_edges) == 0 and len(inv_candidates) == 0:
            no_candidates_indices.append(i)
            continue

        # No dedup candidates but has invalidation candidates — skip Stage 1, go to Stage 2
        if len(related_edges) == 0:
            direct_stage2_indices.append(i)
            continue

        # Fast path 2: exact text match among related_edges — skip all LLM calls
        normalized_fact = _normalize_string_exact(extracted_edge.fact)
        matched: EntityEdge | None = None
        for edge in related_edges:
            if (
                edge.source_node_uuid == extracted_edge.source_node_uuid
                and edge.target_node_uuid == extracted_edge.target_node_uuid
                and _normalize_string_exact(edge.fact) == normalized_fact
            ):
                matched = edge
                break

        if matched is not None:
            if episode is not None and episode.uuid not in matched.episodes:
                matched.episodes.append(episode.uuid)
            final_results[i] = (matched, [], [])
            fast_path_exact_count += 1
            logger.debug(
                'EDGE_RESOLVE_STAGE: edge %s — exact text match fast path', extracted_edge.uuid,
            )
            continue

        batch_indices.append(i)

    # Extract attributes for no-candidates edges (they are new, genuine edges)
    if no_candidates_indices:
        await semaphore_gather(*[
            _extract_edge_attributes(llm_client, extracted_edges[i], episode, edge_types_lst[i])
            for i in no_candidates_indices
        ])
    for i in no_candidates_indices:
        final_results[i] = (extracted_edges[i], [], [])
        logger.debug(
            'EDGE_RESOLVE_STAGE: edge %s — no candidates fast path', extracted_edges[i].uuid,
        )

    logger.debug(
        'EDGE_RESOLVE_STAGE: %d edges total — %d no-candidates, %d exact-match, '
        '%d direct-stage2, %d queued for Stage 1 batch',
        len(extracted_edges), len(no_candidates_indices), fast_path_exact_count,
        len(direct_stage2_indices), len(batch_indices),
    )

    # =========================================================================
    # BATCHED STAGE 1: Single LLM call per sub-batch to identify duplicates
    # =========================================================================
    stage1_duplicate_of: dict[int, int | None] = {i: None for i in batch_indices}

    if batch_indices:
        t_stage1 = time()
        for batch_start in range(0, len(batch_indices), EDGE_DEDUP_BATCH_SIZE):
            sub_batch_indices = batch_indices[batch_start: batch_start + EDGE_DEDUP_BATCH_SIZE]
            batch_payload = [
                {
                    'edge_idx': batch_pos,
                    'fact': extracted_edges[i].fact,
                    'candidates': [
                        {'candidate_idx': j, 'fact': e.fact}
                        for j, e in enumerate(related_edges_lists[i])
                    ],
                }
                for batch_pos, i in enumerate(sub_batch_indices)
            ]
            t_sub = time()
            llm_response = await llm_client.generate_response(
                prompt_library.dedupe_edges.resolve_edges_batch({'edges': batch_payload}),
                response_model=EdgeBatchResolutions,
                model_size=ModelSize.small,
                prompt_name='dedupe_edges.resolve_edges_batch',
            )
            batch_response = EdgeBatchResolutions(**llm_response)
            logger.debug(
                'EDGE_RESOLVE_TIMING: Stage 1 sub-batch %d–%d (%d edges) in %.0f ms',
                batch_start, batch_start + len(sub_batch_indices) - 1,
                len(sub_batch_indices), (time() - t_sub) * 1000,
            )

            response_by_batch_pos: dict[int, int | None] = {}
            for resolution in batch_response.edge_resolutions:
                if 0 <= resolution.edge_idx < len(sub_batch_indices):
                    dup_of = resolution.duplicate_of
                    candidates_len = len(related_edges_lists[sub_batch_indices[resolution.edge_idx]])
                    if dup_of is not None and not (0 <= dup_of < candidates_len):
                        logger.warning(
                            'EDGE_RESOLVE_STAGE1_BATCH: edge_idx=%d duplicate_of=%d out of range '
                            '(candidates=%d), treating as non-duplicate',
                            resolution.edge_idx, dup_of, candidates_len,
                        )
                        dup_of = None
                    response_by_batch_pos[resolution.edge_idx] = dup_of
                else:
                    logger.warning(
                        'EDGE_RESOLVE_STAGE1_BATCH: out-of-range edge_idx=%d in response '
                        '(batch size=%d)',
                        resolution.edge_idx, len(sub_batch_indices),
                    )

            for batch_pos, i in enumerate(sub_batch_indices):
                if batch_pos not in response_by_batch_pos:
                    logger.warning(
                        'EDGE_RESOLVE_STAGE1_BATCH: missing edge_idx=%d in response '
                        '(edge "%s"), treating as non-duplicate',
                        batch_pos, extracted_edges[i].fact[:60],
                    )
                stage1_duplicate_of[i] = response_by_batch_pos.get(batch_pos)

        logger.debug(
            'EDGE_RESOLVE_TIMING: Stage 1 batched LLM (%d edges, %d sub-batches) in %.0f ms',
            len(batch_indices),
            (len(batch_indices) + EDGE_DEDUP_BATCH_SIZE - 1) // EDGE_DEDUP_BATCH_SIZE,
            (time() - t_stage1) * 1000,
        )

    # Extract attributes for batch-identified duplicates and record their results
    stage1_duplicate_indices = [i for i in batch_indices if stage1_duplicate_of[i] is not None]
    stage2_indices = direct_stage2_indices + [
        i for i in batch_indices if stage1_duplicate_of[i] is None
    ]

    if stage1_duplicate_indices:
        await semaphore_gather(*[
            _extract_edge_attributes(
                llm_client,
                related_edges_lists[i][stage1_duplicate_of[i]],  # type: ignore[index]
                episode,
                edge_types_lst[i],
            )
            for i in stage1_duplicate_indices
        ])
    for i in stage1_duplicate_indices:
        dup_idx = stage1_duplicate_of[i]
        resolved = related_edges_lists[i][dup_idx]  # type: ignore[index]
        if episode is not None:
            resolved.episodes.append(episode.uuid)
        final_results[i] = (resolved, [], [resolved])
        logger.debug(
            'EDGE_RESOLVE_STAGE: edge %s — Stage 1 batch duplicate -> %s',
            extracted_edges[i].uuid, resolved.uuid,
        )

    # =========================================================================
    # STAGE 2: Run invalidation check in parallel for all non-duplicate edges
    # =========================================================================
    stage2_results_list: list[tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]] = list(
        await semaphore_gather(
            *[
                resolve_extracted_edge(
                    llm_client,
                    extracted_edges[i],
                    related_edges_lists[i],
                    edge_invalidation_candidates[i],
                    episode,
                    edge_types_lst[i],
                    _bypass_stage1=True,
                )
                for i in stage2_indices
            ]
        )
    )
    for i, result in zip(stage2_indices, stage2_results_list, strict=True):
        final_results[i] = result

    # Collect results in original order
    results: list[tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]] = [
        final_results[i] for i in range(len(extracted_edges))
    ]

    stage1_dedup_count = len(stage1_duplicate_indices)
    fast_path_total = len(no_candidates_indices) + fast_path_exact_count
    matched_existing_count = sum(
        1 for edge, result in zip(extracted_edges, results, strict=True)
        if result[0].uuid != edge.uuid
    )
    logger.debug(
        'EDGE_RESOLVE_TIMING: total LLM resolve for %d edges in %.0f ms '
        '(%d fast-path, %d Stage 1 batch duplicates, %d Stage 2)',
        len(extracted_edges), (time() - t_llm_resolve) * 1000,
        fast_path_total, stage1_dedup_count, len(stage2_indices),
    )
    logger.debug(
        'EDGE_RESOLVE_OUTCOMES: %d edges resolved: %d matched existing (%d batch-dedup, %d exact-match), '
        '%d new, %d with invalidations',
        len(extracted_edges), matched_existing_count,
        stage1_dedup_count, fast_path_exact_count,
        len(extracted_edges) - matched_existing_count,
        sum(1 for r in results if len(r[1]) > 0),
    )

    resolved_edges: list[EntityEdge] = []
    invalidated_edges: list[EntityEdge] = []
    new_edges: list[EntityEdge] = []
    for extracted_edge, result in zip(extracted_edges, results, strict=True):
        resolved_edge = result[0]
        invalidated_edge_chunk = result[1]
        # result[2] is duplicate_edges list

        resolved_edges.append(resolved_edge)
        invalidated_edges.extend(invalidated_edge_chunk)

        # Track edges that are new (not duplicates of existing edges)
        # An edge is new if the resolved edge UUID matches the extracted edge UUID
        if resolved_edge.uuid == extracted_edge.uuid:
            new_edges.append(resolved_edge)

    logger.debug(f'Resolved edges: {[e.uuid for e in resolved_edges]}')
    logger.debug(f'New edges (non-duplicates): {[e.uuid for e in new_edges]}')

    await semaphore_gather(
        create_entity_edge_embeddings(embedder, resolved_edges),
        create_entity_edge_embeddings(embedder, invalidated_edges),
    )

    return resolved_edges, invalidated_edges, new_edges


def resolve_edge_contradictions(
    resolved_edge: EntityEdge, invalidation_candidates: list[EntityEdge]
) -> list[EntityEdge]:
    if len(invalidation_candidates) == 0:
        return []

    # Determine which contradictory edges need to be expired
    invalidated_edges: list[EntityEdge] = []
    for edge in invalidation_candidates:
        # (Edge invalid before new edge becomes valid) or (new edge invalid before edge becomes valid)
        edge_invalid_at_utc = ensure_utc(edge.invalid_at)
        resolved_edge_valid_at_utc = ensure_utc(resolved_edge.valid_at)
        edge_valid_at_utc = ensure_utc(edge.valid_at)
        resolved_edge_invalid_at_utc = ensure_utc(resolved_edge.invalid_at)

        if (
            edge_invalid_at_utc is not None
            and resolved_edge_valid_at_utc is not None
            and edge_invalid_at_utc <= resolved_edge_valid_at_utc
        ) or (
            edge_valid_at_utc is not None
            and resolved_edge_invalid_at_utc is not None
            and resolved_edge_invalid_at_utc <= edge_valid_at_utc
        ):
            continue
        # New edge invalidates edge
        elif (
            edge_valid_at_utc is not None
            and resolved_edge_valid_at_utc is not None
            and edge_valid_at_utc < resolved_edge_valid_at_utc
        ):
            edge.invalid_at = resolved_edge.valid_at
            edge.expired_at = edge.expired_at if edge.expired_at is not None else utc_now()
            invalidated_edges.append(edge)

    return invalidated_edges


async def resolve_extracted_edge(
    llm_client: LLMClient,
    extracted_edge: EntityEdge,
    related_edges: list[EntityEdge],
    existing_edges: list[EntityEdge],
    episode: EpisodicNode,
    edge_type_candidates: dict[str, type[BaseModel]] | None = None,
    *,
    _bypass_stage1: bool = False,
) -> tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]:
    """Resolve an extracted edge against existing graph context.

    Uses a two-stage approach:
      Stage 1 (Dedup): Check if this fact already exists among edges between
                       the same endpoints. If duplicate found, reuse it — done.
      Stage 2 (Invalidation): Only for genuinely NEW facts. Check if this fact
                              contradicts existing facts on connected entities.

    This avoids wasting LLM calls on invalidation checks for duplicate facts,
    which account for the majority of edges in typical workloads.

    Parameters
    ----------
    llm_client : LLMClient
        Client used to invoke the LLM for deduplication and attribute extraction.
    extracted_edge : EntityEdge
        Newly extracted edge whose canonical representation is being resolved.
    related_edges : list[EntityEdge]
        Candidate edges with identical endpoints used for duplicate detection.
    existing_edges : list[EntityEdge]
        Broader set of edges from connected entities for contradiction detection.
    episode : EpisodicNode
        Episode providing content context when extracting edge attributes.
    edge_type_candidates : dict[str, type[BaseModel]] | None
        Custom edge types permitted for the current source/target signature.
    _bypass_stage1 : bool
        Internal flag. When True, skip fast paths and Stage 1 (already handled
        by the batched dedup in resolve_extracted_edges). Used only by
        resolve_extracted_edges; external callers should not set this.

    Returns
    -------
    tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]
        The resolved edge, any invalidated edges, and duplicate edges.
    """
    start = time()

    if not _bypass_stage1:
        # --- Fast path: no candidates at all ---
        if len(related_edges) == 0 and len(existing_edges) == 0:
            await _extract_edge_attributes(llm_client, extracted_edge, episode, edge_type_candidates)
            logger.debug(
                'EDGE_RESOLVE_STAGE: edge %s — no candidates, skipped both LLM calls (%.0f ms)',
                extracted_edge.uuid, (time() - start) * 1000,
            )
            return extracted_edge, [], []

        # --- Fast path: exact text match ---
        normalized_fact = _normalize_string_exact(extracted_edge.fact)
        for edge in related_edges:
            if (
                edge.source_node_uuid == extracted_edge.source_node_uuid
                and edge.target_node_uuid == extracted_edge.target_node_uuid
                and _normalize_string_exact(edge.fact) == normalized_fact
            ):
                resolved = edge
                if episode is not None and episode.uuid not in resolved.episodes:
                    resolved.episodes.append(episode.uuid)
                logger.debug(
                    'EDGE_RESOLVE_STAGE: edge %s — exact text match, skipped both LLM calls (%.0f ms)',
                    extracted_edge.uuid, (time() - start) * 1000,
                )
                return resolved, [], []

        # --- Stage 1 pre-filter: drop low-similarity dedup candidates ---
        # Candidates with None embeddings score 0.0 via _cosine_sim and are dropped — consistent
        # with stage 2 behavior. The None-embedding fallback below only applies to the extracted edge.
        if extracted_edge.fact_embedding is None:
            logger.warning(
                'EDGE_RESOLVE_STAGE1_DEDUP: edge %s — fact_embedding is None, skipping similarity pre-filter',
                extracted_edge.uuid,
            )
        else:
            filtered = [
                c
                for c in related_edges
                if _cosine_sim(extracted_edge.fact_embedding, c.fact_embedding) >= DEDUP_SIM_THRESHOLD
            ]
            if len(filtered) < len(related_edges):
                logger.debug(
                    'EDGE_RESOLVE_STAGE1_DEDUP: edge %s — dropped %d of %d candidates below DEDUP_SIM_THRESHOLD=%.2f',
                    extracted_edge.uuid,
                    len(related_edges) - len(filtered),
                    len(related_edges),
                    DEDUP_SIM_THRESHOLD,
                )
            related_edges = filtered

        # =====================================================================
        # STAGE 1: Dedup — is this fact already known between these endpoints?
        # =====================================================================
        resolved_edge = extracted_edge
        duplicate_edges: list[EntityEdge] = []

        if related_edges:
            t_dedup = time()
            dedup_context = {
                'existing_edges': [
                    {'idx': i, 'fact': edge.fact} for i, edge in enumerate(related_edges)
                ],
                'new_edge': extracted_edge.fact,
                'edge_invalidation_candidates': [],  # empty — dedup only
            }

            logger.debug(
                'EDGE_RESOLVE_STAGE1_DEDUP: edge "%s" — checking %d related edges',
                extracted_edge.fact[:60], len(related_edges),
            )

            llm_response = await llm_client.generate_response(
                prompt_library.dedupe_edges.resolve_edge(dedup_context),
                response_model=EdgeDuplicate,
                model_size=ModelSize.small,
                prompt_name='dedupe_edges.resolve_edge',
            )
            response_object = EdgeDuplicate(**llm_response)

            duplicate_fact_ids: list[int] = [
                i for i in response_object.duplicate_facts if 0 <= i < len(related_edges)
            ]

            if duplicate_fact_ids:
                resolved_edge = related_edges[duplicate_fact_ids[0]]
                if episode is not None:
                    resolved_edge.episodes.append(episode.uuid)
                duplicate_edges = [related_edges[idx] for idx in duplicate_fact_ids]

                await _extract_edge_attributes(llm_client, resolved_edge, episode, edge_type_candidates)

                logger.debug(
                    'EDGE_RESOLVE_STAGE: edge %s — duplicate found, skipped invalidation LLM call (%.0f ms)',
                    extracted_edge.uuid, (time() - start) * 1000,
                )
                return resolved_edge, [], duplicate_edges

            logger.debug(
                'EDGE_RESOLVE_STAGE1_DEDUP: no duplicate found (%.0f ms)',
                (time() - t_dedup) * 1000,
            )
    else:
        resolved_edge = extracted_edge
        duplicate_edges = []

    # =========================================================================
    # STAGE 2: Invalidation — does this NEW fact contradict existing facts?
    # Only reached for genuinely new edges (not duplicates).
    # =========================================================================
    invalidated_edges: list[EntityEdge] = []

    if existing_edges:
        t_invalidation = time()
        invalidation_context = {
            'existing_edges': [],  # empty — invalidation only
            'new_edge': extracted_edge.fact,
            'edge_invalidation_candidates': [
                {'idx': i, 'fact': edge.fact} for i, edge in enumerate(existing_edges)
            ],
        }

        logger.debug(
            'EDGE_RESOLVE_STAGE2_INVALIDATION: edge "%s" — checking %d endpoint edges',
            extracted_edge.fact[:60],
            len(existing_edges),
        )

        llm_response = await llm_client.generate_response(
            prompt_library.dedupe_edges.resolve_edge(invalidation_context),
            response_model=EdgeDuplicate,
            model_size=ModelSize.small,
            prompt_name='dedupe_edges.resolve_edge',
        )
        response_object = EdgeDuplicate(**llm_response)

        contradicted_facts: list[int] = [
            i for i in response_object.contradicted_facts if 0 <= i < len(existing_edges)
        ]

        invalidation_candidates = [existing_edges[idx] for idx in contradicted_facts]

        logger.debug(
            'EDGE_RESOLVE_STAGE2_INVALIDATION: %d contradictions found (%.0f ms)',
            len(invalidation_candidates),
            (time() - t_invalidation) * 1000,
        )

        # Apply temporal invalidation logic
        now = utc_now()

        if resolved_edge.invalid_at and not resolved_edge.expired_at:
            resolved_edge.expired_at = now

        if resolved_edge.expired_at is None:
            invalidation_candidates.sort(key=lambda c: (c.valid_at is None, ensure_utc(c.valid_at)))
            for candidate in invalidation_candidates:
                candidate_valid_at_utc = ensure_utc(candidate.valid_at)
                resolved_edge_valid_at_utc = ensure_utc(resolved_edge.valid_at)
                if (
                    candidate_valid_at_utc is not None
                    and resolved_edge_valid_at_utc is not None
                    and candidate_valid_at_utc > resolved_edge_valid_at_utc
                ):
                    resolved_edge.invalid_at = candidate.valid_at
                    resolved_edge.expired_at = now
                    break

        invalidated_edges = resolve_edge_contradictions(resolved_edge, invalidation_candidates)
    else:
        now = utc_now()
        if resolved_edge.invalid_at and not resolved_edge.expired_at:
            resolved_edge.expired_at = now

    await _extract_edge_attributes(llm_client, resolved_edge, episode, edge_type_candidates)

    logger.debug(
        'EDGE_RESOLVE_STAGE: edge %s -> %s, %d invalidations, total %.0f ms',
        extracted_edge.uuid,
        resolved_edge.uuid,
        len(invalidated_edges),
        (time() - start) * 1000,
    )

    return resolved_edge, invalidated_edges, duplicate_edges


async def filter_existing_duplicate_of_edges(
    driver: GraphDriver, duplicates_node_tuples: list[tuple[EntityNode, EntityNode]]
) -> list[tuple[EntityNode, EntityNode]]:
    if not duplicates_node_tuples:
        return []

    duplicate_nodes_map = {
        (source.uuid, target.uuid): (source, target) for source, target in duplicates_node_tuples
    }

    if driver.provider == GraphProvider.NEPTUNE:
        query: LiteralString = """
            UNWIND $duplicate_node_uuids AS duplicate_tuple
            MATCH (n:Entity {uuid: duplicate_tuple.source})-[r:RELATES_TO {name: 'IS_DUPLICATE_OF'}]->(m:Entity {uuid: duplicate_tuple.target})
            RETURN DISTINCT
                n.uuid AS source_uuid,
                m.uuid AS target_uuid
        """

        duplicate_nodes = [
            {'source': source.uuid, 'target': target.uuid}
            for source, target in duplicates_node_tuples
        ]

        records, _, _ = await driver.execute_query(
            query,
            duplicate_node_uuids=duplicate_nodes,
            routing_='r',
        )
    else:
        if driver.provider == GraphProvider.LADYBUG:
            query = """
                UNWIND $duplicate_node_uuids AS duplicate
                MATCH (n:Entity {uuid: duplicate.src})-[:RELATES_TO]->(e:RelatesToNode_ {name: 'IS_DUPLICATE_OF'})-[:RELATES_TO]->(m:Entity {uuid: duplicate.dst})
                RETURN DISTINCT
                    n.uuid AS source_uuid,
                    m.uuid AS target_uuid
            """
            duplicate_node_uuids = [{'src': src, 'dst': dst} for src, dst in duplicate_nodes_map]
        else:
            query: LiteralString = """
                UNWIND $duplicate_node_uuids AS duplicate_tuple
                MATCH (n:Entity {uuid: duplicate_tuple[0]})-[r:RELATES_TO {name: 'IS_DUPLICATE_OF'}]->(m:Entity {uuid: duplicate_tuple[1]})
                RETURN DISTINCT
                    n.uuid AS source_uuid,
                    m.uuid AS target_uuid
            """
            duplicate_node_uuids = list(duplicate_nodes_map.keys())

        records, _, _ = await driver.execute_query(
            query,
            duplicate_node_uuids=duplicate_node_uuids,
            routing_='r',
        )

    # Remove duplicates that already have the IS_DUPLICATE_OF edge
    for record in records:
        duplicate_tuple = (record.get('source_uuid'), record.get('target_uuid'))
        if duplicate_nodes_map.get(duplicate_tuple):
            duplicate_nodes_map.pop(duplicate_tuple)

    return list(duplicate_nodes_map.values())
