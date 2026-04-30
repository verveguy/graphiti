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

from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class EdgeDuplicate(BaseModel):
    duplicate_facts: list[int] = Field(
        ...,
        description='List of idx values of duplicate facts (only from EXISTING FACTS range). Empty list if none.',
    )
    contradicted_facts: list[int] = Field(
        ...,
        description='List of idx values of contradicted facts (from full idx range). Empty list if none.',
    )


class EdgeResolution(BaseModel):
    edge_idx: int = Field(
        ...,
        description='Index of the new edge in the batch (0-based, matches the edge_idx provided in the prompt).',
    )
    duplicate_of: int | None = Field(
        ...,
        description='candidate_idx of the existing edge this new edge is a duplicate of, or null if not a duplicate.',
    )


class EdgeBatchResolutions(BaseModel):
    edge_resolutions: list[EdgeResolution] = Field(
        ...,
        description='List of per-edge dedup decisions. Must include one entry per edge in the batch.',
    )


class Prompt(Protocol):
    resolve_edge: PromptVersion
    resolve_edges_batch: PromptVersion


class Versions(TypedDict):
    resolve_edge: PromptFunction
    resolve_edges_batch: PromptFunction


def resolve_edge(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a fact deduplication assistant. '
            'NEVER mark facts with key differences as duplicates.',
        ),
        Message(
            role='user',
            content=f"""<EXISTING FACTS>
{context['existing_edges']}
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
{context['edge_invalidation_candidates']}
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
{context['new_edge']}
</NEW FACT>

NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

IMPORTANT constraints:
- duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

1. DUPLICATE DETECTION:
   - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
   - If no duplicates, return an empty list for duplicate_facts.

2. CONTRADICTION DETECTION:
   - Determine which facts the NEW FACT contradicts from either list.
   - A fact from EXISTING FACTS can be both a duplicate AND contradicted (e.g., semantically the same but the new fact updates/supersedes it).
   - Return all contradicted idx values in contradicted_facts.
   - If no contradictions, return an empty list for contradicted_facts.

<EXAMPLE>
EXISTING FACT: idx=0, "Alice joined Acme Corp in 2020"
NEW FACT: "Alice joined Acme Corp in 2020"
Result: duplicate_facts=[0], contradicted_facts=[] (identical factual information)

EXISTING FACT: idx=1, "Alice works at Acme Corp as a software engineer"
NEW FACT: "Alice works at Acme Corp as a senior engineer"
Result: duplicate_facts=[], contradicted_facts=[1] (same relationship but updated title — contradiction, NOT a duplicate)

EXISTING FACT: idx=2, "Bob ran 5 miles on Tuesday"
NEW FACT: "Bob ran 3 miles on Wednesday"
Result: duplicate_facts=[], contradicted_facts=[] (different events on different days — neither duplicate nor contradiction)
</EXAMPLE>
""",
        ),
    ]


def _format_batch_edges_for_prompt(edges: list[dict[str, Any]]) -> str:
    parts = []
    for edge in edges:
        parts.append(f'<EDGE edge_idx="{edge["edge_idx"]}">')
        parts.append(f'  NEW FACT: {edge["fact"]}')
        if edge['candidates']:
            parts.append('  EXISTING CANDIDATES:')
            for candidate in edge['candidates']:
                parts.append(f'    candidate_idx={candidate["candidate_idx"]}: {candidate["fact"]}')
        else:
            parts.append('  EXISTING CANDIDATES: (none)')
        parts.append('</EDGE>')
    return '\n'.join(parts)


def resolve_edges_batch(context: dict[str, Any]) -> list[Message]:
    edges: list[dict[str, Any]] = context['edges']
    edge_indices = ', '.join(str(e['edge_idx']) for e in edges)
    return [
        Message(
            role='system',
            content='You are a fact deduplication assistant. '
            'NEVER mark facts with key differences as duplicates.',
        ),
        Message(
            role='user',
            content=f"""{_format_batch_edges_for_prompt(edges)}

NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

For each NEW FACT below, determine if it is a duplicate of any EXISTING CANDIDATE listed under that fact.
Each new fact has its own candidate list — do NOT compare candidates across different new facts.

For each new fact (identified by edge_idx), return:
- edge_idx: the index of the new fact (as provided above)
- duplicate_of: the candidate_idx of the matching existing fact, or null if not a duplicate

Your response MUST include exactly {len(edges)} resolutions with edge_idx values {edge_indices}.
Do not skip any edge_idx.

A fact is a duplicate only if it represents IDENTICAL factual information.
Do NOT mark a fact as a duplicate if it has different numeric values, different dates, or different qualifiers.

<EXAMPLE>
<EDGE edge_idx="0">
  NEW FACT: "Alice joined Acme Corp in 2020"
  EXISTING CANDIDATES:
    candidate_idx=0: "Alice joined Acme Corp in 2020"
    candidate_idx=1: "Alice works at Acme Corp"
</EDGE>
<EDGE edge_idx="1">
  NEW FACT: "Bob ran 5 miles on Tuesday"
  EXISTING CANDIDATES:
    candidate_idx=0: "Bob ran 3 miles on Wednesday"
</EDGE>
<EDGE edge_idx="2">
  NEW FACT: "Alice is a software engineer at Acme"
  EXISTING CANDIDATES: (none)
</EDGE>

Result:
- edge_idx=0, duplicate_of=0  (identical fact)
- edge_idx=1, duplicate_of=null  (different event)
- edge_idx=2, duplicate_of=null  (no candidates)
</EXAMPLE>
""",
        ),
    ]


versions: Versions = {'resolve_edge': resolve_edge, 'resolve_edges_batch': resolve_edges_batch}
