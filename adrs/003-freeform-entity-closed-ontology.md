# ADR 003: Closed 16-label ontology for freeform entity extraction

## Status

Accepted

## Context

When no custom `entity_types` are provided by the caller, Graphiti operates in **freeform mode**:
the LLM is responsible for choosing a label for each extracted entity. Previously, the prompt
offered open-ended guidance ("use any concise capitalized single-word label such as Person,
Organization, Software…"). This produced unbounded label diversity across different documents and
LLM calls — `Visualization Data Service` was extracted as `Service`, `System`, and `Technology`
across three chunks of the same RFC corpus. Without a stable label namespace, downstream consumers
cannot filter by label reliably, and the dedup candidate pool fills with parallel nodes that
represent the same entity under different labels.

Additionally, the previous `_promote_resolved_node` function in `dedup_helpers.py` contained an
early-return short-circuit: if the canonical node already carried any specific label it returned
immediately, silently discarding the incoming node's labels. This compounded the label-proliferation
problem — even when two nodes were identified as duplicates, the merge did not union their labels.

A 13-document RFC corpus demonstrated 1234 entity-pair duplicates at threshold 0.85, the majority
of which were same-name-different-label cases. Without label-union merge semantics and a stable
ontology, every additional ingested document would add more parallel entities.

## Decision

1. **Replace open-ended label guidance with a closed 16-label ontology** embedded directly in the
   freeform extraction prompt. The LLM must choose one or more labels from this list per entity:

   | Label | Description |
   |-------|-------------|
   | `Person` | An individual human being |
   | `Organization` | A company, institution, group, or governing body |
   | `Software` | A software application, library, or framework |
   | `Service` | A deployed service, platform, or SaaS product |
   | `System` | An integrated technical system or infrastructure component |
   | `Technology` | A protocol, standard, methodology, or technical approach |
   | `Concept` | An abstract idea, principle, or theoretical framework |
   | `Location` | A physical or virtual place, region, or address |
   | `Event` | A dated or scheduled occurrence or incident |
   | `Process` | A workflow, procedure, or repeatable operation |
   | `Requirement` | A constraint, specification, or policy requirement |
   | `Document` | A document, report, specification, or publication |
   | `Product` | A physical or digital product or deliverable |
   | `Project` | A project, initiative, or program |
   | `Award` | A prize, honor, or recognition |
   | `Book` | A book, novel, or long-form publication |

2. **Change `ExtractedEntityFreeform.entity_type: str` → `entity_types: list[str]`** so the LLM
   can assign multiple labels per entity in a single extraction call. The common case is one label
   per entity; multiple labels are used when an entity genuinely spans categories.

3. **Replace `_promote_resolved_node` short-circuit with full union semantics:**
   `resolved_node.labels = sorted(set(resolved_node.labels) | set(extracted_node.labels))`.
   All three call sites (exact-name match, fuzzy match, LLM-escalated dedup) benefit without
   any call-site changes.

4. **Apply the same union semantics in `_collapse_exact_duplicate_extracted_nodes`** to handle
   same-episode same-name entities that carry different label sets.

5. **Filter-not-skip exclusion semantics:** When `excluded_entity_types` filters one of an
   entity's labels, the entity is kept with the remaining labels rather than discarded entirely.
   An entity is skipped only when all of its specific labels are excluded.

The ontology is defined as `_FREEFORM_ENTITY_ONTOLOGY` in `graphiti_core/prompts/extract_nodes.py`
and rendered into both the `extract_message` and `extract_json`/`extract_text` instruction helpers.

## Rationale

### Why a closed ontology instead of an open-ended one?

Open-ended label guidance produces high label entropy across runs and documents. A closed list lets
the LLM choose confidently from a well-defined namespace, makes downstream consumers' label-based
filtering reliable, and bounds the maximum number of labels an entity can accumulate over many
merges. The 16 labels were chosen to cover the label set already observed in existing extraction
examples plus the domain-relevant labels (`Service`, `System`, `Process`, `Requirement`) that
caused the original label-proliferation problem.

### Why multi-label (`list[str]`) instead of single label per call?

The same real-world entity is often validly described by more than one category across different
contexts. Forcing a single label per extraction call means the first chunk to mention an entity
locks in its label, and subsequent chunks that would justify additional labels cannot contribute.
Multi-label at extraction time is the cleanest representation; it aligns with the union merge
semantics applied at dedup time.

### Why union semantics instead of picking a canonical label at merge time?

An additional LLM call to select a canonical label at merge time adds latency and non-determinism,
and discards valid signal that was deliberately extracted from chunk context. Union semantics are
already implemented in the `merge_entities` edge-resolution path (`graphiti.py:1747–1749`); this
decision makes dedup consistent with that path. The `'Entity'` label is always present in the
sorted list, so no dashboard or API consumer that checks for membership loses access to it.

### Why `sorted()` instead of insertion-order deduplication?

`set()` operations are non-deterministic in iteration order. `sorted()` produces stable,
predictable label lists that are easier to test and log. All existing code that sorts label unions
uses `sorted()` consistently.

## Consequences

- **Positive:** Stable, bounded label namespace; multi-label entities receive all valid labels
  at extraction time; dedup merge no longer silently discards labels; downstream consumers can
  filter by label reliably.
- **Positive:** `'Entity'` is always present in the sorted labels list; label order is deterministic and stable in tests and logs.
- **Neutral:** Existing nodes in the graph retain their pre-migration labels. They will acquire
  additional labels only when they are touched by a new dedup merge.
- **Negative:** The ontology is embedded in the prompt — expanding it requires updating the
  constant and re-testing extraction quality. Hierarchical labels or sub-types are future work.
- **Negative:** Entities ingested before this change may carry labels outside the closed ontology
  (from the previous open-ended guidance). The `_sanitize_label` function still accepts any valid
  identifier, so legacy labels are stored correctly; they just won't appear in the new ontology.
