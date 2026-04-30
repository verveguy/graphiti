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

from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS

from .models import Message, PromptFunction, PromptVersion
from .prompt_helpers import to_prompt_json
from .snippets import summary_instructions


class ExtractedEntity(BaseModel):
    name: str = Field(..., description='Name of the extracted entity')
    entity_type_id: int = Field(
        description='ID of the classified entity type. '
        'Must be one of the provided entity_type_id integers.',
    )


_FREEFORM_ENTITY_ONTOLOGY: list[tuple[str, str]] = [
    ('Person', 'An individual human being'),
    ('Organization', 'A company, institution, group, or governing body'),
    ('Software', 'A software application, library, or framework'),
    ('Service', 'A deployed service, platform, or SaaS product'),
    ('System', 'An integrated technical system or infrastructure component'),
    ('Technology', 'A protocol, standard, methodology, or technical approach'),
    ('Concept', 'An abstract idea, principle, or theoretical framework'),
    ('Location', 'A physical or virtual place, region, or address'),
    ('Event', 'A dated or scheduled occurrence or incident'),
    ('Process', 'A workflow, procedure, or repeatable operation'),
    ('Requirement', 'A constraint, specification, or policy requirement'),
    ('Document', 'A document, report, specification, or publication'),
    ('Product', 'A physical or digital product or deliverable'),
    ('Project', 'A project, initiative, or program'),
    ('Award', 'A prize, honor, or recognition'),
    ('Book', 'A book, novel, or long-form publication'),
]

_FREEFORM_ONTOLOGY_TEXT = '\n'.join(
    f'   - {label}: {desc}' for label, desc in _FREEFORM_ENTITY_ONTOLOGY
)


class ExtractedEntityFreeform(BaseModel):
    name: str = Field(..., description='Name of the extracted entity')
    entity_types: list[str] = Field(
        description='One or more ontological labels that describe this entity. '
        'Choose from the closed ontology list provided in the instructions. '
        'Most entities have a single label; use multiple only when an entity '
        'genuinely belongs to more than one category (e.g., a software library '
        'that is also a service).',
    )


class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity] = Field(..., description='List of extracted entities')


class ExtractedEntitiesFreeform(BaseModel):
    extracted_entities: list[ExtractedEntityFreeform] = Field(
        ..., description='List of extracted entities'
    )


class ReclassifiedEntity(BaseModel):
    entity_type: str = Field(
        ...,
        description='The classified entity type as a PascalCase string'
        ' (e.g., Person, Organization, Concept)',
    )


class EntitySummary(BaseModel):
    summary: str = Field(..., description='Summary of the entity')


class SummarizedEntity(BaseModel):
    name: str = Field(..., description='Name of the entity being summarized')
    summary: str = Field(..., description='Updated summary for the entity')


class SummarizedEntities(BaseModel):
    summaries: list[SummarizedEntity] = Field(
        ...,
        description='List of entity summaries. Only include entities that need summary updates.',
    )


class Prompt(Protocol):
    extract_message: PromptVersion
    extract_json: PromptVersion
    extract_text: PromptVersion
    classify_nodes: PromptVersion
    extract_attributes: PromptVersion
    extract_summary: PromptVersion
    extract_summaries_batch: PromptVersion
    reclassify_entity: PromptVersion
    extract_entity_summaries_from_episodes: PromptVersion


class Versions(TypedDict):
    extract_message: PromptFunction
    extract_json: PromptFunction
    extract_text: PromptFunction
    classify_nodes: PromptFunction
    extract_attributes: PromptFunction
    extract_summary: PromptFunction
    extract_summaries_batch: PromptFunction
    reclassify_entity: PromptFunction
    extract_entity_summaries_from_episodes: PromptFunction


def _entity_type_classification_instructions(context: dict[str, Any]) -> str:
    """Generate entity classification instructions based on whether custom types are provided."""
    if context.get('freeform_entity_types'):
        return f"""3. **Entity Classification**:
   - Assign one or more `entity_types` labels from the closed ontology below.
   - Most entities have a single label; use multiple only when the entity genuinely spans categories.
   - Choose only from this list:
{_FREEFORM_ONTOLOGY_TEXT}"""
    else:
        return """3. **Entity Classification**:
   - Use the descriptions in ENTITY TYPES to classify each extracted entity.
   - Assign the appropriate `entity_type_id` for each one."""


def _classification_instruction_inline(context: dict[str, Any]) -> str:
    """Generate inline classification instruction for extract_json and extract_text prompts."""
    if context.get('freeform_entity_types'):
        return (
            'For each entity extracted, assign one or more `entity_types` labels from this closed ontology:\n'
            f'{_FREEFORM_ONTOLOGY_TEXT}\n'
            'Most entities have a single label; use multiple only when the entity genuinely spans categories.'
        )
    return (
        'For each entity extracted, also determine its entity type based on the provided ENTITY TYPES '
        'and their descriptions.\nIndicate the classified entity type by providing its entity_type_id.'
    )


def _entity_types_section(context: dict[str, Any]) -> str:
    """Generate the ENTITY TYPES section, omitting it entirely for freeform mode."""
    if context.get('freeform_entity_types'):
        return ''
    return f"""<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>
"""


def extract_message(context: dict[str, Any]) -> list[Message]:
    sys_prompt = f"""You are an entity extraction specialist for conversational messages. NEVER extract abstract concepts, feelings, or generic words.

NEVER extract any of the following:
- Pronouns (you, me, I, he, she, they, we, us, it, them, him, her, this, that, those)
- Abstract concepts or feelings (joy, balance, growth, resilience, happiness, passion, motivation)
- Generic common nouns or bare object words (day, life, people, work, stuff, things, food, time,
  way, tickets, supplies, clothes, keys, gear)
- Generic media/content nouns unless uniquely identified in the node name itself (photo, pic, picture,
  image, video, post, story)
- Generic event/activity nouns unless uniquely identified in the node name itself (event, game, meeting,
  class, workshop, competition)
- Broad institutional nouns unless explicitly named or uniquely qualified (government, school, company,
  team, office)
- Ambiguous bare nouns whose meaning depends on sentence context rather than the node name itself
- Sentence fragments or clauses ("what you really care about", "results of that effort")
- Adjectives or descriptive phrases ("amazing", "something different", "new hair color")
- Duplicate references to the same real-world entity. Extract each entity at most once per message,
  even if it appears multiple times or both as a speaker label and in the body text.
- Bare relational or kinship terms (dad, mom, mother, father, sister, brother, husband, wife,
  spouse, son, daughter, uncle, aunt, cousin, grandma, grandpa, friend, boss, teacher, neighbor,
  roommate) and bare animal/pet words (dog, cat, pet, puppy, kitten). These are too generic on
  their own. Instead, qualify them with the possessor: extract "Nisha's dad" not "dad",
  "Jordan's dog" not "dog".
- Bare generic objects that cannot be meaningfully qualified with a possessor, brand, or
  distinguishing detail (e.g., NEVER extract "supplies" from "I picked up some supplies")

Your task is to extract **entity nodes** that are **explicitly** mentioned in the CURRENT MESSAGE.
Pronoun references such as he/she/they or this/that/those should be disambiguated to the names of the
reference entities. Only extract distinct entities from the CURRENT MESSAGE.

{_entity_types_section(context)}
1. **Speaker Extraction**: Always extract the speaker (the part before the colon `:` in each dialogue line) as the first entity node.
   - If the speaker is mentioned again in the message, treat both mentions as a **single entity**.

2. **Entity Identification**:
   - Extract named entities and specific, concrete things that are **explicitly** mentioned in the CURRENT MESSAGE.
   - Only extract entities that are specific enough to be uniquely identifiable. Ask: "Could this have its own Wikipedia article or database entry?"
   - When a speaker or named person refers to a relative, pet, or associate using a bare term
     (e.g., "my dad", "his cat"), extract the entity qualified with the possessor's name
     (e.g., "Nisha's dad", "Jordan's cat"). Do NOT extract the bare term alone.
   - **Exclude** entities mentioned only in the PREVIOUS MESSAGES (they are for context only).

{_entity_type_classification_instructions(context)}

4. **Exclusions**:
   - Do NOT extract entities representing relationships or actions.
   - Do NOT extract dates, times, or other temporal information — these will be handled separately.
   - When in doubt, do NOT extract.

5. **Specificity**:
   - Always use the **most specific form** mentioned in the message. If the message says "road cycling",
     extract "road cycling" not "cycling". If it says "wool coat", extract "wool coat" not "coat".
   - When context makes an object's type clear, include that context in the name. For example, if the
     message mentions forgetting a leash while discussing a dog walk, extract "dog leash" not "leash".
   - If a phrase would not be distinguishable when read alone later, do NOT extract it.

6. **Formatting**:
   - Be **explicit and unambiguous** in naming entities (e.g., use full names when available).

<EXAMPLE>
Message: "Jordan: We just moved to Denver last month. My spouse started a new role at Lockheed Martin and I enrolled in a ceramics workshop at the Belmont Arts Center."
Good extractions: "Jordan" (speaker), "Denver" (Location), "Lockheed Martin" (Organization), "Belmont Arts Center" (Location), "ceramics" (Concept)
Do NOT extract: "spouse" (generic reference — extract only if named), "new role" (not an entity), "last month" (temporal), "we" (pronoun)
</EXAMPLE>

<EXAMPLE>
Message: "Nisha: My dad is visiting next week. He loves walking his dogs in Riverside Park."
Good extractions: "Nisha" (speaker), "Nisha's dad" (Person), "Riverside Park" (Location)
Do NOT extract: "dad" (bare relational term — qualify as "Nisha's dad"), "dogs" (bare animal word — no specific identity), "next week" (temporal)
</EXAMPLE>

<EXAMPLE>
Message: "Mary: I forgot Trigger's leash so I couldn't take him on a dog walk. After that I went road cycling in my new wool coat."
Good extractions: "Mary" (speaker), "Trigger" (named entity), "dog leash" (Product), "road cycling" (Process), "wool coat" (Product)
Do NOT extract: "leash" (too generic — use "dog leash"), "cycling" (too generic — use "road cycling"), "coat" (too generic — use "wool coat"), "dog walk" (activity, not an entity)
</EXAMPLE>

<EXAMPLE>
Message: "Alex: I shared a pic from the game after the event."
Good extractions: "Alex" (speaker)
Do NOT extract: "pic" (generic media noun), "game" (generic event noun), "event" (generic event noun)
</EXAMPLE>

<EXAMPLE>
Message: "Jordan: We won by a tight score. Scoring that last basket felt incredible."
Good extractions: "Jordan" (speaker)
Do NOT extract: "basket" (ambiguous bare noun that depends on sentence context)
</EXAMPLE>
"""

    user_prompt = f"""<PREVIOUS MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{context['episode_content']}
</CURRENT MESSAGE>

{context['custom_extraction_instructions']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    classification_instruction = _classification_instruction_inline(context)

    sys_prompt = f"""You are an entity extraction specialist for JSON data. NEVER extract abstract concepts, dates, or generic field values.

NEVER extract:
- Date, time, or timestamp values
- Abstract concepts or generic field values (e.g., "true", "active", "pending")
- Numeric IDs or codes that are not meaningful entity names
- Bare relational or kinship terms (e.g., "spouse", "parent", "pet") — only extract if qualified
  with a possessor name
- Bare generic objects or common nouns (e.g., "supplies", "tickets", "gear") — only extract if
  qualified with a distinguishing detail
- Generic media/content nouns unless uniquely identified in the value itself (photo, pic, picture,
  image, video, post, story)
- Generic event/activity nouns unless uniquely identified in the value itself (event, game, meeting,
  class, workshop, competition)
- Broad institutional nouns unless explicitly named or uniquely qualified (government, school, company,
  team, office)
- Ambiguous bare nouns whose meaning depends on surrounding text rather than the extracted value itself

Extract entities from the JSON and classify each.

{_entity_types_section(context)}
Guidelines:
1. Extract the primary entity the JSON represents (e.g., a "name" or "user" field).
2. Extract named entities referenced in other properties throughout the JSON structure.
3. Only extract entities specific enough to be uniquely identifiable.
4. Be explicit in naming entities — use full names when available.
5. Use the most specific form present in the data (e.g., "road cycling" not "cycling").
6. If a value would not be meaningful and distinguishable when read alone later, do NOT extract it.

<EXAMPLE>
JSON: {{"user": "Jordan Lee", "company": "Acme Corp", "role": "engineer", "start_date": "2024-01-15", "location": "Denver", "active": true}}
Good extractions: "Jordan Lee" (Person), "Acme Corp" (Organization), "Denver" (Location)
Do NOT extract: "engineer" (role, not an entity), "2024-01-15" (date), "true" (field value)
</EXAMPLE>

<EXAMPLE>
JSON: {{"author": "Alex", "attachment_type": "photo", "event_name": "event", "agency": "government"}}
Good extractions: "Alex" (Person)
Do NOT extract: "photo" (generic media noun), "event" (generic event noun), "government" (broad institutional noun)
</EXAMPLE>
"""

    user_prompt = f"""<SOURCE DESCRIPTION>
{context['source_description']}
</SOURCE DESCRIPTION>

<JSON>
{context['episode_content']}
</JSON>

{context['custom_extraction_instructions']}
Given the above source description and JSON, extract relevant entities from the provided JSON.
{classification_instruction}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    classification_instruction = _classification_instruction_inline(context)

    # Restructured prompt (cache-friendly):
    # - All stable instructions, the entity-types section, classification rules,
    #   and examples live in the system message.
    # - User message contains only the per-episode dynamic content (TEXT and any
    #   custom_extraction_instructions provided at call time).
    # When LLMConfig.cache_mode='system-block' is enabled and the system block
    # clears the per-model cacheable threshold (~2048 tokens for Sonnet 4.x,
    # ~4500 for Haiku 4.x — possibly via cache_padding_text), this layout
    # produces real cache reads on calls 2..N.
    sys_prompt = f"""You are an entity extraction specialist for unstructured text.
NEVER extract abstract concepts, feelings, or generic words.

NEVER extract:
- Pronouns (you, me, he, she, they, it, them, him, her, we, us, this, that, those)
- Abstract concepts (joy, balance, growth, resilience, passion, motivation)
- Generic common nouns or bare object words (day, life, people, work, stuff, things, food, time,
  tickets, supplies, clothes, keys, gear)
- Generic media/content nouns unless uniquely identified in the node name itself (photo, pic, picture,
  image, video, post, story)
- Generic event/activity nouns unless uniquely identified in the node name itself (event, game, meeting,
  class, workshop, competition)
- Broad institutional nouns unless explicitly named or uniquely qualified (government, school, company,
  team, office)
- Ambiguous bare nouns whose meaning depends on sentence context rather than the node name itself
- Sentence fragments or clauses as entity names
- Bare relational or kinship terms (dad, mom, sister, brother, spouse, friend, boss, pet, dog,
  cat) unless qualified with a possessor (e.g., "Nisha's dad" is acceptable, "dad" alone is not)
- Bare generic objects that cannot be meaningfully qualified with a possessor, brand, or
  distinguishing detail (e.g., NEVER extract "supplies" from "I picked up some supplies")

Extract entities from the TEXT that are **explicitly mentioned**.
Only extract entities specific enough to be uniquely identifiable — ask: "Could this have its own Wikipedia article or database entry?"

{_entity_types_section(context)}
Guidelines:
1. Extract named entities and specific, concrete things.
2. Do not create nodes for relationships or actions.
3. Do not create nodes for temporal information like dates, times or years.
4. Be explicit in node names, using full names and avoiding abbreviations.
5. Always use the most specific form from the text (e.g., "road cycling" not "cycling",
   "wool coat" not "coat"). Include qualifying context when it's clear from the text.
6. When the text refers to a person's relative, pet, or associate by a bare term, qualify the
   entity with the possessor's name (e.g., "Dr. Osei's colleague" not "colleague").
7. If a phrase would not be meaningful and distinguishable when read alone later, do NOT extract it.
8. When in doubt, do NOT extract.

{classification_instruction}

<EXAMPLE>
Text: "Dr. Amara Osei presented her migraine study results at the AAN conference. The study tracked 340 patients using a new CGRP combination protocol."
Good extractions: "Dr. Amara Osei" (Person), "AAN" (Organization), "migraine study" (Document), "CGRP combination protocol" (Process)
Do NOT extract: "results" (generic noun), "340" (number), "patients" (generic noun), "conference" (generic without a specific name)
</EXAMPLE>

<EXAMPLE>
Text: "Alex shared a pic after the event and said scoring the last basket felt incredible."
Good extractions: "Alex" (Person)
Do NOT extract: "pic" (generic media noun), "event" (generic event noun), "basket" (ambiguous bare noun)
</EXAMPLE>
"""

    user_prompt = f"""<TEXT>
{context['episode_content']}
</TEXT>

{context['custom_extraction_instructions']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = (
        'You are an entity classification specialist. '
        'NEVER assign types not listed in ENTITY TYPES.'
    )

    user_prompt = f"""<PREVIOUS MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{context['episode_content']}
</CURRENT MESSAGE>

<EXTRACTED ENTITIES>
{context['extracted_entities']}
</EXTRACTED ENTITIES>

<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

Given the above conversation, extracted entities, and provided entity types and their descriptions, classify the extracted entities.

Guidelines:
1. Each entity must have exactly one type.
2. NEVER use types not listed in ENTITY TYPES.
3. If none of the provided entity types accurately classify an extracted entity, the type should be set to None.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an entity attribute extraction specialist. NEVER hallucinate or infer values not explicitly stated.

Given the MESSAGES and the following ENTITY, update any of its attributes based on the information provided in MESSAGES. Use the provided attribute descriptions to better understand how each attribute should be determined.

Guidelines:
1. NEVER hallucinate or infer property values — only use values explicitly stated in the MESSAGES.
2. Only use the provided MESSAGES and ENTITY to set attribute values."""

    user_prompt = f"""<MESSAGES>
{to_prompt_json(context['previous_episodes'])}
{to_prompt_json(context['episode_content'])}
</MESSAGES>

<ENTITY>
{context['node']}
</ENTITY>
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_summary(context: dict[str, Any]) -> list[Message]:
    sys_prompt = f"""You are a helpful assistant that extracts entity summaries from the provided text.

Given the MESSAGES and the ENTITY, update the summary that combines relevant information about the entity from the messages and relevant information from the existing summary. Summary must be under {MAX_SUMMARY_CHARS} characters.

{summary_instructions}"""

    user_prompt = f"""<MESSAGES>
{to_prompt_json(context['previous_episodes'])}
{to_prompt_json(context['episode_content'])}
</MESSAGES>

<ENTITY>
{context['node']}
</ENTITY>
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_summaries_batch(context: dict[str, Any]) -> list[Message]:
    sys_prompt = (
        'You are a helpful assistant that generates concise entity summaries from provided context.\n\n'
        f'Given the MESSAGES and a list of ENTITIES, generate an updated summary for each entity '
        f'that needs one.\nEach summary must be under {MAX_SUMMARY_CHARS} characters.\n\n'
        f'{summary_instructions}\n'
        'For each entity, combine relevant information from the MESSAGES with any existing summary content.\n'
        'Only return summaries for entities that have meaningful information to summarize.\n'
        'If an entity has no relevant information in the messages and no existing summary, you may skip it.'
    )

    user_prompt = f"""<MESSAGES>
{to_prompt_json(context['previous_episodes'])}
{to_prompt_json(context['episode_content'])}
</MESSAGES>

<ENTITIES>
{to_prompt_json(context['entities'])}
</ENTITIES>
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def _strip_xml_tags(text: str) -> str:
    """Strip XML-like tags from text to prevent prompt injection via delimiter escape."""
    import re

    return re.sub(r'</?[A-Z_]+\s*>', '', text)


def reclassify_entity(context: dict[str, Any]) -> list[Message]:
    sys_prompt = (
        'You are an AI assistant that classifies entities into semantic ontological types. '
        'Given an entity name and its summary, determine the most appropriate type for the entity. '
        'Return a single PascalCase type name that best describes the entity.\n\n'
        'Classify this entity into a single semantic type. Choose a meaningful ontological type such as:\n'
        'Person, Organization, Location, Event, Concept, Technology, Product, Document, Project, etc.\n\n'
        'Guidelines:\n'
        '1. Return exactly one PascalCase type name (e.g., "Person", "Organization", "Concept").\n'
        '2. Choose the most specific applicable type — prefer "Person" over "Entity" when the entity is clearly a person.\n'
        '3. If the entity cannot be meaningfully classified beyond "Entity", return "Entity".\n'
        '4. Do not invent overly specific or compound types — keep types general and reusable.'
    )

    entity_name = _strip_xml_tags(str(context['entity_name']))
    entity_summary = _strip_xml_tags(str(context['entity_summary']))

    user_prompt = f"""<ENTITY NAME>
{entity_name}
</ENTITY NAME>

<ENTITY SUMMARY>
{entity_summary}
</ENTITY SUMMARY>
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


_entity_episode_summary_system_prompt = """You maintain detailed, information-dense entity memories from episode text.

Use ONLY facts explicitly stated in EPISODES and durable facts already present in EXISTING_SUMMARY.
NEVER infer beyond what is directly supported.

Primary goal:
Write a dense factual summary of the entity that preserves as many supported details as possible while staying coherent and durable.

What to capture:
- Stable facts about the entity
- All materially relevant named people, organizations, places, events, documents, objects, and other entities linked to it
- Explicit actions, roles, responsibilities, relationships, and outcomes
- Counts, sequences, and repeated patterns when the evidence supports them
- Temporal details at the highest fidelity available: dates, months, years, ordering, and changes over time
- Current state over superseded state when newer episodes clearly update older information

Rules:
- Be exhaustive within the evidence. Prefer retaining a supported concrete detail over omitting it for brevity.
- NEVER infer preferences, habits, recurrence, frequency, causality, intent, importance, or category \
from a name, a single mention, or weak evidence.
- Only describe something as recurring, preferred, typical, habitual, or ongoing when multiple episodes \
explicitly support that claim or one episode states it directly.
- Include all materially relevant named participants that appear in the evidence.
- Include temporal qualifiers whenever they are available.
- Mention counts when they are directly supported and meaningful. Prefer direct factual phrasing \
over meta phrasing.
- When the durable fact is the content of what was said, state the content directly instead of \
describing that it was said.
- Use communication verbs only when the act of speaking, asking, sharing, presenting, \
announcing, or telling is itself the important fact.
- NEVER manufacture pattern language from a single occurrence. A single mention can support a fact, \
but not a trend, habit, or preference unless the text states that directly.
- If the evidence is insufficient or ambiguous, omit the claim.
- NEVER mention the source material or summarization process.
- NEVER mention episodes, messages, prompts, summaries, memory, graphs, nodes, labels, node types, \
ontology, schema, or categorization.
- NEVER output phrases like "the summary", "the entity", "categorized as", "tagged as", "suggests", \
"implies", "appears to", or "recorded interaction".
- NEVER use "the entity" as a pronoun. Use the entity's actual name or a natural pronoun \
(he, she, it, they).
- NEVER use meta-language verbs like "mentioned", "described", "stated", "noted", "discussed", \
"referenced", "indicated", or "reported". State the fact directly instead of describing how it \
was communicated.
- NEVER begin the summary with "A ", "An ", or "This is". If the entity's name starts with \
"The" (e.g. "The Washington Post"), that is acceptable; otherwise NEVER lead with "The ". \
Lead with the entity's name or a concrete fact.
- When newer episode text conflicts with older summary content, prefer the newer explicit fact.
- If the new episodes add no durable fact, return the existing summary unchanged.
- The summary should read like a compact brief, not a tagline.
- Write 2-6 dense sentences in third person.
- Return only the summary text.

<EXAMPLES>
Input: {"name": "Jordan Lee", "existing_summary": "Jordan Lee works at Belmont Arts Center.", \
"episodes": [{"content": "Mina: Jordan Lee presented a ceramics workshop at Belmont Arts Center on \
March 3, 2025. The workshop had 24 attendees and focused on wheel-thrown bowls.\\nOwen: After the \
session, Jordan announced a second April workshop for returning students."}, {"content": "Mina: Jordan \
shared that the new kiln room opened last month and that Jordan now supervises two studio assistants.\\n\
Owen: Jordan still teaches beginner ceramics on Wednesday evenings."}]}
GOOD: "Jordan Lee works at Belmont Arts Center. Jordan presented a ceramics workshop there on March 3, \
2025 for 24 attendees focused on wheel-thrown bowls, and later announced a second April workshop for \
returning students. Jordan supervises two studio assistants, teaches beginner ceramics on Wednesday \
evenings, and works out of the new kiln room that opened the previous month."
BAD: "Jordan Lee seems interested in ceramics. Jordan mentioned teaching and was described as busy at \
the arts center."
</EXAMPLES>"""


def extract_entity_summaries_from_episodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = f"""{_entity_episode_summary_system_prompt}

NEVER include meta-language about the summarization process. Use ONLY facts from the provided EPISODES.
Each summary must be under {MAX_SUMMARY_CHARS} characters. Write 2-6 dense sentences in third person. Preserve all material names, roles, dates, counts, and changes over time that are explicitly supported.

For each entity below, generate an updated summary using ONLY the provided EPISODES and any existing summary already on the entity.

Only return summaries for entities that have meaningful information to summarize.
If an entity has no relevant information in the episodes and no existing summary, you may skip it."""

    user_prompt = f"""<EPISODES>
{to_prompt_json(context['previous_episodes'])}
{to_prompt_json(context['episode_content'])}
</EPISODES>

<ENTITIES>
{to_prompt_json(context['entities'])}
</ENTITIES>
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


versions: Versions = {
    'extract_message': extract_message,
    'extract_json': extract_json,
    'extract_text': extract_text,
    'extract_summary': extract_summary,
    'extract_summaries_batch': extract_summaries_batch,
    'extract_entity_summaries_from_episodes': extract_entity_summaries_from_episodes,
    'classify_nodes': classify_nodes,
    'extract_attributes': extract_attributes,
    'reclassify_entity': reclassify_entity,
}
