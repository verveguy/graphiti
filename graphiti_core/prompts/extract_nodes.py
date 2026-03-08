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


class ExtractedEntityFreeform(BaseModel):
    name: str = Field(..., description='Name of the extracted entity')
    entity_type: str = Field(
        description='The ontological type that best describes this entity. '
        'Use a concise, capitalized, single-word label. '
        'You are free to use any type that fits — common examples include '
        'Person, Organization, Software, Concept, Location, Event, Award, '
        'Book, Technology, but use whatever label best captures the entity.',
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


class Versions(TypedDict):
    extract_message: PromptFunction
    extract_json: PromptFunction
    extract_text: PromptFunction
    classify_nodes: PromptFunction
    extract_attributes: PromptFunction
    extract_summary: PromptFunction
    extract_summaries_batch: PromptFunction
    reclassify_entity: PromptFunction


def _entity_type_classification_instructions(context: dict[str, Any]) -> str:
    """Generate entity classification instructions based on whether custom types are provided."""
    if context.get('freeform_entity_types'):
        return """3. **Entity Classification**:
   - Assign an `entity_type` label that best describes each entity's ontological category.
   - Use any type that fits — you are not limited to a fixed set. Choose the most descriptive and specific label.
   - Use concise, capitalized, single-word labels (e.g., Person, Book, Award, Technology, Methodology)."""
    else:
        return """3. **Entity Classification**:
   - Use the descriptions in ENTITY TYPES to classify each extracted entity.
   - Assign the appropriate `entity_type_id` for each one."""


def _classification_instruction_inline(context: dict[str, Any]) -> str:
    """Generate inline classification instruction for extract_json and extract_text prompts."""
    if context.get('freeform_entity_types'):
        return (
            'For each entity extracted, assign an `entity_type` label that best describes its '
            'ontological category. Use any type that fits — you are not limited to a fixed set. '
            'Use concise, capitalized, single-word labels.'
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
    sys_prompt = """You are an AI assistant that extracts entity nodes from conversational messages.
    Your primary task is to extract and classify the speaker and other significant entities mentioned in the conversation."""

    user_prompt = f"""
{_entity_types_section(context)}
<PREVIOUS MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{context['episode_content']}
</CURRENT MESSAGE>

Instructions:

You are given a conversation context and a CURRENT MESSAGE. Your task is to extract **entity nodes** mentioned **explicitly or implicitly** in the CURRENT MESSAGE.
Pronoun references such as he/she/they or this/that/those should be disambiguated to the names of the
reference entities. Only extract distinct entities from the CURRENT MESSAGE. Don't extract pronouns like you, me, he/she/they, we/us as entities.

1. **Speaker Extraction**: Always extract the speaker (the part before the colon `:` in each dialogue line) as the first entity node.
   - If the speaker is mentioned again in the message, treat both mentions as a **single entity**.

2. **Entity Identification**:
   - Extract all significant entities, concepts, or actors that are **explicitly or implicitly** mentioned in the CURRENT MESSAGE.
   - **Exclude** entities mentioned only in the PREVIOUS MESSAGES (they are for context only).

{_entity_type_classification_instructions(context)}

4. **Exclusions**:
   - Do NOT extract entities representing relationships or actions.
   - Do NOT extract dates, times, or other temporal information—these will be handled separately.

5. **Formatting**:
   - Be **explicit and unambiguous** in naming entities (e.g., use full names when available).

{context['custom_extraction_instructions']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that extracts entity nodes from JSON.
    Your primary task is to extract and classify relevant entities from JSON files"""

    classification_instruction = _classification_instruction_inline(context)

    user_prompt = f"""
{_entity_types_section(context)}
<SOURCE DESCRIPTION>:
{context['source_description']}
</SOURCE DESCRIPTION>
<JSON>
{context['episode_content']}
</JSON>

{context['custom_extraction_instructions']}

Given the above source description and JSON, extract relevant entities from the provided JSON.
{classification_instruction}

Guidelines:
1. Extract all entities that the JSON represents. This will often be something like a "name" or "user" field
2. Extract all entities mentioned in all other properties throughout the JSON structure
3. Do NOT extract any properties that contain dates
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that extracts entity nodes from text.
    Your primary task is to extract and classify the speaker and other significant entities mentioned in the provided text."""

    classification_instruction = _classification_instruction_inline(context)

    user_prompt = f"""
{_entity_types_section(context)}
<TEXT>
{context['episode_content']}
</TEXT>

Given the above text, extract entities from the TEXT that are explicitly or implicitly mentioned.
{classification_instruction}

{context['custom_extraction_instructions']}

Guidelines:
1. Extract significant entities, concepts, or actors mentioned in the conversation.
2. Avoid creating nodes for relationships or actions.
3. Avoid creating nodes for temporal information like dates, times or years (these will be added to edges later).
4. Be as explicit as possible in your node names, using full names and avoiding abbreviations.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that classifies entity nodes given the context from which they were extracted"""

    user_prompt = f"""
    <PREVIOUS MESSAGES>
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
    1. Each entity must have exactly one type
    2. Only use the provided ENTITY TYPES as types, do not use additional types to classify entities.
    3. If none of the provided entity types accurately classify an extracted node, the type should be set to None
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts entity properties from the provided text.',
        ),
        Message(
            role='user',
            content=f"""
        Given the MESSAGES and the following ENTITY, update any of its attributes based on the information provided
        in MESSAGES. Use the provided attribute descriptions to better understand how each attribute should be determined.

        Guidelines:
        1. Do not hallucinate entity property values if they cannot be found in the current context.
        2. Only use the provided MESSAGES and ENTITY to set attribute values.

        <MESSAGES>
        {to_prompt_json(context['previous_episodes'])}
        {to_prompt_json(context['episode_content'])}
        </MESSAGES>

        <ENTITY>
        {context['node']}
        </ENTITY>
        """,
        ),
    ]


def extract_summary(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts entity summaries from the provided text.',
        ),
        Message(
            role='user',
            content=f"""
        Given the MESSAGES and the ENTITY, update the summary that combines relevant information about the entity
        from the messages and relevant information from the existing summary. Summary must be under {MAX_SUMMARY_CHARS} characters.

        {summary_instructions}

        <MESSAGES>
        {to_prompt_json(context['previous_episodes'])}
        {to_prompt_json(context['episode_content'])}
        </MESSAGES>

        <ENTITY>
        {context['node']}
        </ENTITY>
        """,
        ),
    ]


def extract_summaries_batch(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that generates concise entity summaries from provided context.',
        ),
        Message(
            role='user',
            content=f"""
Given the MESSAGES and a list of ENTITIES, generate an updated summary for each entity that needs one.
Each summary must be under {MAX_SUMMARY_CHARS} characters.

{summary_instructions}

<MESSAGES>
{to_prompt_json(context['previous_episodes'])}
{to_prompt_json(context['episode_content'])}
</MESSAGES>

<ENTITIES>
{to_prompt_json(context['entities'])}
</ENTITIES>

For each entity, combine relevant information from the MESSAGES with any existing summary content.
Only return summaries for entities that have meaningful information to summarize.
If an entity has no relevant information in the messages and no existing summary, you may skip it.
""",
        ),
    ]


def _strip_xml_tags(text: str) -> str:
    """Strip XML-like tags from text to prevent prompt injection via delimiter escape."""
    import re

    return re.sub(r'</?[A-Z_]+\s*>', '', text)


def reclassify_entity(context: dict[str, Any]) -> list[Message]:
    sys_prompt = (
        'You are an AI assistant that classifies entities into semantic ontological types. '
        'Given an entity name and its summary, determine the most appropriate type for the entity. '
        'Return a single PascalCase type name that best describes the entity.'
    )

    entity_name = _strip_xml_tags(str(context['entity_name']))
    entity_summary = _strip_xml_tags(str(context['entity_summary']))

    user_prompt = f"""
<ENTITY NAME>
{entity_name}
</ENTITY NAME>

<ENTITY SUMMARY>
{entity_summary}
</ENTITY SUMMARY>

Classify this entity into a single semantic type. Choose a meaningful ontological type such as:
Person, Organization, Location, Event, Concept, Technology, Product, Document, Project, etc.

Guidelines:
1. Return exactly one PascalCase type name (e.g., "Person", "Organization", "Concept").
2. Choose the most specific applicable type — prefer "Person" over "Entity" when the entity is clearly a person.
3. If the entity cannot be meaningfully classified beyond "Entity", return "Entity".
4. Do not invent overly specific or compound types — keep types general and reusable.
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
    'classify_nodes': classify_nodes,
    'extract_attributes': extract_attributes,
    'reclassify_entity': reclassify_entity,
}
