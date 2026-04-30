"""
Layout invariant tests for restructured extract_nodes.py prompt functions.

Each test asserts:
  - messages[0].role == 'system'
  - len(messages[0].content) > 0
  - A key dynamic token appears in messages[1].content (not in the system message)
"""

import pytest

from graphiti_core.prompts.extract_nodes import (
    classify_nodes,
    extract_attributes,
    extract_entity_summaries_from_episodes,
    extract_json,
    extract_message,
    extract_summaries_batch,
    extract_summary,
    reclassify_entity,
)

EPISODE = 'UNIQUE_EPISODE_CONTENT_ABC123'


@pytest.fixture
def base_context():
    return {
        'previous_episodes': [],
        'episode_content': EPISODE,
        'entity_types': 'Person: A human being\nLocation: A place',
        'freeform_entity_types': False,
        'custom_extraction_instructions': '',
    }


def test_extract_message_layout(base_context):
    messages = extract_message(base_context)
    assert messages[0].role == 'system'
    assert len(messages[0].content) > 0
    assert EPISODE in messages[1].content
    assert EPISODE not in messages[0].content


def test_extract_message_layout_freeform():
    context = {
        'previous_episodes': [],
        'episode_content': EPISODE,
        'freeform_entity_types': True,
        'custom_extraction_instructions': '',
    }
    messages = extract_message(context)
    assert messages[0].role == 'system'
    assert len(messages[0].content) > 0
    assert EPISODE in messages[1].content
    assert EPISODE not in messages[0].content


def test_extract_json_layout(base_context):
    context = dict(base_context)
    context['source_description'] = 'UNIQUE_SOURCE_DESC_XYZ789'
    messages = extract_json(context)
    assert messages[0].role == 'system'
    assert len(messages[0].content) > 0
    assert EPISODE in messages[1].content
    assert EPISODE not in messages[0].content
    assert 'UNIQUE_SOURCE_DESC_XYZ789' in messages[1].content


def test_extract_summaries_batch_layout(base_context):
    context = dict(base_context)
    context['entities'] = [{'name': 'Jordan Lee', 'summary': ''}]
    messages = extract_summaries_batch(context)
    assert messages[0].role == 'system'
    assert len(messages[0].content) > 0
    assert EPISODE in messages[1].content
    assert EPISODE not in messages[0].content


def test_classify_nodes_layout():
    context = {
        'previous_episodes': [],
        'episode_content': EPISODE,
        'extracted_entities': 'Jordan, Denver',
        'entity_types': 'Person: A human being\nLocation: A place',
    }
    messages = classify_nodes(context)
    assert messages[0].role == 'system'
    assert len(messages[0].content) > 0
    assert EPISODE in messages[1].content
    assert EPISODE not in messages[0].content


def test_reclassify_entity_layout():
    context = {
        'entity_name': 'UNIQUE_ENTITY_NAME_FOO456',
        'entity_summary': 'UNIQUE_ENTITY_SUMMARY_BAR789',
    }
    messages = reclassify_entity(context)
    assert messages[0].role == 'system'
    assert len(messages[0].content) > 0
    assert 'UNIQUE_ENTITY_NAME_FOO456' in messages[1].content
    assert 'UNIQUE_ENTITY_SUMMARY_BAR789' in messages[1].content
    assert 'UNIQUE_ENTITY_NAME_FOO456' not in messages[0].content


def test_extract_attributes_layout():
    context = {
        'previous_episodes': [],
        'episode_content': EPISODE,
        'node': 'UNIQUE_NODE_CONTENT_QRS321',
    }
    messages = extract_attributes(context)
    assert messages[0].role == 'system'
    assert len(messages[0].content) > 0
    assert EPISODE in messages[1].content
    assert 'UNIQUE_NODE_CONTENT_QRS321' in messages[1].content
    assert EPISODE not in messages[0].content


def test_extract_summary_layout():
    context = {
        'previous_episodes': [],
        'episode_content': EPISODE,
        'node': 'UNIQUE_NODE_CONTENT_LMN654',
    }
    messages = extract_summary(context)
    assert messages[0].role == 'system'
    assert len(messages[0].content) > 0
    assert EPISODE in messages[1].content
    assert 'UNIQUE_NODE_CONTENT_LMN654' in messages[1].content
    assert EPISODE not in messages[0].content


def test_extract_entity_summaries_from_episodes_layout():
    context = {
        'previous_episodes': [],
        'episode_content': EPISODE,
        'entities': [{'name': 'Jordan Lee', 'summary': ''}],
    }
    messages = extract_entity_summaries_from_episodes(context)
    assert messages[0].role == 'system'
    assert len(messages[0].content) > 0
    assert EPISODE in messages[1].content
    assert EPISODE not in messages[0].content
