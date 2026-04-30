"""
Layout invariant tests for restructured summarize_nodes.py prompt functions.

Each test asserts:
  - messages[0].role == 'system'
  - len(messages[0].content) > 0
  - A key dynamic token appears in messages[1].content (not in the system message)
"""

from graphiti_core.prompts.summarize_nodes import (
    summarize_context,
    summarize_pair,
    summary_description,
)

NODE_SUMMARIES = 'UNIQUE_NODE_SUMMARIES_ABC123'
EPISODE = 'UNIQUE_EPISODE_CONTENT_XYZ789'
NODE_NAME = 'UNIQUE_NODE_NAME_FOO456'
NODE_SUMMARY = 'UNIQUE_NODE_SUMMARY_BAR789'
SUMMARY = 'UNIQUE_SUMMARY_QRS321'


def test_summarize_pair_layout():
    context = {'node_summaries': NODE_SUMMARIES}
    messages = summarize_pair(context)
    assert messages[0].role == 'system'
    assert len(messages[0].content) > 0
    assert NODE_SUMMARIES in messages[1].content
    assert NODE_SUMMARIES not in messages[0].content


def test_summarize_context_layout():
    context = {
        'previous_episodes': [],
        'episode_content': EPISODE,
        'node_name': NODE_NAME,
        'node_summary': NODE_SUMMARY,
        'attributes': {},
    }
    messages = summarize_context(context)
    assert messages[0].role == 'system'
    assert len(messages[0].content) > 0
    assert NODE_NAME in messages[1].content
    assert NODE_NAME not in messages[0].content
    assert NODE_SUMMARY in messages[1].content
    assert NODE_SUMMARY not in messages[0].content


def test_summary_description_layout():
    context = {'summary': SUMMARY}
    messages = summary_description(context)
    assert messages[0].role == 'system'
    assert len(messages[0].content) > 0
    assert SUMMARY in messages[1].content
    assert SUMMARY not in messages[0].content
