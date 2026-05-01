"""Tests for WAL replay helpers used by `replay_wal_ladybug`.

`strip_vecf32_wrappers` normalizes FalkorDB-era WAL entries (produced
by `wal_dump.py`, which wraps vector parameters in `vecf32($name)`) for
execution against LadybugDB, which has no `vecf32()` function — the
FLOAT[N] column type handles the cast implicitly.

These tests target the helper directly from
`graphiti_core.driver.wal_replay_helpers` (pure Python, no DB imports)
so they run without requiring the `real-ladybug` native module.
"""

import base64

import numpy as np

from graphiti_core.driver.wal_replay_helpers import (
    decode_embedding_param,
    encode_embedding,
    strip_vecf32_wrappers,
)


class TestStripVecf32Wrappers:
    def test_strips_simple_ident(self):
        cypher = 'SET n.name_embedding = vecf32($name_embedding)'
        assert strip_vecf32_wrappers(cypher) == 'SET n.name_embedding = $name_embedding'

    def test_strips_fact_embedding(self):
        cypher = 'SET r.fact_embedding = vecf32($fact_embedding)'
        assert strip_vecf32_wrappers(cypher) == 'SET r.fact_embedding = $fact_embedding'

    def test_strips_dotted_ident(self):
        """wal_dump.py writes bare `$name_embedding`, but the model
        query templates use `$entity_data.name_embedding`. Handle both
        so the strip is safe against any future WAL source."""
        cypher = 'SET n.name_embedding = vecf32($entity_data.name_embedding)'
        assert strip_vecf32_wrappers(cypher) == 'SET n.name_embedding = $entity_data.name_embedding'

    def test_strips_multiple_wrappers_in_one_query(self):
        cypher = (
            'MERGE (n:Entity {uuid: $uuid}) SET n = $props '
            'SET n.name_embedding = vecf32($name_embedding) '
            'SET n.fact_embedding = vecf32($fact_embedding)'
        )
        expected = (
            'MERGE (n:Entity {uuid: $uuid}) SET n = $props '
            'SET n.name_embedding = $name_embedding '
            'SET n.fact_embedding = $fact_embedding'
        )
        assert strip_vecf32_wrappers(cypher) == expected

    def test_noop_on_clean_query(self):
        """LadybugDB-era WAL entries have no vecf32 wrappers — must pass through unchanged."""
        cypher = (
            'MERGE (n:Entity {uuid: $uuid}) SET n.name = $name, n.name_embedding = $name_embedding'
        )
        assert strip_vecf32_wrappers(cypher) == cypher

    def test_noop_on_empty_query(self):
        assert strip_vecf32_wrappers('') == ''

    def test_does_not_match_vecf32_without_param(self):
        """Defensive: only `vecf32($ident)` form should be stripped. A bare
        `vecf32(...)` call with a literal argument (not that we ever emit
        one) must NOT match and must pass through unchanged."""
        cypher = 'SET n.x = vecf32([1.0, 2.0, 3.0])'
        assert strip_vecf32_wrappers(cypher) == cypher

    def test_does_not_match_similar_function_name(self):
        """Defensive: don't match things like `not_vecf32($x)`."""
        cypher = 'SET n.x = not_vecf32($y)'
        assert strip_vecf32_wrappers(cypher) == cypher

    def test_wal_dump_emission_format_from_nodes(self):
        """Exact cypher fragment produced by wal_dump._dump_nodes (line 128)."""
        cypher = 'MERGE (n:Entity {uuid: $uuid}) SET n = $props SET n.name_embedding = vecf32($name_embedding)'
        out = strip_vecf32_wrappers(cypher)
        assert 'vecf32' not in out
        assert 'n.name_embedding = $name_embedding' in out

    def test_wal_dump_emission_format_from_edges(self):
        """Exact cypher fragment produced by wal_dump._dump_relationships (line 210)."""
        cypher = (
            'MATCH (src:Entity {uuid: $src_uuid}) '
            'MATCH (dst:Entity {uuid: $dst_uuid}) '
            'MERGE (src)-[r:RELATES_TO {uuid: $uuid}]->(dst) '
            'SET r = $props '
            'SET r.fact_embedding = vecf32($fact_embedding)'
        )
        out = strip_vecf32_wrappers(cypher)
        assert 'vecf32' not in out
        assert 'r.fact_embedding = $fact_embedding' in out


class TestEncodeEmbedding:
    def test_returns_f16_prefix(self):
        vec = [0.1, 0.2, 0.3] * 30  # 90 elements > 64 threshold
        result = encode_embedding(vec)
        assert result.startswith('f16:')

    def test_base64_payload_is_valid(self):
        vec = [float(i) / 100.0 for i in range(100)]
        result = encode_embedding(vec)
        payload = result[4:]
        # Should decode cleanly without error
        raw = base64.b64decode(payload)
        assert len(raw) == 100 * 2  # 2 bytes per float16

    def test_encode_768_dim(self):
        vec = [float(i) / 1000.0 for i in range(768)]
        result = encode_embedding(vec)
        assert result.startswith('f16:')
        raw = base64.b64decode(result[4:])
        assert len(raw) == 768 * 2


class TestDecodeEmbeddingParam:
    def test_f16_roundtrip_within_tolerance(self):
        """Encode → decode round-trip within float16 precision."""
        vec = [float(i) / 1000.0 for i in range(768)]
        encoded = encode_embedding(vec)
        decoded = decode_embedding_param(encoded)
        assert isinstance(decoded, list)
        assert len(decoded) == 768
        # float16 tolerance is ~0.001 for values in [0, 1)
        for orig, dec in zip(vec, decoded):
            assert abs(orig - dec) < 0.001

    def test_legacy_list_passthrough(self):
        """JSON-array (legacy) lists are returned unchanged."""
        vec = [1.0, 2.0, 3.0]
        assert decode_embedding_param(vec) is vec

    def test_f32_decode(self):
        """f32: prefix is decoded to list[float] correctly."""
        vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        arr = np.array(vec, dtype=np.float32)
        encoded = 'f32:' + base64.b64encode(arr.tobytes()).decode('ascii')
        decoded = decode_embedding_param(encoded)
        assert isinstance(decoded, list)
        assert len(decoded) == 5
        for orig, dec in zip(vec, decoded):
            assert abs(orig - dec) < 1e-6

    def test_non_embedding_string_passthrough(self):
        """Non-embedding strings pass through unchanged."""
        assert decode_embedding_param('hello') == 'hello'
        assert decode_embedding_param('2026-01-01T00:00:00Z') == '2026-01-01T00:00:00Z'

    def test_nested_dict_recursion(self):
        """Dict values are recursed into; non-embedding values pass through."""
        vec = [float(i) / 100.0 for i in range(100)]
        encoded = encode_embedding(vec)
        params = {'embedding': encoded, 'uuid': 'abc-123', 'count': 5}
        result = decode_embedding_param(params)
        assert result['uuid'] == 'abc-123'
        assert result['count'] == 5
        assert isinstance(result['embedding'], list)
        assert len(result['embedding']) == 100

    def test_returns_standard_python_floats(self):
        """Decoded values are Python floats, not numpy scalars."""
        vec = [0.1, 0.2, 0.3] * 30
        encoded = encode_embedding(vec)
        decoded = decode_embedding_param(encoded)
        # Each element must be a Python float, not np.float32 or np.float16
        assert all(type(x) is float for x in decoded)
