"""Integration tests for WAL mixed-format replay pipeline.

Tests the shared helpers layer (wal_replay_helpers) with mixed WAL data —
some entries using legacy list[float] embeddings, some using f16: base64
strings. This validates the pipeline shared by both LadybugDB and FalkorDB
replay paths without requiring native database modules.

The LadybugDB replay path calls:
  strip_vecf32_wrappers → expand_bulk_property_set → _deserialize_wal_params
where _deserialize_wal_params uses decode_embedding_param internally.

The FalkorDB replay path applies decode_embedding_param directly.

Both paths converge on decode_embedding_param for embedding handling.
"""

from __future__ import annotations

import json

from graphiti_core.driver.wal_replay_helpers import (
    decode_embedding_param,
    encode_embedding,
    expand_bulk_property_set,
    strip_vecf32_wrappers,
)


def _make_mixed_wal_entries():
    """Generate WAL entries with both legacy list and f16: encoded embeddings."""
    vec_a = [float(i) / 1000.0 for i in range(768)]
    vec_b = [float(i) / 500.0 for i in range(768)]
    return [
        {
            'seq': 0,
            'ts': '2026-03-24T00:00:00Z',
            'db': 'db',
            'cypher': 'MERGE (n:Entity {uuid: $uuid}) SET n.name_embedding = $name_embedding',
            'params': {'uuid': 'a', 'name_embedding': vec_a},
        },
        {
            'seq': 1,
            'ts': '2026-03-24T00:00:01Z',
            'db': 'db',
            'cypher': 'MERGE (n:Entity {uuid: $uuid}) SET n.name_embedding = $name_embedding',
            'params': {'uuid': 'b', 'name_embedding': encode_embedding(vec_b)},
        },
    ]


class TestMixedWalHelperPipeline:
    """Verify the shared helper pipeline handles both embedding formats correctly."""

    def test_decode_both_formats_produce_list_float(self):
        """Legacy list and f16: string both decode to list[float] via decode_embedding_param."""
        entries = _make_mixed_wal_entries()

        results = []
        for entry in entries:
            params = entry['params']
            decoded = {k: decode_embedding_param(v) for k, v in params.items()}
            results.append(decoded)

        # Both entries must yield list[float] for name_embedding
        for result in results:
            emb = result['name_embedding']
            assert isinstance(emb, list), f'Expected list, got {type(emb)}'
            assert len(emb) == 768
            assert all(isinstance(x, float) for x in emb)

    def test_legacy_and_compressed_decode_to_same_values(self):
        """Legacy list and f16: encoding of the same vector decode to equivalent values."""
        vec = [float(i) / 1000.0 for i in range(768)]

        legacy_decoded = decode_embedding_param(vec)
        f16_decoded = decode_embedding_param(encode_embedding(vec))

        assert len(legacy_decoded) == len(f16_decoded) == 768
        for orig, compressed in zip(legacy_decoded, f16_decoded):
            # float16 precision: max deviation ~0.001 for values in [0, 1)
            assert abs(orig - compressed) < 0.001

    def test_pipeline_with_vecf32_wrapped_cypher(self):
        """FalkorDB-era WAL: vecf32 wrapper stripped, f16: embedding decoded correctly."""
        vec = [float(i) / 1000.0 for i in range(768)]
        entry = {
            'cypher': 'MERGE (n:Entity {uuid: $uuid}) SET n.name_embedding = vecf32($name_embedding)',
            'params': {'uuid': 'x', 'name_embedding': encode_embedding(vec)},
        }

        cypher = strip_vecf32_wrappers(entry['cypher'])
        params = {k: decode_embedding_param(v) for k, v in entry['params'].items()}

        assert 'vecf32' not in cypher
        assert '$name_embedding' in cypher
        assert isinstance(params['name_embedding'], list)
        assert len(params['name_embedding']) == 768

    def test_pipeline_with_bulk_property_set_expansion(self):
        """FalkorDB-era WAL with nested props dict: expansion + embedding decode."""
        vec = [float(i) / 1000.0 for i in range(768)]
        entry = {
            'cypher': 'MERGE (n:Entity {uuid: $uuid}) SET n = $props SET n.name_embedding = vecf32($name_embedding)',
            'params': {
                'uuid': 'x',
                'props': {'name': 'Alice', 'summary': 'A node'},
                'name_embedding': encode_embedding(vec),
            },
        }

        cypher = strip_vecf32_wrappers(entry['cypher'])
        cypher, params = expand_bulk_property_set(cypher, entry['params'])
        params = {k: decode_embedding_param(v) for k, v in params.items()}

        # Bulk SET expanded
        assert 'n = $props' not in cypher
        assert 'n.name = $props_name' in cypher
        # Embedding decoded
        assert isinstance(params['name_embedding'], list)
        assert len(params['name_embedding']) == 768

    def test_roundtrip_jsonl_serialize_deserialize(self, tmp_path):
        """Write WAL entries as JSONL (with f16: encoding), parse back, decode correctly."""
        vec = [float(i) / 1000.0 for i in range(768)]
        entry = {
            'seq': 0,
            'ts': '2026-03-24T00:00:00Z',
            'db': 'db',
            'cypher': 'MERGE (n:Entity {uuid: $uuid}) SET n.emb = $emb',
            'params': {'uuid': 'roundtrip', 'emb': encode_embedding(vec)},
        }

        wal_file = tmp_path / 'test.jsonl'
        wal_file.write_text(json.dumps(entry) + '\n')

        with open(wal_file) as f:
            parsed = json.loads(f.readline())

        params = {k: decode_embedding_param(v) for k, v in parsed['params'].items()}

        assert params['uuid'] == 'roundtrip'
        assert isinstance(params['emb'], list)
        assert len(params['emb']) == 768
        for orig, dec in zip(vec, params['emb']):
            assert abs(orig - dec) < 0.001
