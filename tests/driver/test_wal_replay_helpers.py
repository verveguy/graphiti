"""Tests for WAL replay helpers used by `replay_wal_ladybug`.

`strip_vecf32_wrappers` normalizes FalkorDB-era WAL entries (produced
by `wal_dump.py`, which wraps vector parameters in `vecf32($name)`) for
execution against LadybugDB, which has no `vecf32()` function — the
FLOAT[N] column type handles the cast implicitly.

These tests target the helper directly from
`graphiti_core.driver.wal_replay_helpers` (pure Python, no DB imports)
so they run without requiring the `real-ladybug` native module.
"""

from graphiti_core.driver.wal_replay_helpers import strip_vecf32_wrappers


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
