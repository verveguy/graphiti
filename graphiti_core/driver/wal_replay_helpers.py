"""Pure-Python helpers shared across WAL replay paths.

These helpers are intentionally dependency-free (no database client
imports) so they can be unit-tested without pulling in native extensions
like `real_ladybug` or `falkordb` at test collection time.
"""

from __future__ import annotations

import re

# WAL entries produced by wal_dump.py (the one-time FalkorDB → WAL dump
# path) wrap vector parameters in `vecf32($ident)` because that's the
# FalkorDB Cypher syntax. LadybugDB has no `vecf32()` function — its
# `FLOAT[N]` column type handles the cast implicitly when the parameter
# is supplied as a plain list[float]. This regex rewrites
# `vecf32($x)` or `vecf32($x.y)` back to the bare parameter reference
# `$x` / `$x.y` before execution, so WAL dumps from the FalkorDB era
# can be replayed against a LadybugDB target.
#
# Live writes from LadybugDriver against LadybugDB never emit vecf32()
# (the provider-switched model queries in graphiti_core/models/ have a
# KUZU branch that uses `$name_embedding` directly), so on pure
# LadybugDB-era WAL this substitution is a no-op.
_VECF32_WRAPPER_RE = re.compile(
    r'\bvecf32\(\$([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\)'
)


def strip_vecf32_wrappers(cypher: str) -> str:
    """Remove `vecf32($ident)` wrappers from a Cypher query string.

    Used by `replay_wal_ladybug()` to normalize FalkorDB-era WAL entries
    for execution against LadybugDB. Returns the original string
    unchanged when no wrappers are present (idempotent / no-op safe).
    """
    return _VECF32_WRAPPER_RE.sub(r'$\1', cypher)
