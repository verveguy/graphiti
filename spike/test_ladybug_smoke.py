"""
LadybugDB spike smoke test.

Verifies that LadybugDB works as a drop-in replacement for KuzuDB
in the graphiti-core driver layer.

Run from the graphiti repo root:
    uv run python spike/test_ladybug_smoke.py
"""

import asyncio
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def check(name: str, passed: bool, detail: str = ''):
    status = 'PASS' if passed else 'FAIL'
    suffix = f' — {detail}' if detail else ''
    print(f'  [{status}] {name}{suffix}')
    if not passed:
        raise AssertionError(f'{name} failed{suffix}')


async def main():
    print('=== LadybugDB Spike Smoke Test ===\n')

    # --- Check 1: import real_ladybug ---
    print('1. Import real_ladybug')
    try:
        import real_ladybug as kuzu

        check('import real_ladybug', True, f'version {kuzu.__version__}')
    except ImportError as e:
        check('import real_ladybug', False, str(e))
        return 1

    # --- Check 2: import LadybugDriver ---
    print('2. Import LadybugDriver')
    try:
        from graphiti_core.driver.ladybug_driver import LadybugDriver

        check('import LadybugDriver', True)
    except ImportError as e:
        check('import LadybugDriver', False, str(e))
        return 1

    # --- Check 3: schema initialisation ---
    print('3. Schema initialisation')
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / 'test.db')
        try:
            driver = LadybugDriver(db=db_path)
            check('schema init', True)
        except Exception as e:
            check('schema init', False, str(e))
            return 1

        # --- Check 4: Graphiti init + build_indices_and_constraints ---
        print('4. Graphiti init + build_indices_and_constraints')
        try:
            from graphiti_core import Graphiti

            graphiti = Graphiti(driver=driver)
            await graphiti.build_indices_and_constraints()
            check('graphiti init + build_indices', True)
        except Exception as e:
            check('graphiti init + build_indices', False, str(e))
            return 1

        # --- Check 5: ingest 3 episodes ---
        print('5. Ingest 3 episodes')
        episodes = [
            'Alice is a software engineer at Acme Corp.',
            'Bob works with Alice on the search team.',
            'The search team shipped a new ranking algorithm last week.',
        ]
        try:
            for i, text in enumerate(episodes):
                await graphiti.add_episode(
                    name=f'episode_{i}',
                    episode_body=text,
                    source_description='smoke test',
                    reference_time=datetime.now(timezone.utc),
                    group_id='spike_test',
                )
            check('ingest 3 episodes', True)
        except Exception as e:
            check('ingest 3 episodes', False, str(e))
            return 1

        # --- Check 6: search returns results ---
        print('6. Search returns results')
        try:
            results = await graphiti.search(
                query='Who works at Acme?',
                group_ids=['spike_test'],
            )
            check('search returns results', len(results) > 0, f'{len(results)} results')
        except Exception as e:
            check('search returns results', False, str(e))
            return 1

        # --- Check 7: timestamps are timezone-aware ---
        print('7. Timestamp timezone fix')
        try:
            rows, _, _ = await driver.execute_query(
                'MATCH (e:Episodic) RETURN e.created_at LIMIT 1'
            )
            if rows:
                ts = rows[0]['e.created_at']
                has_tz = isinstance(ts, datetime) and ts.tzinfo is not None
                check(
                    'timestamps are tz-aware',
                    has_tz,
                    f'tzinfo={ts.tzinfo}' if isinstance(ts, datetime) else f'type={type(ts)}',
                )
            else:
                check('timestamps are tz-aware', False, 'no rows returned')
                return 1
        except Exception as e:
            check('timestamps are tz-aware', False, str(e))
            return 1

        await driver.close()

    print('\n=== All checks passed ===')
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
