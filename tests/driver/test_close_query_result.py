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

from unittest.mock import MagicMock, patch

import pytest

from graphiti_core.driver.ladybug_driver import LadybugDriver, _close_query_result
from graphiti_core.errors import LadybugConnectionError


class _FakeQR:
    """Minimal stand-in for real_ladybug.QueryResult that records close() calls."""

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class TestCloseQueryResult:
    def test_closes_single_query_result(self) -> None:
        qr = _FakeQR()
        _close_query_result(qr)
        assert qr.close_calls == 1

    def test_closes_every_item_in_list(self) -> None:
        qrs = [_FakeQR(), _FakeQR(), _FakeQR()]
        _close_query_result(qrs)
        assert [qr.close_calls for qr in qrs] == [1, 1, 1]

    def test_noop_on_none(self) -> None:
        # None is returned when Connection.execute is short-circuited
        # (e.g. by the early-return branch in LadybugDriver.execute_query).
        _close_query_result(None)

    def test_noop_on_empty_list(self) -> None:
        _close_query_result([])


class TestExecuteQueryClosesResult:
    """Regression: LadybugDriver.execute_query must close the QueryResult
    on every path — including the empty-results early-return — to prevent
    the late-GC segfault in QueryResult.__del__ (see fix/replay-query-result-leak).
    """

    @pytest.mark.asyncio
    async def test_closes_single_query_result(self) -> None:
        driver = LadybugDriver(db=':memory:')
        try:
            fake_qr = _FakeQR()
            fake_qr.rows_as_dict = MagicMock(return_value=[])  # type: ignore[attr-defined]
            with patch.object(driver.client, 'execute', return_value=fake_qr) as mock_exec:
                await driver.execute_query('MATCH (n) RETURN n')
                mock_exec.assert_awaited_once()
            assert fake_qr.close_calls == 1
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_closes_list_of_query_results(self) -> None:
        driver = LadybugDriver(db=':memory:')
        try:
            fake_qrs = [_FakeQR(), _FakeQR()]
            for qr in fake_qrs:
                qr.rows_as_dict = MagicMock(return_value=[])  # type: ignore[attr-defined]
            with patch.object(driver.client, 'execute', return_value=fake_qrs):
                await driver.execute_query('RETURN 1; RETURN 2;')
            assert [qr.close_calls for qr in fake_qrs] == [1, 1]
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_closes_on_empty_list_early_return(self) -> None:
        """An empty list still needs to be recognised as 'nothing to process'
        while not skipping the close step — the finally block handles it."""
        driver = LadybugDriver(db=':memory:')
        try:
            # Empty list is falsy → hits `if not results: return [], None, None`,
            # but must still flow through finally. _close_query_result([]) is a no-op,
            # so there's nothing to assert on fakes here; the test exists to lock in
            # that the early return stays inside try/finally and doesn't regress.
            with patch.object(driver.client, 'execute', return_value=[]):
                rows, _, _ = await driver.execute_query('RETURN 1')
            assert rows == []
        finally:
            await driver.close()


class TestNativeExceptionHandling:
    """Tests for A2/A3/A4: native C++ exception interception and connection health probe.

    The actual graph-size trigger (~40K+ entities) cannot be reproduced in CI without
    the operator's WAL files. These tests exercise the exception-handling code path by
    injecting RuntimeError at the client.execute call site.
    """

    @pytest.mark.asyncio
    async def test_cpp_exception_reraises_when_probe_succeeds(self) -> None:
        driver = LadybugDriver(db=':memory:')
        try:
            cpp_exc = RuntimeError('unordered_map::at: key not found')
            probe_result = MagicMock()
            with patch.object(
                driver.client, 'execute', side_effect=[cpp_exc, probe_result]
            ):
                with pytest.raises(RuntimeError, match='unordered_map::at: key not found'):
                    await driver.execute_query('MATCH (n) RETURN n')
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_cpp_exception_raises_connection_error_when_probe_fails(self) -> None:
        driver = LadybugDriver(db=':memory:')
        try:
            cpp_exc = RuntimeError('unordered_map::at: key not found')
            probe_exc = RuntimeError('connection is dead')
            with patch.object(
                driver.client, 'execute', side_effect=[cpp_exc, probe_exc]
            ):
                with pytest.raises(LadybugConnectionError):
                    await driver.execute_query('MATCH (n) RETURN n')
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_generic_exception_also_probes_connection(self) -> None:
        driver = LadybugDriver(db=':memory:')
        try:
            generic_exc = RuntimeError('some internal error')
            probe_result = MagicMock()
            with patch.object(
                driver.client, 'execute', side_effect=[generic_exc, probe_result]
            ):
                with pytest.raises(RuntimeError, match='some internal error'):
                    await driver.execute_query('MATCH (n) RETURN n')
        finally:
            await driver.close()
