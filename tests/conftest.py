"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from specunode.journal.journal import close_all_writers


@pytest.fixture(autouse=True)
def _close_journal_writers() -> Iterator[None]:
    """Journal writers are process-global by database path; do not leak them between tests."""
    yield
    close_all_writers()
