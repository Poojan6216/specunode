"""The front door works as the README says it does, and keeps working.

``examples/quickstart.py`` is the first thing a new user runs, so it runs here too: a card is
charged, the process dies after the charge went through and before its reply came back, and the
resume finishes the run with one charge and one receipt.
"""

from __future__ import annotations

import contextlib
import importlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


async def test_the_quickstart_charges_once_through_a_crash() -> None:
    sys.path.insert(0, str(REPO / "examples"))
    try:
        quickstart = importlib.import_module("quickstart")
        outcome = await quickstart.main()
    finally:
        sys.path.remove(str(REPO / "examples"))
    assert outcome == {"charges": 1, "receipts": 1, "ok": 1}


def test_importing_the_package_loads_nothing_it_does_not_need() -> None:
    """``import specunode`` must stay cheap: the runtime and the integrations load on use."""
    probe = (
        "import sys, specunode; "
        "heavy = [m for m in sys.modules if m.startswith(('specunode.core', 'specunode.journal', "
        "'langgraph', 'anthropic', 'mcp'))]; "
        "print(heavy)"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert loaded == "[]", loaded


def test_every_public_name_resolves() -> None:
    import specunode

    assert set(specunode.__all__) - {"__version__"} == set(specunode._EXPORTS)
    for name in specunode._EXPORTS:
        assert getattr(specunode, name) is not None, name


async def test_a_resumed_run_still_has_the_inputs_it_started_with(tmp_path: Path) -> None:
    """Recovery rebuilt state from the nodes' deltas alone, so a resumed node that read an input
    found it gone -- and a delta that changed an input key had nothing to apply to."""
    import specunode
    from specunode.journal.journal import Journal
    from specunode.journal.replay import recover

    class Died(BaseException):
        pass

    plug = {"armed": True}

    @specunode.tool(effect="write", idempotent=True)
    async def note(text: str) -> dict[str, str]:
        if plug["armed"]:
            plug["armed"] = False
            raise Died("killed mid-write")
        return {"noted": text}

    @specunode.node()
    async def first(session: specunode.RunSession) -> specunode.Decision:
        session.state["customer_id"] = f"{session.state['customer_id']}-checked"
        return specunode.FreeText.of("first")

    @specunode.node()
    async def second(session: specunode.RunSession) -> specunode.Decision:
        await session.call_tool("note", {"text": str(session.state["customer_id"])})
        session.state["done"] = True
        return specunode.FreeText.of("second")

    def route(state: dict[str, object]) -> str | None:
        if not str(state.get("customer_id", "")).endswith("-checked"):
            return "first"
        return None if state.get("done") else "second"

    runtime = specunode.Runtime(
        specunode.graph([first, second], route), tools=[note], journal=tmp_path / "j.db"
    )
    with contextlib.suppress(Died):
        await runtime.run({"customer_id": "cus-1", "plan": "pro"}, run_id="r1")
    assert recover(Journal(tmp_path / "j.db"), "r1").state == {
        "customer_id": "cus-1-checked",
        "plan": "pro",
    }
    resumed = await runtime.resume("r1")
    assert resumed.ok, resumed.error
    assert resumed.state == {"customer_id": "cus-1-checked", "plan": "pro", "done": True}
