"""The MCP proxy, driven by a generic client against a real upstream server (Phase Gate 4).

Everything else about the proxy is tested through :class:`ProxyState` without a transport --
the rules carry the correctness claims and the SDK does not. That split is defensible right up
until it is the only thing tested, which is what happened: the proxy could not register a
single tool against a real server, and no rules test could see it.

So this drives the whole chain. A generic ``mcp`` client talks to ``specunode mcp-proxy``, which
talks to a real stdio MCP server, and the assertions are about what that server *received* --
read from a log it writes itself -- rather than about what the proxy reports having forwarded.

Marked slow: three processes and two stdio handshakes per test.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="the proxy needs the optional [mcp] extra")

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "tests" / "integration" / "_mcp_upstream.py"

#: The proxy spawns the upstream itself, so both must be startable by the same interpreter the
#: tests run under -- the one with ``mcp`` installed.
PYTHON = sys.executable

pytestmark = pytest.mark.slow


def write_config(directory: Path) -> Path:
    """Declare the effect classes out of band, which is the only way the proxy learns them."""
    path = directory / "specunode.yaml"
    path.write_text(
        "schema_version: 1\n"
        "tools:\n"
        "  get_ticket:\n"
        "    effect: read\n"
        "  close_ticket:\n"
        "    effect: write\n",
        encoding="utf-8",
    )
    return path


def upstream_calls(log: Path) -> list[str]:
    if not log.exists():
        return []
    return [
        str(json.loads(line)["tool"])
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@asynccontextmanager
async def proxy_client(directory: Path, *extra: str) -> AsyncIterator[ClientSession]:
    log = directory / "upstream.log"
    params = StdioServerParameters(
        command=PYTHON,
        args=[
            "-m",
            "specunode.cli",
            "mcp-proxy",
            "--upstream",
            f"{PYTHON} {UPSTREAM} {log}",
            "--config",
            str(write_config(directory)),
            *extra,
        ],
        env=dict(os.environ),
        cwd=str(REPO),
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await asyncio.wait_for(session.initialize(), 60)
        yield session


def text_of(result: object) -> str:
    return "".join(str(block.model_dump().get("text", "")) for block in result.content)  # type: ignore[attr-defined]


def tool_names(listing: object) -> Sequence[str]:
    return sorted(tool.name for tool in listing.tools)  # type: ignore[attr-defined]


async def test_the_upstream_tools_are_listed_through_the_proxy(tmp_path: Path) -> None:
    async with proxy_client(tmp_path) as session:
        listing = await asyncio.wait_for(session.list_tools(), 60)
    names = tool_names(listing)
    assert "get_ticket" in names and "close_ticket" in names
    # And the proxy's own control surface, so a client can see and steer what is held.
    assert "specunode.status" in names and "specunode.retire" in names


async def test_the_upstreams_own_schema_is_what_the_client_is_shown(tmp_path: Path) -> None:
    """Not the one the SDK infers from the proxy's forwarding function.

    A proxy advertising a free-form schema where the upstream declares typed arguments pushes
    every argument error from the client's validation out to the server's, which is a worse
    place to find it -- and for a while this advertised one schema and then rejected every call
    made against it.
    """
    async with proxy_client(tmp_path) as session:
        listing = await asyncio.wait_for(session.list_tools(), 60)
    tool = next(t for t in listing.tools if t.name == "get_ticket")
    schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
    assert isinstance(schema, dict), schema
    assert "ticket_id" in schema.get("properties", {}), schema
    assert "kwargs" not in schema.get("properties", {}), (
        "the client was shown the schema inferred from the proxy's own forwarder"
    )


async def test_a_read_reaches_the_upstream_immediately(tmp_path: Path) -> None:
    async with proxy_client(tmp_path) as session:
        result = await asyncio.wait_for(session.call_tool("get_ticket", {"ticket_id": "tkt-1"}), 60)
        assert "cus-1" in text_of(result), text_of(result)
    assert upstream_calls(tmp_path / "upstream.log") == ["get_ticket"]


async def test_a_write_is_held_until_a_decision_arrives(tmp_path: Path) -> None:
    """The claim the proxy exists for, asserted against the upstream rather than the proxy.

    In blocking mode the write does not return until a decision arrives, so the call and the
    decision have to be in flight together -- which is also how a real client drives it.
    """
    log = tmp_path / "upstream.log"
    async with proxy_client(tmp_path) as session:
        held = asyncio.create_task(session.call_tool("close_ticket", {"ticket_id": "tkt-1"}))
        await asyncio.sleep(0.5)

        assert not held.done(), "the write returned before any decision was reported"
        assert upstream_calls(log) == [], "a held write reached the upstream anyway"

        status = await asyncio.wait_for(session.call_tool("specunode.status", {}), 60)
        assert "1" in text_of(status), text_of(status)

        await asyncio.wait_for(
            session.call_tool(
                "specunode.retire",
                {"tool": "close_ticket", "args": {"ticket_id": "tkt-1"}},
            ),
            60,
        )
        await asyncio.wait_for(held, 60)

    assert upstream_calls(log) == ["close_ticket"], "the retired write never reached the world"


async def test_a_write_the_decision_contradicts_is_never_sent(tmp_path: Path) -> None:
    """Hard Rule 3 over the proxy: a different decision discards, it does not dispatch."""
    log = tmp_path / "upstream.log"
    async with proxy_client(tmp_path) as session:
        held = asyncio.create_task(session.call_tool("close_ticket", {"ticket_id": "tkt-1"}))
        await asyncio.sleep(0.5)
        assert not held.done()

        # The model actually asked to close a *different* ticket.
        await asyncio.wait_for(
            session.call_tool(
                "specunode.retire",
                {"tool": "close_ticket", "args": {"ticket_id": "tkt-999"}},
            ),
            60,
        )
        result = await asyncio.wait_for(held, 60)
        assert "discarded" in text_of(result), text_of(result)

    assert upstream_calls(log) == [], "a contradicted write reached the upstream"


# -- the two shapes the first version of this file did not cover ---------------------------------
#
# It drove BLOCKING mode for exactly one turn, and both of the proxy's remaining Hard Rule 3
# breaks lived outside that box: one in the other mode, one in the second turn.


async def test_a_confirmed_write_reaches_the_upstream_in_handles_mode(tmp_path: Path) -> None:
    """``--handles`` is a shipped flag, and a write confirmed under it was never sent.

    The forwarder returned the placeholder and its coroutine ended, so nothing was left to do
    the sending when a decision finally arrived. ``retire`` recorded the call as dispatched and
    ``specunode.status`` and ``specunode.ledger`` both reported it that way, while the upstream
    had never heard of it -- three surfaces agreeing on something that had not happened.
    """
    log = tmp_path / "upstream.log"
    async with proxy_client(tmp_path, "--handles") as session:
        held = await asyncio.wait_for(session.call_tool("close_ticket", {"ticket_id": "tkt-1"}), 60)
        # The client is handed a placeholder, not a result, and nothing has been sent yet.
        assert "$specunode.handle:" in text_of(held), text_of(held)
        assert upstream_calls(log) == []

        result = await asyncio.wait_for(
            session.call_tool(
                "specunode.retire", {"tool": "close_ticket", "args": {"ticket_id": "tkt-1"}}
            ),
            60,
        )
        assert "mcp-" in text_of(result), text_of(result)

    assert upstream_calls(log) == ["close_ticket"], (
        "the write was reported as dispatched and never reached the upstream"
    )


async def test_a_write_discarded_in_a_later_turn_is_not_forwarded(tmp_path: Path) -> None:
    """Hard Rule 3 across turns, which is where call identity used to collapse.

    Effect ids were numbered from ``len(staged)``, and ``retire`` empties that list -- so the
    first write of turn 2 got the id of the first write of turn 1. ``StagedCall`` is frozen, so
    the two compared equal, and the dispatch guard (a membership test against the dispatched
    list, by value) answered yes for a call this turn's decision had just discarded.
    """
    log = tmp_path / "upstream.log"
    async with proxy_client(tmp_path) as session:
        # Turn 1: the model confirms it, so it is sent.
        first = asyncio.create_task(session.call_tool("close_ticket", {"ticket_id": "tkt-1"}))
        await asyncio.sleep(0.5)
        await asyncio.wait_for(
            session.call_tool(
                "specunode.retire", {"tool": "close_ticket", "args": {"ticket_id": "tkt-1"}}
            ),
            60,
        )
        await asyncio.wait_for(first, 60)
        assert upstream_calls(log) == ["close_ticket"]

        # Turn 2: the same call, structurally identical -- and this time the model asks for
        # something else.
        second = asyncio.create_task(session.call_tool("close_ticket", {"ticket_id": "tkt-1"}))
        await asyncio.sleep(0.5)
        await asyncio.wait_for(
            session.call_tool(
                "specunode.retire", {"tool": "close_ticket", "args": {"ticket_id": "tkt-999"}}
            ),
            60,
        )
        result = await asyncio.wait_for(second, 60)
        assert "discarded" in text_of(result), text_of(result)

    assert upstream_calls(log) == ["close_ticket"], (
        "the contradicted second write was forwarded because it looked like the first"
    )


# -- the three defects that composed into a double-write -----------------------------------------


async def test_a_confirmed_write_returns_its_real_result_to_the_blocking_caller(
    tmp_path: Path,
) -> None:
    """The write happened; the client has to be able to learn that.

    ``retire()`` set the decision event synchronously and the tool then *awaited* the upstream
    before recording the result -- so the blocked caller resumed first, saw its call in
    ``dispatched``, and read a result that was still ``None``. In BLOCKING mode, which is the
    default and the only mode a generic client gets, every write the model confirmed returned
    nothing to the caller that issued it. The natural response to that is to reissue the write.
    """
    log = tmp_path / "upstream.log"
    async with proxy_client(tmp_path) as session:
        held = asyncio.create_task(session.call_tool("close_ticket", {"ticket_id": "tkt-1"}))
        await asyncio.sleep(0.5)
        await asyncio.wait_for(
            session.call_tool(
                "specunode.retire", {"tool": "close_ticket", "args": {"ticket_id": "tkt-1"}}
            ),
            60,
        )
        result = await asyncio.wait_for(held, 60)

    body = text_of(result)
    assert "closed" in body, f"the caller got no result back: {body!r}"
    assert upstream_calls(log) == ["close_ticket"]


async def test_one_decision_authorises_one_write(tmp_path: Path) -> None:
    """Two identical held writes, one decision: exactly one goes out.

    Every structurally identical staged call used to match, and the tool forwarded all of them.
    The proxy computes no idempotency key and keeps no dedupe table, so nothing downstream could
    absorb the repeat.
    """
    log = tmp_path / "upstream.log"
    async with proxy_client(tmp_path, "--handles") as session:
        # Handles mode so both can be staged without either blocking.
        await asyncio.wait_for(session.call_tool("close_ticket", {"ticket_id": "tkt-1"}), 60)
        await asyncio.wait_for(session.call_tool("close_ticket", {"ticket_id": "tkt-1"}), 60)
        assert upstream_calls(log) == []

        await asyncio.wait_for(
            session.call_tool(
                "specunode.retire", {"tool": "close_ticket", "args": {"ticket_id": "tkt-1"}}
            ),
            60,
        )

    assert upstream_calls(log) == ["close_ticket"], (
        "one model decision sent the same write more than once"
    )


async def test_a_timed_out_write_is_not_reported_as_discarded(tmp_path: Path) -> None:
    """A timeout and a contradiction are different facts, and were reported identically.

    The call stays in ``state.staged`` after a timeout -- ``specunode.status`` still lists it and
    a later matching decision forwards it for real -- so telling the client it was *discarded*
    invites exactly the reissue that turns one intended write into two.
    """
    log = tmp_path / "upstream.log"
    async with proxy_client(tmp_path, "--deadline", "1") as session:
        result = await asyncio.wait_for(
            session.call_tool("close_ticket", {"ticket_id": "tkt-1"}), 60
        )
        body = text_of(result)
        assert "timed_out" in body, body
        assert "still_held" in body, body
        assert '"discarded": true' not in body.lower().replace(" ", ""), body
        assert upstream_calls(log) == [], "a timed-out write was sent anyway"

        # And it really is still held, which is the fact the message now states.
        status = await asyncio.wait_for(session.call_tool("specunode.status", {}), 60)
        assert "1" in text_of(status), text_of(status)


# -- classification from the upstream's own annotations -----------------------------------------


async def test_a_tool_annotated_read_only_upstream_is_forwarded_without_an_override(
    tmp_path: Path,
) -> None:
    """The spec says "declare the effect class of each tool (or rely on MCP tool annotations)".

    The second half was false for the whole build. ``ToolRegistry.from_mcp_tools`` existed for
    exactly this and was called only by itself; the proxy passed the upstream's annotations
    *through* to the client and never consulted them, so a tool marked ``readOnlyHint: true``
    with no config override was synthesised as an unknown WRITE and held until a decision that
    nobody would ever make for a read. The default proxy was unusable without a full override
    table, and this test's config deliberately does not list ``peek_ticket``.

    Asserted at the upstream's own log and with a short deadline: if the annotation is ignored,
    the call blocks until the deadline and the log stays empty.
    """
    log = tmp_path / "upstream.log"
    async with proxy_client(tmp_path, "--deadline", "2") as session:
        listing = await asyncio.wait_for(session.list_tools(), 60)
        assert "peek_ticket" in tool_names(listing)
        result = await asyncio.wait_for(
            session.call_tool("peek_ticket", {"ticket_id": "tkt-1"}), 60
        )
    body = text_of(result)
    assert "peeked" in body, f"the annotated read was held instead of forwarded: {body!r}"
    assert "timed_out" not in body, "the read waited out the decision deadline"
    assert upstream_calls(log) == ["peek_ticket"], (
        "the upstream never received the read; the annotation was ignored and it was held"
    )
