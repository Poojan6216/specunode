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
        result = await asyncio.wait_for(
            session.call_tool("get_ticket", {"ticket_id": "tkt-1"}), 60
        )
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
