"""A minimal upstream MCP server, for driving the proxy end to end.

Deliberately not a mock. Phase Gate 4 asks whether the proxy works against a *generic* client
and a real server, and a stand-in for either one answers a different question. This is a real
stdio MCP server with two tools -- one that reads and one that writes -- and it records every
call it receives to a file so the test can assert on what actually reached it rather than on
what the proxy says it forwarded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("upstream")


def _log_path() -> Path | None:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else None


def _record(tool: str, args: dict[str, object]) -> None:
    path = _log_path()
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"tool": tool, "args": args}, sort_keys=True) + "\n")


@server.tool()
def get_ticket(ticket_id: str) -> dict[str, object]:
    """Read a ticket. No side effects."""
    _record("get_ticket", {"ticket_id": ticket_id})
    return {"ticket_id": ticket_id, "status": "open", "customer_id": "cus-1"}


@server.tool()
def close_ticket(ticket_id: str) -> dict[str, object]:
    """Close a ticket. This one changes something."""
    _record("close_ticket", {"ticket_id": ticket_id})
    return {"ticket_id": ticket_id, "status": "closed"}


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def peek_ticket(ticket_id: str) -> dict[str, object]:
    """A read that declares itself one through MCP annotations, and nowhere else.

    The proxy's test config deliberately does NOT list this tool. If the proxy classifies
    from the upstream's ``readOnlyHint`` it forwards this immediately; if it ignores the
    annotation -- as it did for the whole build, while the spec said "or rely on MCP tool
    annotations" -- it synthesises an unknown WRITE and holds the call until a decision
    arrives, which for a read that nobody will ever "decide" means the client waits out the
    deadline.
    """
    _record("peek_ticket", {"ticket_id": ticket_id})
    return {"ticket_id": ticket_id, "status": "open", "peeked": True}


if __name__ == "__main__":
    server.run("stdio")
