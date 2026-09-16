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


if __name__ == "__main__":
    server.run("stdio")
