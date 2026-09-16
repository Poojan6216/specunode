"""SpecuNode — speculative out-of-order execution for agent graphs, with a store buffer.

Reads issue early; writes wait in a branch-scoped store buffer; the target model's actual
output is the branch-resolution signal. Nothing reaches the world from an unretired branch.

The public API is exposed lazily so that ``import specunode`` stays cheap and does not pull
in optional integrations (LangGraph, MCP, Anthropic).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
