"""argus MCP server — exposes the triage/track facade as MCP tools over HTTP.

Thin, model-agnostic consumer of the public ``argus`` facade (the MCP analogue of the CLI).
Run with the ``argus-mcp`` console script or ``python -m argus.mcp``.
"""

from .server import main, mcp

__all__ = ["main", "mcp"]
