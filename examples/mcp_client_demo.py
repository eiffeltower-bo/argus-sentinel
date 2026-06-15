#!/usr/bin/env python
"""Tiny MCP client to smoke-test the argus MCP server over HTTP.

Start the server first (see context/mcp-server.md), then:

    uv run python examples/mcp_client_demo.py --url http://127.0.0.1:8000/mcp --dir /data

Lists the server's tools, calls `list_clips` on --dir, and `peek_clip` on the first clip found.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def _run(url: str, directory: str, glob: str) -> None:
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("tools:", ", ".join(sorted(t.name for t in tools.tools)))

            lc = await session.call_tool("list_clips", {"directory": directory, "glob": glob})
            clips = lc.structuredContent["clips"]
            print(f"list_clips({directory!r}, {glob!r}) -> {lc.structuredContent['n_clips']} clips")

            if not clips:
                print("no clips found — point --dir at a folder with videos")
                return

            first = clips[0]["path"]
            pc = await session.call_tool("peek_clip", {"path": first})
            print(f"peek_clip({first!r}) ->")
            print(json.dumps(pc.structuredContent, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8000/mcp", help="MCP server URL")
    ap.add_argument("--dir", default="/data", help="server-side folder to list/peek")
    ap.add_argument("--glob", default="**/*.mp4", help="glob for clips under --dir")
    args = ap.parse_args()
    asyncio.run(_run(args.url, args.dir, args.glob))


if __name__ == "__main__":
    main()
