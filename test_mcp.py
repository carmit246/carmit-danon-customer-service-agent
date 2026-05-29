#!/usr/bin/env python3
"""Quick smoke-test for the MCP server — spawns the server via stdio and calls every tool."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastmcp import Client


async def main():
    server_path = Path(__file__).parent / "mcp_server.py"

    async with Client(server_path) as client:

        tools = await client.list_tools()
        print(f"\n{'='*60}")
        print(f"Tools available ({len(tools)}):")
        for t in tools:
            print(f"  • {t.name}")

        print(f"\n{'='*60}")
        print("TEST 1 — get_categories()")
        result = await client.call_tool("get_categories", {})
        data = result.data
        print(f"  count  : {data['count']}")
        print(f"  sample : {data['categories'][:5]}")

        first_cat = data["categories"][0]

        print(f"\n{'='*60}")
        print(f"TEST 2 — get_intents(category='{first_cat}')")
        result = await client.call_tool("get_intents", {"category": first_cat})
        data = result.data
        print(f"  intents: {data.get('intents', data.get('error'))}")

        first_intent = data.get("intents", [None])[0]

        print(f"\n{'='*60}")
        print(f"TEST 3 — count_records(category='{first_cat}')")
        result = await client.call_tool("count_records", {"category": first_cat})
        data = result.data
        print(f"  count  : {data['count']}")

        print(f"\n{'='*60}")
        print(f"TEST 4 — show_examples(n=2, category='{first_cat}')")
        result = await client.call_tool("show_examples", {"n": 2, "category": first_cat})
        data = result.data
        print(f"  returned / total: {data['returned']} / {data['total_matching']}")
        for ex in data["examples"]:
            print(f"    [{ex['intent']}] {ex['instruction'][:80]}…")

        print(f"\n{'='*60}")
        print(f"TEST 5 — get_intent_distribution(category='{first_cat}')")
        result = await client.call_tool("get_intent_distribution", {"category": first_cat})
        data = result.data
        print(f"  total  : {data['total']}")
        for row in data["distribution"][:3]:
            print(f"    {row['intent']:40s} {row['count']:5d}  ({row['percentage']}%)")

        print(f"\n{'='*60}")
        if first_intent:
            print(f"TEST 6 — get_sample_for_summary(category='{first_cat}', intent='{first_intent}', n=3)")
            result = await client.call_tool(
                "get_sample_for_summary",
                {"category": first_cat, "intent": first_intent, "n": 3},
            )
            data = result.data
            print(f"  sample_size / total: {data['sample_size']} / {data['total_in_scope']}")
            for entry in data["sample"]:
                print(f"    • {entry['instruction'][:80]}…")

        print(f"\n{'='*60}")
        print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
