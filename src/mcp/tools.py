"""
MCP tool schema definitions for the analytics data layer.

These schemas follow the Model Context Protocol tool-definition format.
Each entry describes one callable tool that an MCP server exposes: its name,
a human-readable description the model uses to decide when to invoke it, and
a JSON Schema for its input parameters.

The schemas are kept here rather than inline in the client so that:
- They can be inspected and tested independently of any live server.
- A future Anthropic SDK integration can pass them directly to
  ``client.messages.create(tools=ANALYTICS_TOOLS)`` without change.
- Documentation generation can read them without instantiating a client.

Usage
-----
Import ``ANALYTICS_TOOLS`` when you need the raw schema list (e.g. for
registering with an MCP server or passing to the Claude API).  For actual
data fetching, use ``MCPClient`` in ``src/mcp/client.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Tool schema definitions
# ---------------------------------------------------------------------------

ANALYTICS_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "get_user_browsing_history",
        "description": (
            "Fetch the most recently viewed asset IDs for a user. "
            "Returns an ordered list of asset IDs, most recent first. "
            "Use this to understand a buyer's current browsing interests."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Stable user identifier.",
                },
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "description": "Maximum number of asset IDs to return.",
                },
            },
            "required": ["user_id"],
        },
    },
    {
        "name": "get_user_purchase_history",
        "description": (
            "Fetch asset IDs for all items purchased by a user. "
            "Returns an ordered list of asset IDs, most recent first. "
            "Use this to understand a buyer's proven purchase preferences."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Stable user identifier.",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Maximum number of asset IDs to return.",
                },
            },
            "required": ["user_id"],
        },
    },
    {
        "name": "get_asset_metadata",
        "description": (
            "Fetch ContentMetadata for a list of asset IDs. "
            "Returns one metadata record per requested ID; IDs not found in "
            "the catalogue are silently omitted from the response."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "asset_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of asset IDs to resolve.",
                },
            },
            "required": ["asset_ids"],
        },
    },
]
