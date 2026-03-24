"""
MCPClient — thin adapter between the recommendation agent and analytics data.

The client presents a single stable async interface regardless of whether the
underlying data comes from mock fixtures (``use_mock=True``, default) or a
real MCP analytics server (``use_mock=False``).

Switching from mock to real is a one-line change at the call site::

    # Development / CI — no external dependencies
    client = MCPClient(use_mock=True)

    # Production — requires a running analytics MCP server
    client = MCPClient(use_mock=False)

Mock data design
----------------
All mock methods are *deterministic*: the same ``user_id`` or ``asset_id``
always produces the same output.  This is achieved via ``hash()`` (stable
within a process) rather than ``random``, so tests are reproducible without
seeding.

The mock is intentionally minimal — its job is to return valid
``ContentMetadata`` objects so downstream code can run end-to-end in CI.
The specific values are unimportant as long as they round-trip through the
``Category``, ``Condition``, and ``PriceRange`` enum validators.
"""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from src.schemas.metadata import Category, Condition, ContentMetadata, PriceRange


logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Mock data pools (deterministic, domain-valid)
# ---------------------------------------------------------------------------

_MOCK_TAGS: List[str] = [
    "modern", "minimal", "corporate", "creative", "elegant",
    "portable", "wireless", "premium", "budget", "popular",
]

_MOCK_CATEGORIES: List[str] = [c.value for c in Category]
_MOCK_CONDITIONS: List[str] = [c.value for c in Condition]
_MOCK_PRICE_RANGES: List[str] = [p.value for p in PriceRange]


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------


class MCPClient:
    """Fetch buyer history and asset metadata from a real or mock data source.

    Parameters
    ----------
    use_mock:
        When ``True`` (default), all methods return deterministic in-process
        data — no network connections are made.  Set to ``False`` only when
        a real analytics MCP server is available; the stub ``_call_tool``
        method will raise ``NotImplementedError`` until a real implementation
        is wired in.
    """

    def __init__(self, use_mock: bool = True) -> None:
        self.use_mock = use_mock

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def get_browsing_history(
        self,
        user_id: str,
        limit: int = 50,
    ) -> List[str]:
        """Return a list of recently-browsed asset IDs for ``user_id``.

        The list is ordered most-recent-first.  The ``limit`` parameter is
        honoured by both the mock (which caps its output) and the real
        implementation (which passes it to the server).
        """
        if self.use_mock:
            ids = self._mock_browsing_history(user_id)
            return ids[:limit]

        result: List[str] = await self._call_tool(
            "get_user_browsing_history",
            {"user_id": user_id, "limit": limit},
        )
        logger.info(
            "mcp_browsing_history_fetched",
            user_id=user_id,
            count=len(result),
        )
        return result

    async def get_purchase_history(
        self,
        user_id: str,
        limit: int = 20,
    ) -> List[str]:
        """Return a list of purchased asset IDs for ``user_id``.

        The list is ordered most-recent-first.
        """
        if self.use_mock:
            ids = self._mock_purchase_history(user_id)
            return ids[:limit]

        result: List[str] = await self._call_tool(
            "get_user_purchase_history",
            {"user_id": user_id, "limit": limit},
        )
        logger.info(
            "mcp_purchase_history_fetched",
            user_id=user_id,
            count=len(result),
        )
        return result

    async def get_asset_metadata(
        self,
        asset_ids: List[str],
    ) -> List[ContentMetadata]:
        """Return ``ContentMetadata`` objects for the given asset IDs.

        IDs that cannot be resolved are silently omitted, consistent with the
        tool schema contract.
        """
        if self.use_mock:
            return self._mock_asset_metadata(asset_ids)

        result: List[ContentMetadata] = await self._call_tool(
            "get_asset_metadata",
            {"asset_ids": asset_ids},
        )
        logger.info(
            "mcp_asset_metadata_fetched",
            requested=len(asset_ids),
            resolved=len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Real MCP server stub
    # ------------------------------------------------------------------

    async def _call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Dispatch a tool call to the real MCP server.

        Not implemented — override or replace this method when connecting to
        a live analytics MCP server.  Callers should only reach this path
        when ``use_mock=False``.
        """
        raise NotImplementedError(
            f"Real MCP server not configured. "
            f"Tool '{tool_name}' cannot be called with use_mock=False. "
            f"Either set use_mock=True or wire a real MCP server connection "
            f"into _call_tool()."
        )

    # ------------------------------------------------------------------
    # Mock data helpers
    # ------------------------------------------------------------------

    def _mock_browsing_history(self, user_id: str) -> List[str]:
        """Return 20 deterministic asset IDs derived from ``user_id``."""
        seed = abs(hash(user_id)) % 100
        return [f"asset_{seed + i}" for i in range(20)]

    def _mock_purchase_history(self, user_id: str) -> List[str]:
        """Return 5 deterministic asset IDs derived from ``user_id``.

        The purchase IDs are a subset of the browsing IDs (first 5) so that
        tag-affinity computation and top_category inference produce coherent,
        consistent profiles.
        """
        seed = abs(hash(user_id)) % 100
        return [f"asset_{seed + i}" for i in range(5)]

    def _mock_asset_metadata(
        self,
        asset_ids: List[str],
    ) -> List[ContentMetadata]:
        """Generate a valid ``ContentMetadata`` record for each asset ID.

        All field values are derived deterministically from the asset ID so
        that the same ID always produces the same mock record.  This keeps
        tests reproducible without mocking the hash function.
        """
        results: List[ContentMetadata] = []
        for asset_id in asset_ids:
            h = abs(hash(asset_id))

            category = _MOCK_CATEGORIES[h % len(_MOCK_CATEGORIES)]
            condition = _MOCK_CONDITIONS[(h + 1) % len(_MOCK_CONDITIONS)]
            price_range = _MOCK_PRICE_RANGES[(h + 2) % len(_MOCK_PRICE_RANGES)]

            # Pick two distinct tags; use (h+7) offset to avoid duplicates.
            tag_a = _MOCK_TAGS[h % len(_MOCK_TAGS)]
            tag_b = _MOCK_TAGS[(h + 7) % len(_MOCK_TAGS)]
            # Deduplicate while preserving order.
            tags = list(dict.fromkeys([tag_a, tag_b]))

            results.append(
                ContentMetadata(
                    content_id=asset_id,
                    title=f"Product {asset_id}",
                    category=category,
                    condition=condition,
                    price_range=price_range,
                    tags=tags,
                )
            )
        return results
