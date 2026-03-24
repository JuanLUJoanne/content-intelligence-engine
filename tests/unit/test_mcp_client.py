"""Tests for MCPClient and build_buyer_profile integration."""

from __future__ import annotations

import pytest

from src.agents.memory.buyer_profile import build_buyer_profile
from src.mcp.client import MCPClient
from src.schemas.metadata import ContentMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_corpus(n: int = 10) -> list[ContentMetadata]:
    """Return n minimal ContentMetadata objects for use as available_assets."""
    client = MCPClient(use_mock=True)
    asset_ids = [f"corpus_asset_{i}" for i in range(n)]
    return client._mock_asset_metadata(asset_ids)


# ---------------------------------------------------------------------------
# 1. use_mock=True — determinism
# ---------------------------------------------------------------------------


class TestMockDeterminism:
    def test_same_user_browsing_history_is_reproducible(self):
        client = MCPClient(use_mock=True)
        first = client._mock_browsing_history("user-abc")
        second = client._mock_browsing_history("user-abc")
        assert first == second

    def test_same_user_purchase_history_is_reproducible(self):
        client = MCPClient(use_mock=True)
        first = client._mock_purchase_history("user-abc")
        second = client._mock_purchase_history("user-abc")
        assert first == second

    def test_different_users_get_different_browsing_histories(self):
        client = MCPClient(use_mock=True)
        h1 = client._mock_browsing_history("alice")
        h2 = client._mock_browsing_history("bob")
        # It is *possible* (but unlikely) for two users to collide; this test
        # uses two names that are known to produce different seeds.
        assert h1 != h2

    def test_same_asset_metadata_is_reproducible(self):
        client = MCPClient(use_mock=True)
        ids = ["asset_42", "asset_99"]
        first = client._mock_asset_metadata(ids)
        second = client._mock_asset_metadata(ids)
        assert [a.content_id for a in first] == [a.content_id for a in second]
        assert [a.category for a in first] == [a.category for a in second]

    def test_independent_instances_agree(self):
        ids = ["asset_7", "asset_13"]
        m1 = MCPClient(use_mock=True)._mock_asset_metadata(ids)
        m2 = MCPClient(use_mock=True)._mock_asset_metadata(ids)
        assert [(a.content_id, a.category, a.tags) for a in m1] == [
            (a.content_id, a.category, a.tags) for a in m2
        ]


# ---------------------------------------------------------------------------
# 2. use_mock=True — shape / validity
# ---------------------------------------------------------------------------


class TestMockShape:
    def test_browsing_history_returns_list_of_strings(self):
        client = MCPClient(use_mock=True)
        result = client._mock_browsing_history("u1")
        assert isinstance(result, list)
        assert all(isinstance(x, str) for x in result)

    def test_purchase_history_is_subset_of_browsing_history(self):
        client = MCPClient(use_mock=True)
        browsing = client._mock_browsing_history("u1")
        purchases = client._mock_purchase_history("u1")
        assert set(purchases).issubset(set(browsing))

    def test_asset_metadata_returns_valid_content_metadata(self):
        client = MCPClient(use_mock=True)
        assets = client._mock_asset_metadata(["asset_1", "asset_2"])
        assert len(assets) == 2
        for asset in assets:
            assert isinstance(asset, ContentMetadata)
            assert asset.content_id in {"asset_1", "asset_2"}
            assert asset.title.startswith("Product ")

    def test_asset_metadata_content_id_matches_input(self):
        client = MCPClient(use_mock=True)
        ids = ["x", "y", "z"]
        assets = client._mock_asset_metadata(ids)
        assert [a.content_id for a in assets] == ids

    def test_browsing_history_limit_is_honoured(self):
        client = MCPClient(use_mock=True)
        result = client._mock_browsing_history("u1")
        assert len(result) == 20  # default mock returns 20

    def test_purchase_history_is_shorter_than_browsing(self):
        client = MCPClient(use_mock=True)
        browsing = client._mock_browsing_history("u1")
        purchases = client._mock_purchase_history("u1")
        assert len(purchases) < len(browsing)


# ---------------------------------------------------------------------------
# 3. Async interface
# ---------------------------------------------------------------------------


class TestAsyncInterface:
    @pytest.mark.asyncio
    async def test_get_browsing_history_returns_list(self):
        client = MCPClient(use_mock=True)
        result = await client.get_browsing_history("async-user")
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_get_browsing_history_limit_truncates(self):
        client = MCPClient(use_mock=True)
        result = await client.get_browsing_history("async-user", limit=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_get_purchase_history_returns_list(self):
        client = MCPClient(use_mock=True)
        result = await client.get_purchase_history("async-user")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_asset_metadata_returns_content_metadata(self):
        client = MCPClient(use_mock=True)
        assets = await client.get_asset_metadata(["asset_5", "asset_6"])
        assert len(assets) == 2
        assert all(isinstance(a, ContentMetadata) for a in assets)

    @pytest.mark.asyncio
    async def test_get_asset_metadata_empty_input(self):
        client = MCPClient(use_mock=True)
        assets = await client.get_asset_metadata([])
        assert assets == []


# ---------------------------------------------------------------------------
# 4. build_buyer_profile integration
# ---------------------------------------------------------------------------


class TestBuildBuyerProfile:
    @pytest.mark.asyncio
    async def test_returns_buyer_profile_with_correct_user_id(self):
        client = MCPClient(use_mock=True)
        profile = await build_buyer_profile("profile-user", client)
        assert profile["user_id"] == "profile-user"

    @pytest.mark.asyncio
    async def test_profile_has_tag_affinity(self):
        client = MCPClient(use_mock=True)
        profile = await build_buyer_profile("tag-user", client)
        assert isinstance(profile["tag_affinity"], dict)
        # Browsing history is non-empty so affinity should be populated.
        assert len(profile["tag_affinity"]) > 0

    @pytest.mark.asyncio
    async def test_profile_affinity_scores_are_normalised(self):
        client = MCPClient(use_mock=True)
        profile = await build_buyer_profile("norm-user", client)
        scores = list(profile["tag_affinity"].values())
        assert max(scores) == pytest.approx(1.0)
        assert all(0.0 <= s <= 1.0 for s in scores)

    @pytest.mark.asyncio
    async def test_profile_has_top_category(self):
        client = MCPClient(use_mock=True)
        profile = await build_buyer_profile("cat-user", client)
        assert isinstance(profile["top_category"], str)
        assert len(profile["top_category"]) > 0

    @pytest.mark.asyncio
    async def test_profile_purchase_history_is_list_of_strings(self):
        client = MCPClient(use_mock=True)
        profile = await build_buyer_profile("hist-user", client)
        assert isinstance(profile["purchase_history"], list)
        assert all(isinstance(x, str) for x in profile["purchase_history"])

    @pytest.mark.asyncio
    async def test_same_user_produces_same_profile(self):
        client = MCPClient(use_mock=True)
        p1 = await build_buyer_profile("stable-user", client)
        p2 = await build_buyer_profile("stable-user", client)
        assert p1["top_category"] == p2["top_category"]
        assert p1["tag_affinity"] == p2["tag_affinity"]
        assert p1["purchase_history"] == p2["purchase_history"]


# ---------------------------------------------------------------------------
# 5. RecommendationAgent integration with MCPClient
# ---------------------------------------------------------------------------


class TestRecommendationAgentWithMCPClient:
    @pytest.mark.asyncio
    async def test_run_with_mcp_client_returns_result(self):
        from src.agents.recommendation_agent import RecommendationAgent

        agent = RecommendationAgent(model_id="dummy")
        corpus = _make_corpus(10)
        client = MCPClient(use_mock=True)

        result = await agent.run("mcp-user", corpus, mcp_client=client)
        assert isinstance(result.email, str)
        assert len(result.email) > 0

    @pytest.mark.asyncio
    async def test_run_with_mcp_client_returns_assets(self):
        from src.agents.recommendation_agent import RecommendationAgent

        agent = RecommendationAgent(model_id="dummy")
        corpus = _make_corpus(10)
        client = MCPClient(use_mock=True)

        result = await agent.run("mcp-user", corpus, mcp_client=client)
        assert isinstance(result.assets, list)

    @pytest.mark.asyncio
    async def test_run_without_mcp_client_still_works(self):
        """Passing no mcp_client falls back to the direct assets path."""
        from src.agents.recommendation_agent import RecommendationAgent

        agent = RecommendationAgent(model_id="dummy")
        corpus = _make_corpus(10)

        result = await agent.run("no-mcp-user", corpus)
        assert isinstance(result.email, str)

    @pytest.mark.asyncio
    async def test_mcp_client_false_raises_not_implemented(self):
        """use_mock=False raises NotImplementedError when _call_tool is invoked."""
        client = MCPClient(use_mock=False)
        with pytest.raises(NotImplementedError):
            await client.get_browsing_history("any-user")
