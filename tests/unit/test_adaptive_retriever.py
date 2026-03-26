"""Tests for AdaptiveRetriever — LLM-in-the-loop search refinement."""

from __future__ import annotations

import pytest

from src.agents.adaptive_retriever import AdaptiveRetriever
from src.agents.memory.buyer_profile import BuyerProfile, make_buyer_profile
from src.gateway.providers import DummyProvider
from src.mcp.client import MCPClient
from src.retrieval.asset_retriever import AssetRetriever
from src.schemas.metadata import ContentMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_corpus(n: int = 20) -> list[ContentMetadata]:
    client = MCPClient(use_mock=True)
    return client._mock_asset_metadata([f"asset_{i}" for i in range(n)])


def _make_profile() -> BuyerProfile:
    corpus = _make_corpus(10)
    return make_buyer_profile("test-user", corpus[:3], corpus[:8])


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


class TestAdaptiveRetrieverBasic:
    @pytest.mark.asyncio
    async def test_returns_results_with_initial_query(self):
        corpus = _make_corpus()
        retriever = AssetRetriever(corpus, top_n=5)
        adaptive = AdaptiveRetriever(
            retriever, DummyProvider(), model_id="dummy", max_rounds=3,
        )
        profile = _make_profile()

        results = await adaptive.search(
            profile, initial_query="electronics", initial_filters={"category": "electronics"},
        )
        assert isinstance(results, list)
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_returns_results_without_initial_query(self):
        corpus = _make_corpus()
        retriever = AssetRetriever(corpus, top_n=5)
        adaptive = AdaptiveRetriever(
            retriever, DummyProvider(), model_id="dummy", max_rounds=3,
        )
        profile = _make_profile()

        results = await adaptive.search(profile)
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_results_are_content_metadata(self):
        corpus = _make_corpus()
        retriever = AssetRetriever(corpus, top_n=5)
        adaptive = AdaptiveRetriever(
            retriever, DummyProvider(), model_id="dummy",
        )
        profile = _make_profile()
        results = await adaptive.search(profile, initial_query="electronics")
        assert all(isinstance(r, ContentMetadata) for r in results)

    @pytest.mark.asyncio
    async def test_no_duplicate_content_ids(self):
        corpus = _make_corpus()
        retriever = AssetRetriever(corpus, top_n=5)
        adaptive = AdaptiveRetriever(
            retriever, DummyProvider(), model_id="dummy", max_rounds=3,
        )
        profile = _make_profile()
        results = await adaptive.search(profile, initial_query="electronics")
        ids = [r.content_id for r in results]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Max rounds respected
# ---------------------------------------------------------------------------


class TestMaxRounds:
    @pytest.mark.asyncio
    async def test_single_round(self):
        corpus = _make_corpus()
        retriever = AssetRetriever(corpus, top_n=5)
        adaptive = AdaptiveRetriever(
            retriever, DummyProvider(), model_id="dummy", max_rounds=1,
        )
        profile = _make_profile()
        results = await adaptive.search(profile, initial_query="electronics")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Integration with RecommendationAgent
# ---------------------------------------------------------------------------


class TestRecommendationAgentIntegration:
    @pytest.mark.asyncio
    async def test_agent_run_with_adaptive_retrieval(self):
        from src.agents.recommendation_agent import RecommendationAgent

        agent = RecommendationAgent(model_id="dummy")
        corpus = _make_corpus(10)

        result = await agent.run("adaptive-user", corpus)
        assert isinstance(result.email, str)
        assert len(result.email) > 0
        assert len(result.assets) > 0

    @pytest.mark.asyncio
    async def test_agent_run_with_mcp_and_adaptive(self):
        from src.agents.recommendation_agent import RecommendationAgent

        agent = RecommendationAgent(model_id="dummy")
        corpus = _make_corpus(10)
        client = MCPClient(use_mock=True)

        result = await agent.run("mcp-adaptive-user", corpus, mcp_client=client)
        assert isinstance(result.email, str)
        assert len(result.assets) > 0

    @pytest.mark.asyncio
    async def test_agent_deterministic_with_same_user(self):
        from src.agents.recommendation_agent import RecommendationAgent

        agent = RecommendationAgent(model_id="dummy")
        corpus = _make_corpus(10)

        r1 = await agent.run("det-user", corpus)
        r2 = await agent.run("det-user", corpus)
        assert r1.variant == r2.variant
        assert [a.content_id for a in r1.assets] == [a.content_id for a in r2.assets]
