"""Tests for TrendingTool — deterministic trending items fallback."""

from __future__ import annotations

import pytest

from src.mcp.trending import TrendingTool
from src.schemas.metadata import ContentMetadata


@pytest.fixture
def tool() -> TrendingTool:
    return TrendingTool()


class TestTrendingTool:
    def test_returns_content_metadata(self, tool: TrendingTool):
        results = tool.get_trending(top_k=3)
        assert all(isinstance(r, ContentMetadata) for r in results)

    def test_respects_top_k(self, tool: TrendingTool):
        results = tool.get_trending(top_k=2)
        assert len(results) == 2

    def test_category_filter(self, tool: TrendingTool):
        results = tool.get_trending(category="electronics", top_k=10)
        assert len(results) > 0
        assert len(results) <= 4  # pool has 4 electronics items

    def test_unknown_category_returns_all(self, tool: TrendingTool):
        results = tool.get_trending(category="nonexistent", top_k=5)
        assert len(results) == 5

    def test_no_category_returns_mixed(self, tool: TrendingTool):
        results = tool.get_trending(top_k=10)
        categories = {r.category.value for r in results}
        assert len(categories) > 1

    def test_deterministic(self, tool: TrendingTool):
        a = tool.get_trending(category="home", top_k=3)
        b = tool.get_trending(category="home", top_k=3)
        assert [r.content_id for r in a] == [r.content_id for r in b]

    def test_content_ids_are_unique(self, tool: TrendingTool):
        results = tool.get_trending(top_k=10)
        ids = [r.content_id for r in results]
        assert len(ids) == len(set(ids))

    def test_items_have_tags(self, tool: TrendingTool):
        results = tool.get_trending(top_k=5)
        assert all(len(r.tags) > 0 for r in results)
