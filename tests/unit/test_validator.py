"""Tests for item validation and MMR diversity reranking."""

from __future__ import annotations

import pytest

from src.retrieval.validator import mmr_rerank, validate_item, _attribute_similarity
from src.schemas.metadata import Category, Condition, ContentMetadata, PriceRange


def _make_item(
    content_id: str = "x",
    category: str = "electronics",
    tags: list[str] | None = None,
    title: str = "Item",
) -> ContentMetadata:
    return ContentMetadata(
        content_id=content_id,
        title=title,
        category=Category(category),
        condition=Condition.NEW,
        price_range=PriceRange.MID_RANGE,
        tags=tags or ["tag_a"],
    )


# ---------------------------------------------------------------------------
# validate_item
# ---------------------------------------------------------------------------


class TestValidateItem:
    def test_valid_item_passes(self):
        assert validate_item(_make_item()) is True

    def test_excluded_id_rejected(self):
        item = _make_item(content_id="purchased_1")
        assert validate_item(item, exclude_ids=frozenset({"purchased_1"})) is False

    def test_empty_title_rejected(self):
        item = _make_item(title="")
        assert validate_item(item) is False

    def test_empty_tags_rejected(self):
        item = _make_item()
        # Mutate after construction to bypass Pydantic default
        object.__setattr__(item, "tags", [])
        assert validate_item(item) is False

    def test_valid_item_not_in_exclude_passes(self):
        item = _make_item(content_id="ok")
        assert validate_item(item, exclude_ids=frozenset({"other"})) is True


# ---------------------------------------------------------------------------
# _attribute_similarity
# ---------------------------------------------------------------------------


class TestAttributeSimilarity:
    def test_identical_items_score_one(self):
        a = _make_item(category="electronics", tags=["x", "y"])
        sim = _attribute_similarity(a, a)
        assert sim == 1.0

    def test_different_category_different_tags(self):
        a = _make_item(category="electronics", tags=["x"])
        b = _make_item(category="clothing", tags=["y"])
        assert _attribute_similarity(a, b) == 0.0

    def test_same_category_different_tags(self):
        a = _make_item(category="electronics", tags=["x"])
        b = _make_item(category="electronics", tags=["y"])
        sim = _attribute_similarity(a, b)
        assert sim == 0.5  # category match only

    def test_different_category_overlapping_tags(self):
        a = _make_item(category="electronics", tags=["x", "y"])
        b = _make_item(category="clothing", tags=["x", "z"])
        sim = _attribute_similarity(a, b)
        # tag overlap: 1 / 3 (x,y,z) * 0.5 = ~0.167
        assert 0.1 < sim < 0.2


# ---------------------------------------------------------------------------
# mmr_rerank
# ---------------------------------------------------------------------------


class TestMMRRerank:
    def test_empty_input(self):
        assert mmr_rerank([]) == []

    def test_respects_top_k(self):
        items = [_make_item(content_id=f"i{i}") for i in range(10)]
        result = mmr_rerank(items, top_k=3)
        assert len(result) == 3

    def test_top_k_larger_than_input(self):
        items = [_make_item(content_id=f"i{i}") for i in range(3)]
        result = mmr_rerank(items, top_k=10)
        assert len(result) == 3

    def test_first_item_is_most_relevant(self):
        items = [_make_item(content_id=f"i{i}") for i in range(5)]
        result = mmr_rerank(items, top_k=5)
        assert result[0].content_id == "i0"

    def test_diversity_pushes_different_categories(self):
        # 3 electronics, then 1 clothing — MMR should pull clothing forward
        items = [
            _make_item(content_id="e1", category="electronics", tags=["a"]),
            _make_item(content_id="e2", category="electronics", tags=["a"]),
            _make_item(content_id="e3", category="electronics", tags=["a"]),
            _make_item(content_id="c1", category="clothing", tags=["b"]),
        ]
        result = mmr_rerank(items, top_k=4, lambda_param=0.5)
        # With lambda=0.5 (strong diversity), clothing should appear before e3
        categories = [r.category.value for r in result]
        assert "clothing" in categories[:3]

    def test_lambda_one_preserves_original_order(self):
        items = [_make_item(content_id=f"i{i}", tags=[f"t{i}"]) for i in range(5)]
        result = mmr_rerank(items, top_k=5, lambda_param=1.0)
        assert [r.content_id for r in result] == [f"i{i}" for i in range(5)]

    def test_no_duplicates(self):
        items = [_make_item(content_id=f"i{i}") for i in range(5)]
        result = mmr_rerank(items, top_k=5)
        ids = [r.content_id for r in result]
        assert len(ids) == len(set(ids))
