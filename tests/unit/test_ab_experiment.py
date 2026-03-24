"""Tests for ABExperiment: variant assignment, prompt generation, result logging."""

from __future__ import annotations

import pytest

from src.ab_test.experiment import ABExperiment, Variant
from src.agents.memory.buyer_profile import BuyerProfile
from src.api.review import ReviewStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile(
    *,
    user_id: str = "test-user",
    top_category: str = "electronics",
    purchase_ids: list[str] | None = None,
    tag_affinity: dict[str, float] | None = None,
) -> BuyerProfile:
    """Return a minimal BuyerProfile for testing."""
    return BuyerProfile(
        user_id=user_id,
        purchase_history=purchase_ids if purchase_ids is not None else ["asset-1", "asset-2"],
        browsing_history=["asset-3", "asset-4"],
        tag_affinity=tag_affinity if tag_affinity is not None else {
            "bluetooth": 1.0,
            "wireless": 0.8,
            "speaker": 0.6,
        },
        top_category=top_category,
    )


def _make_experiment(tmp_path=None) -> tuple[ABExperiment, ReviewStore]:
    """Return an ABExperiment wired to a fresh ReviewStore."""
    import tempfile, pathlib
    path = str((pathlib.Path(tmp_path) if tmp_path else pathlib.Path(tempfile.mkdtemp())) / "golden.json")
    store = ReviewStore(golden_set_path=path)
    return ABExperiment(review_store=store), store


# ---------------------------------------------------------------------------
# Variant assignment
# ---------------------------------------------------------------------------


class TestVariantAssignment:
    def test_same_user_always_gets_same_variant(self):
        """Repeated calls with the same user_id must return the same variant."""
        exp = ABExperiment()
        user_id = "stable-user-abc-123"
        first = exp.assign_variant(user_id)
        for _ in range(10):
            assert exp.assign_variant(user_id) == first

    def test_deterministic_across_separate_instances(self):
        """Two independent ABExperiment objects must agree on variant."""
        user_id = "cross-instance-user"
        v1 = ABExperiment().assign_variant(user_id)
        v2 = ABExperiment().assign_variant(user_id)
        assert v1 == v2

    def test_returns_valid_variant_enum(self):
        exp = ABExperiment()
        v = exp.assign_variant("any-user-id")
        assert isinstance(v, Variant)
        assert v in (Variant.A, Variant.B)

    def test_distribution_roughly_50_50_across_1000_users(self):
        """hash(user_id) % 2 should split ~500/500 over a large population."""
        exp = ABExperiment()
        variants = [exp.assign_variant(f"user-{i}") for i in range(1000)]
        count_a = variants.count(Variant.A)
        count_b = variants.count(Variant.B)
        # Allow ±10% tolerance around the expected 50/50 split.
        assert 400 <= count_a <= 600, f"Variant A count {count_a} outside [400, 600]"
        assert 400 <= count_b <= 600, f"Variant B count {count_b} outside [400, 600]"

    def test_all_1000_users_get_a_variant(self):
        exp = ABExperiment()
        for i in range(1000):
            v = exp.assign_variant(f"user-{i}")
            assert v in (Variant.A, Variant.B)


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


class TestPromptGeneration:
    def test_variant_a_and_b_produce_different_prompts(self):
        exp = ABExperiment()
        profile = _make_profile()
        assert exp.get_prompt(Variant.A, profile) != exp.get_prompt(Variant.B, profile)

    def test_variant_a_references_purchase_history(self):
        exp = ABExperiment()
        profile = _make_profile(purchase_ids=["item-99", "item-100"])
        prompt = exp.get_prompt(Variant.A, profile)
        # Variant A is purchase-oriented.
        assert any(kw in prompt.lower() for kw in ("purchase", "previous", "bought"))

    def test_variant_b_references_browsing_tags(self):
        exp = ABExperiment()
        profile = _make_profile(tag_affinity={"headphones": 1.0, "audiophile": 0.9, "hifi": 0.7})
        prompt = exp.get_prompt(Variant.B, profile)
        # The top tag must appear in the prompt.
        assert "headphones" in prompt

    def test_variant_a_contains_top_category(self):
        exp = ABExperiment()
        profile = _make_profile(top_category="sports")
        assert "sports" in exp.get_prompt(Variant.A, profile)

    def test_variant_b_contains_top_category(self):
        exp = ABExperiment()
        profile = _make_profile(top_category="books")
        assert "books" in exp.get_prompt(Variant.B, profile)

    def test_variant_b_includes_top_3_affinity_tags(self):
        exp = ABExperiment()
        # 5 tags; only the top 3 by score should appear.
        profile = _make_profile(tag_affinity={
            "yoga": 1.0,
            "fitness": 0.9,
            "mat": 0.7,
            "towel": 0.3,
            "bottle": 0.1,
        })
        prompt = exp.get_prompt(Variant.B, profile)
        assert "yoga" in prompt
        assert "fitness" in prompt
        assert "mat" in prompt

    def test_variant_a_with_no_purchases_shows_none(self):
        exp = ABExperiment()
        profile = _make_profile(purchase_ids=[])
        prompt = exp.get_prompt(Variant.A, profile)
        assert "none" in prompt.lower()

    def test_variant_b_with_no_affinity_tags_still_returns_prompt(self):
        exp = ABExperiment()
        profile = _make_profile(tag_affinity={})
        prompt = exp.get_prompt(Variant.B, profile)
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------


class TestLogResult:
    def test_log_result_stores_data_correctly(self, tmp_path):
        exp, store = _make_experiment(tmp_path)

        exp.log_result(
            user_id="user-42",
            variant=Variant.A,
            email_content="Hi, check out these products!",
            asset_ids=["asset-1", "asset-2"],
        )

        pending = store.get_pending()
        assert len(pending) == 1
        item = pending[0]
        assert item.record["user_id"] == "user-42"
        assert item.record["variant"] == Variant.A.value
        assert item.record["asset_ids"] == ["asset-1", "asset-2"]
        assert item.output["email"] == "Hi, check out these products!"

    def test_log_result_returns_string_item_id(self, tmp_path):
        exp, _ = _make_experiment(tmp_path)
        item_id = exp.log_result("u1", Variant.B, "email body", ["a1"])
        assert isinstance(item_id, str)
        assert len(item_id) > 0

    def test_multiple_results_accumulate_in_store(self, tmp_path):
        exp, store = _make_experiment(tmp_path)
        for i in range(5):
            exp.log_result(f"user-{i}", Variant.A, f"email {i}", [f"asset-{i}"])
        assert len(store.get_pending()) == 5

    def test_variant_b_result_recorded_with_correct_variant_value(self, tmp_path):
        exp, store = _make_experiment(tmp_path)
        exp.log_result("buyer-99", Variant.B, "discovery email", ["p1", "p2", "p3"])
        item = store.get_pending()[0]
        assert item.record["variant"] == Variant.B.value

    def test_reason_label_includes_variant_value(self, tmp_path):
        exp, store = _make_experiment(tmp_path)
        exp.log_result("u", Variant.A, "email", [])
        item = store.get_pending()[0]
        assert Variant.A.value in item.reason
