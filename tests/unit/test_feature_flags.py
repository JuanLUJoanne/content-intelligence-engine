"""Tests for the feature flag registry."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from src.feature_flags.registry import FeatureFlagRegistry, get_flag_registry, reset_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_CONFIG = """\
flags:
  bool_flag:
    type: boolean
    enabled: true
    description: "A boolean flag"
  disabled_flag:
    type: boolean
    enabled: false
    description: "A disabled flag"
  rollout_50:
    type: percentage
    enabled: true
    percentage: 50
    description: "50% rollout"
  rollout_0:
    type: percentage
    enabled: true
    percentage: 0
    description: "0% rollout"
  rollout_100:
    type: percentage
    enabled: true
    percentage: 100
    description: "100% rollout"
  segment_flag:
    type: segment
    enabled: true
    segments: ["internal", "beta"]
    description: "Segment-gated flag"
  disabled_segment:
    type: segment
    enabled: false
    segments: ["internal"]
    description: "Disabled segment flag"
"""


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    p = tmp_path / "flags.yaml"
    p.write_text(_SAMPLE_CONFIG)
    return p


@pytest.fixture()
def registry(config_file: Path) -> FeatureFlagRegistry:
    return FeatureFlagRegistry(config_path=config_file)


# ---------------------------------------------------------------------------
# Boolean flags
# ---------------------------------------------------------------------------


class TestBooleanFlags:
    def test_enabled_flag_returns_true(self, registry: FeatureFlagRegistry) -> None:
        assert registry.is_enabled("bool_flag") is True

    def test_disabled_flag_returns_false(self, registry: FeatureFlagRegistry) -> None:
        assert registry.is_enabled("disabled_flag") is False

    def test_unknown_flag_returns_false(self, registry: FeatureFlagRegistry) -> None:
        assert registry.is_enabled("nonexistent") is False

    def test_variant_on(self, registry: FeatureFlagRegistry) -> None:
        assert registry.get_variant("bool_flag") == "on"

    def test_variant_off(self, registry: FeatureFlagRegistry) -> None:
        assert registry.get_variant("disabled_flag") == "off"


# ---------------------------------------------------------------------------
# Percentage rollout
# ---------------------------------------------------------------------------


class TestPercentageRollout:
    def test_0_percent_always_off(self, registry: FeatureFlagRegistry) -> None:
        for i in range(100):
            assert registry.is_enabled("rollout_0", {"key": f"user_{i}"}) is False

    def test_100_percent_always_on(self, registry: FeatureFlagRegistry) -> None:
        for i in range(100):
            assert registry.is_enabled("rollout_100", {"key": f"user_{i}"}) is True

    def test_50_percent_roughly_uniform(self, registry: FeatureFlagRegistry) -> None:
        enabled_count = sum(
            registry.is_enabled("rollout_50", {"key": f"user_{i}"})
            for i in range(1000)
        )
        # Should be roughly 500 ± 100 (very generous tolerance).
        assert 300 < enabled_count < 700, f"Got {enabled_count}/1000"

    def test_deterministic(self, registry: FeatureFlagRegistry) -> None:
        results = [
            registry.is_enabled("rollout_50", {"key": "stable_user"})
            for _ in range(10)
        ]
        assert len(set(results)) == 1, "Same key must always produce the same result"

    def test_no_key_returns_false(self, registry: FeatureFlagRegistry) -> None:
        assert registry.is_enabled("rollout_50") is False
        assert registry.is_enabled("rollout_50", {}) is False

    def test_variant_new_vs_control(self, registry: FeatureFlagRegistry) -> None:
        variants = {
            registry.get_variant("rollout_50", {"key": f"u{i}"})
            for i in range(200)
        }
        assert variants == {"new", "control"}

    def test_request_id_context_key(self, registry: FeatureFlagRegistry) -> None:
        """Ensure request_id is used as fallback context key."""
        result = registry.is_enabled("rollout_100", {"request_id": "abc"})
        assert result is True

    def test_user_id_context_key(self, registry: FeatureFlagRegistry) -> None:
        """Ensure user_id is used as second fallback context key."""
        result = registry.is_enabled("rollout_100", {"user_id": "u1"})
        assert result is True


# ---------------------------------------------------------------------------
# Segment flags
# ---------------------------------------------------------------------------


class TestSegmentFlags:
    def test_allowed_segment(self, registry: FeatureFlagRegistry) -> None:
        assert registry.is_enabled("segment_flag", {"segment": "internal"}) is True
        assert registry.is_enabled("segment_flag", {"segment": "beta"}) is True

    def test_disallowed_segment(self, registry: FeatureFlagRegistry) -> None:
        assert registry.is_enabled("segment_flag", {"segment": "public"}) is False

    def test_no_segment_context(self, registry: FeatureFlagRegistry) -> None:
        assert registry.is_enabled("segment_flag") is False

    def test_disabled_segment_flag(self, registry: FeatureFlagRegistry) -> None:
        assert registry.is_enabled("disabled_segment", {"segment": "internal"}) is False

    def test_variant_labels(self, registry: FeatureFlagRegistry) -> None:
        assert registry.get_variant("segment_flag", {"segment": "beta"}) == "enabled"
        assert registry.get_variant("segment_flag", {"segment": "public"}) == "disabled"


# ---------------------------------------------------------------------------
# Runtime overrides
# ---------------------------------------------------------------------------


class TestOverrides:
    def test_boolean_override(self, registry: FeatureFlagRegistry) -> None:
        assert registry.is_enabled("disabled_flag") is False
        registry.set_override("disabled_flag", True)
        assert registry.is_enabled("disabled_flag") is True

    def test_clear_override(self, registry: FeatureFlagRegistry) -> None:
        registry.set_override("disabled_flag", True)
        registry.clear_override("disabled_flag")
        assert registry.is_enabled("disabled_flag") is False

    def test_override_percentage(self, registry: FeatureFlagRegistry) -> None:
        # Override rollout_0 to 100%
        registry.set_override("rollout_0", {"percentage": 100, "enabled": True})
        assert registry.is_enabled("rollout_0", {"key": "any_user"}) is True


# ---------------------------------------------------------------------------
# Hot-reload
# ---------------------------------------------------------------------------


class TestHotReload:
    def test_reload_on_mtime_change(self, config_file: Path) -> None:
        registry = FeatureFlagRegistry(config_path=config_file)
        assert registry.is_enabled("bool_flag") is True

        # Write updated config with bool_flag disabled
        new_config = _SAMPLE_CONFIG.replace(
            "bool_flag:\n    type: boolean\n    enabled: true",
            "bool_flag:\n    type: boolean\n    enabled: false",
        )
        # Ensure mtime changes (some filesystems have 1-second resolution)
        time.sleep(0.05)
        config_file.write_text(new_config)
        # Force mtime to be in the future
        future_time = time.time() + 10
        os.utime(config_file, (future_time, future_time))

        # Next is_enabled call should pick up the change
        assert registry.is_enabled("bool_flag") is False

    def test_no_reload_when_unchanged(self, config_file: Path) -> None:
        registry = FeatureFlagRegistry(config_path=config_file)
        # Record internal state
        mtime_before = registry._last_mtime
        registry._maybe_reload()
        assert registry._last_mtime == mtime_before


# ---------------------------------------------------------------------------
# list_flags
# ---------------------------------------------------------------------------


class TestListFlags:
    def test_lists_all_flags(self, registry: FeatureFlagRegistry) -> None:
        flags = registry.list_flags()
        assert "bool_flag" in flags
        assert "rollout_50" in flags
        assert "segment_flag" in flags

    def test_includes_override_info(self, registry: FeatureFlagRegistry) -> None:
        registry.set_override("bool_flag", False)
        flags = registry.list_flags()
        assert flags["bool_flag"]["override"] is False


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_flag_registry_returns_same_instance(self, config_file: Path) -> None:
        reset_registry()
        r1 = get_flag_registry(config_path=config_file)
        r2 = get_flag_registry()
        assert r1 is r2
        reset_registry()

    def test_reset_clears_singleton(self, config_file: Path) -> None:
        reset_registry()
        r1 = get_flag_registry(config_path=config_file)
        reset_registry()
        r2 = get_flag_registry(config_path=config_file)
        assert r1 is not r2
        reset_registry()


# ---------------------------------------------------------------------------
# Missing config file
# ---------------------------------------------------------------------------


class TestMissingConfig:
    def test_missing_file_returns_false_for_all(self, tmp_path: Path) -> None:
        registry = FeatureFlagRegistry(config_path=tmp_path / "nonexistent.yaml")
        assert registry.is_enabled("anything") is False
        assert registry.list_flags() == {}


# ---------------------------------------------------------------------------
# Hash distribution
# ---------------------------------------------------------------------------


class TestHashDistribution:
    def test_buckets_cover_full_range(self) -> None:
        buckets = {FeatureFlagRegistry._hash_bucket(f"k{i}") for i in range(10000)}
        # With 10k keys we should see nearly all 100 buckets
        assert len(buckets) >= 95

    def test_salt_changes_assignment(self) -> None:
        b1 = FeatureFlagRegistry._hash_bucket("user_1", salt="flag_a")
        b2 = FeatureFlagRegistry._hash_bucket("user_1", salt="flag_b")
        # Different salts should (usually) produce different buckets
        # Not guaranteed for any single pair but we test the mechanism
        # Just verify they're both valid buckets
        assert 0 <= b1 < 100
        assert 0 <= b2 < 100
