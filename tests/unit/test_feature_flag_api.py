"""Tests for feature flag API endpoints."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.feature_flags.registry import reset_registry


@pytest.fixture(autouse=True)
def _reset_flags():
    """Ensure flag singleton is fresh for each test."""
    reset_registry()
    yield
    reset_registry()


@pytest.mark.asyncio
async def test_list_flags_returns_all():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/flags")
    assert resp.status_code == 200
    data = resp.json()
    assert "flags" in data
    flags = data["flags"]
    assert "new_model_rollout" in flags
    assert "enhanced_prompt_v2" in flags
    assert "premium_tier_enabled" in flags
    assert "reflection_enabled" in flags


@pytest.mark.asyncio
async def test_toggle_boolean_flag():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # enhanced_prompt_v2 is disabled by default
        resp = await client.get("/api/flags")
        assert resp.json()["flags"]["enhanced_prompt_v2"]["enabled"] is False

        # Enable it
        resp = await client.post("/api/flags/enhanced_prompt_v2", json={"enabled": True})
        assert resp.status_code == 200
        assert resp.json()["flag"] == "enhanced_prompt_v2"

        # Verify override is reflected
        resp = await client.get("/api/flags")
        assert resp.json()["flags"]["enhanced_prompt_v2"].get("override") is True


@pytest.mark.asyncio
async def test_toggle_percentage_flag():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/flags/new_model_rollout",
            json={"percentage": 50},
        )
        assert resp.status_code == 200
        state = resp.json()["state"]
        assert state.get("override") == {"percentage": 50, "enabled": True}


@pytest.mark.asyncio
async def test_clear_override():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Set and then clear
        await client.post("/api/flags/enhanced_prompt_v2", json={"enabled": True})
        resp = await client.post("/api/flags/enhanced_prompt_v2", json={})
        assert resp.status_code == 200
        # Override should be gone
        resp = await client.get("/api/flags")
        assert "override" not in resp.json()["flags"]["enhanced_prompt_v2"]


@pytest.mark.asyncio
async def test_unknown_flag_returns_404():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/flags/nonexistent", json={"enabled": True})
        assert resp.status_code == 404
