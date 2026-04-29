"""
In-memory feature flag registry with YAML config and hot-reload.

Flags are loaded from a YAML file at startup and re-read automatically when
the file's mtime changes. No external dependencies (no LaunchDarkly, no
Unleash) — the entire system is ~120 lines of pure Python so it stays easy
to reason about in interviews and production debugging.

Flag types:

* **boolean** — simple on/off toggle.
* **percentage** — hash-based rollout (0–100). Uses the same deterministic
  ``hash(context_key) % 100`` pattern as the A/B experiment framework so
  the same user always gets the same decision within a process.
* **segment** — enabled only when the request's segment label appears in an
  allowed list (e.g. ``["internal", "beta"]``).
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


logger = structlog.get_logger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "feature_flags.yaml"


class FeatureFlagRegistry:
    """Thread-safe feature flag store with file-mtime hot-reload.

    Parameters
    ----------
    config_path:
        Path to YAML flag definitions. Defaults to ``config/feature_flags.yaml``
        relative to the repo root.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._path = config_path or _DEFAULT_CONFIG_PATH
        self._lock = threading.Lock()
        self._flags: Dict[str, Dict[str, Any]] = {}
        self._overrides: Dict[str, Any] = {}
        self._last_mtime: float = 0.0
        self._load()

    # ------------------------------------------------------------------
    # Loading / hot-reload
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Read the YAML file and replace the in-memory flag dict."""
        if not self._path.exists():
            logger.warning("feature_flags_config_missing", path=str(self._path))
            return

        try:
            mtime = os.path.getmtime(self._path)
        except OSError:
            return

        if yaml is None:
            raw_text = self._path.read_text()
            flags = self._parse_simple_yaml(raw_text)
        else:
            with open(self._path) as fh:
                flags = yaml.safe_load(fh) or {}

        with self._lock:
            self._flags = flags.get("flags", flags)
            self._last_mtime = mtime

        logger.info("feature_flags_loaded", count=len(self._flags), path=str(self._path))

    def _maybe_reload(self) -> None:
        """Re-read the config file if its mtime has changed."""
        try:
            mtime = os.path.getmtime(self._path)
        except OSError:
            return
        if mtime > self._last_mtime:
            self._load()

    @staticmethod
    def _parse_simple_yaml(text: str) -> Dict[str, Any]:
        """Minimal YAML-subset parser (flat key: value only) for envs without PyYAML.

        Only used as a fallback — production installs should have PyYAML available.
        """
        import json as _json

        result: Dict[str, Any] = {}
        stack: list[tuple[int, dict]] = [(-1, result)]

        for line in text.splitlines():
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(stripped)
            if ":" not in stripped:
                continue

            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()

            while stack and indent <= stack[-1][0]:
                stack.pop()

            parent = stack[-1][1]

            if value == "":
                child: Dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                if value.startswith("[") and value.endswith("]"):
                    try:
                        value = _json.loads(value)
                    except Exception:
                        pass
                elif value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                parent[key] = value

        return result

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def is_enabled(self, flag_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Return whether *flag_name* is active for the given context.

        * **boolean** flags: return the ``enabled`` value directly.
        * **percentage** flags: ``hash(context["key"]) % 100 < percentage``.
        * **segment** flags: ``context["segment"] in allowed_segments``.
        """
        self._maybe_reload()

        with self._lock:
            if flag_name in self._overrides:
                override = self._overrides[flag_name]
                if isinstance(override, bool):
                    return override
                # For percentage overrides, treat as new percentage value
                # (handled below via flag copy)

            flag = self._flags.get(flag_name)

        if flag is None:
            return False

        # Apply runtime override to percentage if present
        with self._lock:
            if flag_name in self._overrides and not isinstance(self._overrides[flag_name], bool):
                flag = {**flag, **self._overrides[flag_name]}

        flag_type = flag.get("type", "boolean")
        enabled = flag.get("enabled", False)

        if not enabled:
            return False

        ctx = context or {}

        if flag_type == "boolean":
            return True

        if flag_type == "percentage":
            percentage = flag.get("percentage", 0)
            key = ctx.get("key", ctx.get("request_id", ctx.get("user_id", "")))
            if not key:
                return False
            bucket = self._hash_bucket(str(key), flag_name)
            return bucket < percentage

        if flag_type == "segment":
            allowed: List[str] = flag.get("segments", [])
            request_segment = ctx.get("segment", "")
            return request_segment in allowed

        return False

    def get_variant(self, flag_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Return the active variant name for a multi-variant flag.

        For boolean flags returns ``"on"`` / ``"off"``.
        For percentage flags returns ``"new"`` / ``"control"``.
        For segment flags returns ``"enabled"`` / ``"disabled"``.
        """
        enabled = self.is_enabled(flag_name, context)

        with self._lock:
            flag = self._flags.get(flag_name, {})

        flag_type = flag.get("type", "boolean")

        if flag_type == "percentage":
            return "new" if enabled else "control"
        if flag_type == "segment":
            return "enabled" if enabled else "disabled"
        return "on" if enabled else "off"

    # ------------------------------------------------------------------
    # Runtime override (for admin API)
    # ------------------------------------------------------------------

    def set_override(self, flag_name: str, value: Any) -> None:
        """Set a runtime override for a flag (does not persist to YAML)."""
        with self._lock:
            self._overrides[flag_name] = value
        logger.info("feature_flag_override_set", flag=flag_name, value=value)

    def clear_override(self, flag_name: str) -> None:
        """Remove a runtime override, reverting to YAML config."""
        with self._lock:
            self._overrides.pop(flag_name, None)
        logger.info("feature_flag_override_cleared", flag=flag_name)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_flags(self) -> Dict[str, Dict[str, Any]]:
        """Return all flag definitions with current override status."""
        self._maybe_reload()
        with self._lock:
            result = {}
            for name, flag in self._flags.items():
                entry = {**flag}
                if name in self._overrides:
                    entry["override"] = self._overrides[name]
                result[name] = entry
            return result

    # ------------------------------------------------------------------
    # Deterministic hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_bucket(key: str, salt: str = "") -> int:
        """Map a key to a 0–99 bucket using SHA-256 for uniform distribution.

        SHA-256 gives better uniformity than Python's built-in ``hash()``
        (which is randomised per process) and is consistent across restarts
        and machines — important for percentage rollouts that should be stable.
        """
        digest = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
        return int(digest, 16) % 100


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_REGISTRY: Optional[FeatureFlagRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def get_flag_registry(config_path: Optional[Path] = None) -> FeatureFlagRegistry:
    """Return (or create) the process-wide flag registry singleton."""
    global _REGISTRY
    if _REGISTRY is not None and config_path is None:
        return _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None or config_path is not None:
            _REGISTRY = FeatureFlagRegistry(config_path=config_path)
        return _REGISTRY


def reset_registry() -> None:
    """Reset the singleton (for tests)."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        _REGISTRY = None
