"""
SQLite-backed response cache for LLM calls.

Caching identical (model, prompt, input) triples avoids burning tokens on
repeated requests — common when re-running eval suites or retrying failed
batch jobs. SQLite is sufficient for a single-process pipeline and keeps
deployment simple compared to Redis or Memcached.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import structlog


logger = structlog.get_logger(__name__)


def make_cache_key(model_id: str, prompt_text: str, input_text: str) -> str:
    """Return a stable SHA-256 key for a (model, prompt, input) triple.

    Hashing keeps keys compact and avoids leaking prompt content into log
    lines or filesystem paths.
    """
    h = hashlib.sha256()
    h.update(model_id.encode())
    h.update(b"\x00")
    h.update(prompt_text.encode())
    h.update(b"\x00")
    h.update(input_text.encode())
    return h.hexdigest()


@runtime_checkable
class CacheBackend(Protocol):
    """Minimal interface every cache implementation must satisfy.

    Keeping get/set/delete behind a Protocol lets tests inject in-memory
    fakes and lets production swap SQLite for Redis without touching callers.
    """

    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ttl: int) -> None: ...

    async def delete(self, key: str) -> None: ...

    def stats(self) -> dict[str, Any]: ...


class SQLiteCacheBackend:
    """SQLite implementation of CacheBackend with lazy TTL expiry.

    Expiring entries on read (rather than via a background sweeper) keeps
    the implementation dependency-free and avoids waking a thread just to
    prune old rows — acceptable when hit rates are high and misses are cheap.
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS cache (
            key          TEXT PRIMARY KEY,
            response     TEXT NOT NULL,
            model_id     TEXT NOT NULL,
            created_at   REAL NOT NULL,
            ttl_seconds  INTEGER NOT NULL
        )
    """

    def __init__(self, db_path: str | Path = "data/cache.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0
        self._lock = asyncio.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute(self._CREATE_TABLE)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    async def get(self, key: str) -> str | None:
        """Return cached response string or None if absent/expired."""
        async with self._lock:
            return await asyncio.to_thread(self._get_sync, key)

    def _get_sync(self, key: str) -> str | None:
        now = time.time()
        cur = self._conn.cursor()
        cur.execute(
            "SELECT response, model_id, created_at, ttl_seconds FROM cache WHERE key = ?",
            (key,),
        )
        row = cur.fetchone()
        if row is None:
            self._misses += 1
            logger.info("cache_miss", key_prefix=key[:8])
            return None

        response, model_id, created_at, ttl_seconds = row
        if created_at + ttl_seconds < now:
            # Lazy expiry: remove the stale row so the next writer starts fresh.
            cur.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()
            self._misses += 1
            logger.info(
                "cache_miss", model=model_id, key_prefix=key[:8], reason="expired"
            )
            return None

        self._hits += 1
        logger.info("cache_hit", model=model_id, key_prefix=key[:8])
        return response

    async def set(
        self,
        key: str,
        value: str,
        ttl: int,
        *,
        model_id: str = "",
    ) -> None:
        """Store a response string; upserts so repeat sets don't error."""
        async with self._lock:
            await asyncio.to_thread(self._set_sync, key, value, ttl, model_id)

    def _set_sync(
        self, key: str, value: str, ttl: int, model_id: str
    ) -> None:
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO cache(key, response, model_id, created_at, ttl_seconds)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                response=excluded.response,
                model_id=excluded.model_id,
                created_at=excluded.created_at,
                ttl_seconds=excluded.ttl_seconds
            """,
            (key, value, model_id, now, ttl),
        )
        self._conn.commit()

    async def delete(self, key: str) -> None:
        """Remove a single cache entry by key."""
        async with self._lock:
            await asyncio.to_thread(
                self._conn.execute, "DELETE FROM cache WHERE key = ?", (key,)
            )
            self._conn.commit()

    def stats(self) -> dict[str, Any]:
        """Return hit/miss counts and the derived hit rate."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }
