"""
LLM-in-the-loop adaptive retrieval for recommendation.

Standard retrieval fires a single fixed query derived from the buyer profile.
AdaptiveRetriever wraps AssetRetriever and lets the LLM inspect search results,
then decide whether to refine the query or accept the current set.  This
observe-reason-act loop is the core agentic pattern: the LLM is not just
generating text — it is making tool-use decisions based on intermediate results.

The loop is bounded by ``max_rounds`` (default 3) so cost and latency stay
predictable, and every search call goes through the same AssetRetriever that
the non-adaptive path uses, keeping scoring logic in one place.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import structlog

from src.agents.memory.buyer_profile import BuyerProfile
from src.gateway.providers import LLMProvider
from src.retrieval.asset_retriever import AssetRetriever
from src.schemas.metadata import ContentMetadata


logger = structlog.get_logger(__name__)


# Tool definition following MCP / Claude function-calling schema so the
# same structure works if this is later wired to a real Claude tool_use loop.
SEARCH_TOOL = {
    "name": "search_assets",
    "description": (
        "Search the product catalog. Returns up to 5 items ranked by "
        "relevance. You may call this multiple times with different queries "
        "to refine results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text search terms (space-separated).",
            },
            "filters": {
                "type": "object",
                "description": "Optional structured filters.",
                "properties": {
                    "category": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "required": ["query"],
    },
}

_SYSTEM_PROMPT = """\
You are a product search assistant. Your goal is to find the most relevant \
products for a buyer.

Buyer profile:
- Top category: {top_category}
- Top interest tags: {top_tags}
- Purchase history: {purchase_ids}

You have a search_assets tool. You may search up to {max_rounds} times.

After each search you will see the results. Decide:
- If the results are relevant and diverse enough, respond with exactly: DONE
- If the results are not good enough, call search_assets again with a \
different query.

Strategy tips:
- Start with the most specific query (e.g. top tags).
- If results are poor, try broadening (e.g. category name) or narrowing.
- Try combining different tags or adding category filters.\
"""

_RESULTS_TEMPLATE = """\
Search results for "{query}":
{result_lines}

Are these results relevant? If yes, respond with DONE. \
If not, call search_assets with a different query.\
"""


class AdaptiveRetriever:
    """LLM-driven iterative search with up to ``max_rounds`` refinements.

    On each round the LLM either calls ``search_assets`` (with a query and
    optional filters it chooses) or emits ``DONE`` to accept the accumulated
    results.  The best ``top_n`` unique items across all rounds are returned.

    When the provider is a DummyProvider the LLM cannot actually reason, so
    the first round's results are returned immediately — no cost or latency
    penalty in tests.
    """

    def __init__(
        self,
        retriever: AssetRetriever,
        provider: LLMProvider,
        model_id: str = "dummy",
        *,
        max_rounds: int = 3,
        top_n: int = 5,
    ) -> None:
        self._retriever = retriever
        self._provider = provider
        self._model_id = model_id
        self._max_rounds = max_rounds
        self._top_n = top_n

    async def search(
        self,
        profile: BuyerProfile,
        *,
        initial_query: Optional[str] = None,
        initial_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ContentMetadata]:
        """Run the adaptive retrieval loop.

        Parameters
        ----------
        profile:
            Buyer profile used to build the system prompt context.
        initial_query:
            Optional first-round query.  If omitted, the LLM decides the
            first query as well.
        initial_filters:
            Optional first-round filters (passed only on round 1).
        """
        tag_affinity = profile.get("tag_affinity", {})
        top_tags = sorted(tag_affinity, key=tag_affinity.get, reverse=True)[:5]

        system_prompt = _SYSTEM_PROMPT.format(
            top_category=profile.get("top_category", "unknown"),
            top_tags=", ".join(top_tags) if top_tags else "none",
            purchase_ids=", ".join(profile.get("purchase_history", [])[:5]) or "none",
            max_rounds=self._max_rounds,
        )

        seen_ids: set[str] = set()
        all_results: list[ContentMetadata] = []

        # Round 1: use initial query if provided, otherwise ask LLM
        if initial_query is not None:
            results = self._retriever.search(initial_query, initial_filters)
            all_results, seen_ids = self._merge(all_results, results, seen_ids)

            logger.info(
                "adaptive_retriever_round",
                round=1,
                query=initial_query,
                result_count=len(results),
                total_unique=len(all_results),
            )

            # Ask LLM if these results are good enough
            result_lines = self._format_results(results)
            eval_prompt = (
                f"{system_prompt}\n\n"
                + _RESULTS_TEMPLATE.format(
                    query=initial_query,
                    result_lines=result_lines,
                )
            )

            try:
                llm_response = await self._provider.generate(
                    eval_prompt, self._model_id
                )
                if self._is_done(llm_response):
                    logger.info("adaptive_retriever_done", rounds=1)
                    return all_results[: self._top_n]

                # LLM wants to refine — extract next query
                next_query, next_filters = self._parse_tool_call(llm_response)
            except Exception as exc:
                logger.warning(
                    "adaptive_retriever_llm_error",
                    round=1,
                    error=str(exc),
                )
                return all_results[: self._top_n]

            start_round = 2
        else:
            # No initial query — ask LLM to generate the first one
            ask_prompt = (
                f"{system_prompt}\n\n"
                "Generate your first search query by calling search_assets."
            )
            try:
                llm_response = await self._provider.generate(
                    ask_prompt, self._model_id
                )
                next_query, next_filters = self._parse_tool_call(llm_response)
            except Exception as exc:
                logger.warning(
                    "adaptive_retriever_llm_error",
                    round=0,
                    error=str(exc),
                )
                # Fallback: use top tags as query
                fallback_query = " ".join(top_tags) if top_tags else profile.get("top_category", "")
                results = self._retriever.search(fallback_query)
                return results[: self._top_n]

            start_round = 1

        # Subsequent rounds
        for round_num in range(start_round, self._max_rounds + 1):
            results = self._retriever.search(next_query, next_filters)
            all_results, seen_ids = self._merge(all_results, results, seen_ids)

            logger.info(
                "adaptive_retriever_round",
                round=round_num,
                query=next_query,
                result_count=len(results),
                total_unique=len(all_results),
            )

            if round_num >= self._max_rounds:
                logger.info("adaptive_retriever_max_rounds", rounds=round_num)
                break

            # Ask LLM to evaluate
            result_lines = self._format_results(results)
            eval_prompt = (
                f"{system_prompt}\n\n"
                + _RESULTS_TEMPLATE.format(
                    query=next_query,
                    result_lines=result_lines,
                )
            )

            try:
                llm_response = await self._provider.generate(
                    eval_prompt, self._model_id
                )

                if self._is_done(llm_response):
                    logger.info("adaptive_retriever_done", rounds=round_num)
                    break

                next_query, next_filters = self._parse_tool_call(llm_response)
            except Exception as exc:
                logger.warning(
                    "adaptive_retriever_llm_error",
                    round=round_num,
                    error=str(exc),
                )
                break

        return all_results[: self._top_n]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge(
        accumulated: list[ContentMetadata],
        new: list[ContentMetadata],
        seen_ids: set[str],
    ) -> tuple[list[ContentMetadata], set[str]]:
        """Append new results that haven't been seen yet."""
        for item in new:
            if item.content_id not in seen_ids:
                accumulated.append(item)
                seen_ids.add(item.content_id)
        return accumulated, seen_ids

    @staticmethod
    def _format_results(results: list[ContentMetadata]) -> str:
        lines = []
        for r in results:
            tags_str = ", ".join(r.tags[:4])
            lines.append(
                f"- {r.title} (category={r.category.value}, "
                f"price={r.price_range.value}, tags=[{tags_str}])"
            )
        return "\n".join(lines) if lines else "(no results)"

    @staticmethod
    def _is_done(response: Any) -> bool:
        """Check if the LLM response indicates satisfaction with results."""
        if isinstance(response, dict):
            # DummyProvider returns a dict — treat as non-DONE so tests
            # exercise at least one search round, then fall through.
            text = json.dumps(response)
        else:
            text = str(response)
        return "DONE" in text.upper()

    @staticmethod
    def _parse_tool_call(
        response: Any,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Extract query and filters from an LLM response.

        Supports two response shapes:
        1. A dict with a "query" key (from structured JSON providers).
        2. A dict with a tool-call-style structure.

        Falls back to using the full response as a query string.
        """
        if isinstance(response, dict):
            # Direct JSON response with query field
            if "query" in response:
                return response["query"], response.get("filters")

            # Tool call wrapper: {"name": "search_assets", "input": {...}}
            tool_input = response.get("input", response.get("arguments", {}))
            if isinstance(tool_input, dict) and "query" in tool_input:
                return tool_input["query"], tool_input.get("filters")

            # DummyProvider returns product-like dicts; use title as query
            if "title" in response:
                return response["title"], None

        # Last resort: stringify
        return str(response).strip()[:100], None
