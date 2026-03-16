"""
Content metadata schemas shared across the pipeline.

Defining a single, strongly‑typed metadata model up front prevents an entire
class of silent drift between ingestion, enrichment, and storage layers. The
enums and validators here encode domain assumptions once so that any schema
breakage is caught immediately during validation instead of deep inside model
prompts.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


logger = structlog.get_logger(__name__)


class Category(str, Enum):
    """Top-level product category for classification."""

    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    HOME = "home"
    SPORTS = "sports"
    BOOKS = "books"


class Condition(str, Enum):
    """Item condition for product listings."""

    NEW = "new"
    LIKE_NEW = "like_new"
    GOOD = "good"
    FAIR = "fair"
    REFURBISHED = "refurbished"


class PriceRange(str, Enum):
    """Price tier for routing and cost estimation."""

    BUDGET = "budget"
    MID_RANGE = "mid_range"
    PREMIUM = "premium"
    LUXURY = "luxury"
    UNPRICED = "unpriced"


class ContentMetadata(BaseModel):
    """Canonical schema describing a single content item.

    This model acts as the contract between upstream enrichment (e.g. LLM
    classification) and downstream consumers; tightening it here makes retries
    and migrations far less error‑prone.
    """

    model_config = ConfigDict(extra="forbid")

    content_id: str = Field(description="Stable identifier used across the pipeline.")
    title: str
    description: Optional[str] = None
    category: Category
    condition: Condition
    price_range: PriceRange
    tags: List[str] = Field(default_factory=list)
    language: str = Field(default="en")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the item first entered the pipeline.",
    )

    @field_validator("tags", mode="after")
    @classmethod
    def _normalise_tags(cls, value: List[str]) -> List[str]:
        """Lower‑case and de‑duplicate tags to keep indexing predictable."""

        seen = set()
        normalised: List[str] = []
        for t in value:
            slug = t.strip().lower().replace(" ", "_")
            if slug and slug not in seen:
                seen.add(slug)
                normalised.append(slug)
        return normalised

    @field_validator("description")
    @classmethod
    def _truncate_description(cls, value: Optional[str]) -> Optional[str]:
        """Trim overly long descriptions to avoid bloating prompts."""

        if value is None:
            return None
        max_len = 2000
        if len(value) > max_len:
            logger.info(
                "metadata_description_truncated",
                original_length=len(value),
                max_len=max_len,
            )
            return value[:max_len]
        return value

    @field_validator("price_range")
    @classmethod
    def _validate_price_range_vs_condition(
        cls,
        value: PriceRange,
        info,
    ) -> PriceRange:
        """Log suspicious price_range/condition combinations for later analysis.

        Instead of rejecting combinations outright we treat them as weak
        signals that the upstream classifier might be drifting, which can be
        surfaced via metrics without blocking the pipeline.
        """

        condition = info.data.get("condition")
        if condition is not None:
            if condition is Condition.FAIR and value is PriceRange.LUXURY:
                logger.warning(
                    "metadata_unusual_price_range_for_condition",
                    condition=condition.value,
                    price_range=value.value,
                )
        return value


class SchemaRetryPrompt(BaseModel):
    """Template for asking an LLM to repair an invalid schema instance.

    Encoding this as a dedicated type makes it easy to keep retry behaviour
    consistent across call‑sites and to upgrade prompts in one place when the
    schema evolves.
    """

    message: str

    @classmethod
    def from_validation_error(
        cls,
        error: ValidationError,
        *,
        schema_name: str = "ContentMetadata",
    ) -> "SchemaRetryPrompt":
        """Construct a retry prompt that explains how the payload was invalid."""

        return cls(
            message=(
                f"The previous response could not be parsed into the {schema_name} "
                "schema. Please respond again with ONLY a JSON object that matches "
                "the following error report:\n"
                f"{error}\n"
                "Do not include explanations, comments, or additional text."
            )
        )

