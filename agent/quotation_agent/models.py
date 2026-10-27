"""Structured data shared by the classifier, router and operation nodes."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """All graph routes. The first six values are also MCP tool names."""

    GET_QUOTATION = "get_quotation"
    LIST_CUSTOMER_QUOTATIONS = "list_customer_quotations"
    DUPLICATE_QUOTATION = "duplicate_quotation"
    UPDATE_PRICING_FACTOR = "update_pricing_factor"
    ADJUST_TOTAL_PRICE = "adjust_total_price"
    CONVERT_TO_ORDER = "convert_to_order"
    OFF_TOPIC = "off_topic"
    UNCLEAR = "unclear"


class Intent(BaseModel):
    """The only structured output produced by the LLM classifier."""

    intent: IntentType = Field(
        description=(
            "One of the six quotation operations, or off_topic for unrelated "
            "requests, or unclear when the request is quotation-related but the "
            "operation cannot be determined."
        )
    )
    quotation_number: str | None = Field(
        default=None,
        description="Quotation number in the form QT-YYYYMMDD-NNN.",
    )
    customer_name: str | None = Field(
        default=None,
        description="Customer name for list_customer_quotations.",
    )
    new_pricing_factor: float | None = Field(
        default=None,
        description="New pricing factor for update_pricing_factor. Must be > 0.",
    )
    target_total_price: float | None = Field(
        default=None,
        description="Target total price for adjust_total_price. Must be > 0.",
    )
    reason: str = Field(
        default="",
        description="Short reason for off_topic or unclear classifications.",
    )


OPERATION_INTENTS: tuple[IntentType, ...] = (
    IntentType.GET_QUOTATION,
    IntentType.LIST_CUSTOMER_QUOTATIONS,
    IntentType.DUPLICATE_QUOTATION,
    IntentType.UPDATE_PRICING_FACTOR,
    IntentType.ADJUST_TOTAL_PRICE,
    IntentType.CONVERT_TO_ORDER,
)
