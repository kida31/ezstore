from quotation_agent import routing
from quotation_agent.models import Intent, IntentType


def test_extracts_quotation_number_from_sentence() -> None:
    text = "我要查詢報價單，報價單編號：qt-20260330-001。"
    assert routing.extract_quotation_number(text) == "QT-20260330-001"


def test_extracts_factor_and_total() -> None:
    assert (
        routing.extract_pricing_factor("報價因子改為 1.5") == 1.5
    )
    assert routing.extract_target_total("總價改為 8000") == 8000.0


def test_extracts_customer_name_without_suffix() -> None:
    assert (
        routing.extract_customer_name("客戶名稱：史塔克工業的報價單")
        == "史塔克工業"
    )


def test_weather_is_off_topic() -> None:
    assert routing.deterministic_off_topic("今天台北天氣如何？") == "天氣"


def test_customer_crud_is_off_topic() -> None:
    assert (
        routing.deterministic_off_topic("幫我新增一個客戶，名稱：測試公司")
        == "客戶或產品的新增修改"
    )


def test_customer_quotation_list_is_not_off_topic() -> None:
    assert (
        routing.deterministic_off_topic(
            "我要查詢客戶的報價單清單，客戶名稱：史塔克工業"
        )
        is None
    )


def test_convert_to_order_is_not_off_topic() -> None:
    assert (
        routing.deterministic_off_topic(
            "我要將報價單轉訂單，報價單編號：QT-20260330-001"
        )
        is None
    )


def test_clarification_continuation_with_quotation_number() -> None:
    pending = Intent(intent=IntentType.GET_QUOTATION)
    classified = Intent(intent=IntentType.UNCLEAR)

    merged = routing.merge_intent(
        classified, pending, "QT-20260330-001"
    )

    assert merged.intent == IntentType.GET_QUOTATION
    assert merged.quotation_number == "QT-20260330-001"
    assert routing.missing_fields(merged) == []


def test_clarification_continuation_with_factor() -> None:
    pending = Intent(
        intent=IntentType.UPDATE_PRICING_FACTOR,
        quotation_number="QT-20260330-001",
    )
    classified = Intent(intent=IntentType.UNCLEAR)

    merged = routing.merge_intent(classified, pending, "1.5")

    assert merged.intent == IntentType.UPDATE_PRICING_FACTOR
    assert merged.quotation_number == "QT-20260330-001"
    assert merged.new_pricing_factor == 1.5
    assert routing.missing_fields(merged) == []


def test_clarification_continuation_with_customer_name() -> None:
    pending = Intent(intent=IntentType.LIST_CUSTOMER_QUOTATIONS)
    classified = Intent(intent=IntentType.UNCLEAR)

    merged = routing.merge_intent(classified, pending, "史塔克工業")

    assert merged.intent == IntentType.LIST_CUSTOMER_QUOTATIONS
    assert merged.customer_name == "史塔克工業"
    assert routing.missing_fields(merged) == []


def test_clarify_message_asks_for_missing_field() -> None:
    message = routing.clarify_message(
        IntentType.GET_QUOTATION.value, ["quotation_number"]
    )
    assert message.startswith("請提供")
    assert "報價單編號" in message


def test_infer_pending_from_history_text() -> None:
    pending = routing.infer_pending_from_user_text("我要查詢報價單")
    assert pending is not None
    assert pending.intent == IntentType.GET_QUOTATION
