from quotation_agent.mcp_client import parse_tool_result


def test_parses_successful_text_result() -> None:
    raw = {
        "content": [{"type": "text", "text": "報價單編號: QT-20260330-001"}],
        "isError": False,
    }

    result = parse_tool_result("get_quotation", raw)

    assert result.is_error is False
    assert result.text == "報價單編號: QT-20260330-001"


def test_strips_adk_error_prefix() -> None:
    raw = {
        "content": [
            {
                "type": "text",
                "text": "Error executing tool get_quotation: 找不到報價單編號 QT-20990101-001",
            }
        ],
        "isError": True,
    }

    result = parse_tool_result("get_quotation", raw)

    assert result.is_error is True
    assert result.text == "找不到報價單編號 QT-20990101-001"


def test_falls_back_to_structured_content() -> None:
    raw = {
        "content": [],
        "structuredContent": {
            "result": "客戶「史塔克工業」的報價單清單（共 1 筆）"
        },
    }

    result = parse_tool_result("list_customer_quotations", raw)

    assert result.text.startswith("客戶「史塔克工業」")
