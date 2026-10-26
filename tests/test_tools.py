import json
import pytest
from src.tools import TOOL_SCHEMAS, execute_tool


# 中文注释：验证五个工具的 OpenAI schema 结构完整且名称正确。
class TestToolSchemas:
    def test_all_five_tools_defined(self):
        names = {t["function"]["name"] for t in TOOL_SCHEMAS}
        assert names == {"calculator", "get_weather", "search", "convert_units", "lookup_db"}

    def test_schemas_are_openai_format(self):
        for tool in TOOL_SCHEMAS:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]
            assert tool["function"]["parameters"]["type"] == "object"


# 中文注释：覆盖计算器的正常算术和非法表达式防护。
class TestCalculator:
    def test_basic_addition(self):
        result = execute_tool("calculator", {"expression": "3 + 5"})
        assert result == "8"

    def test_compound_expression(self):
        result = execute_tool("calculator", {"expression": "(10 + 5) * 2"})
        assert result == "30"

    def test_division(self):
        result = execute_tool("calculator", {"expression": "100 / 4"})
        assert result == "25.0"

    def test_invalid_expression(self):
        result = execute_tool("calculator", {"expression": "import os"})
        assert "error" in result.lower()


# 中文注释：验证城市查询返回字段以及摄氏度到华氏度的转换。
class TestGetWeather:
    def test_known_city(self):
        result = json.loads(execute_tool("get_weather", {"city": "Paris"}))
        assert "temperature" in result
        assert "condition" in result
        assert isinstance(result["temperature"], (int, float))

    def test_with_units(self):
        result_c = json.loads(execute_tool("get_weather", {"city": "Paris", "units": "celsius"}))
        result_f = json.loads(execute_tool("get_weather", {"city": "Paris", "units": "fahrenheit"}))
        assert result_f["temperature"] > result_c["temperature"]


# 中文注释：验证搜索返回列表并包含与查询相关的事实。
class TestSearch:
    def test_returns_results(self):
        result = json.loads(execute_tool("search", {"query": "Einstein birth year"}))
        assert isinstance(result, list)
        assert len(result) > 0
        assert "text" in result[0]

    def test_relevant_result(self):
        result = json.loads(execute_tool("search", {"query": "Einstein birth year"}))
        assert any("1879" in r["text"] for r in result)


# 中文注释：验证主要单位换算的精度和字符串输出。
class TestConvertUnits:
    def test_celsius_to_fahrenheit(self):
        result = execute_tool("convert_units", {"value": 22, "from_unit": "celsius", "to_unit": "fahrenheit"})
        assert result == "71.6"

    def test_km_to_miles(self):
        result = execute_tool("convert_units", {"value": 10, "from_unit": "km", "to_unit": "miles"})
        assert result == "6.2137"


# 中文注释：验证国家数据库命中和未知键的错误信息。
class TestLookupDb:
    def test_country_population(self):
        result = json.loads(execute_tool("lookup_db", {"table": "countries", "key": "France"}))
        assert "population" in result

    def test_unknown_key(self):
        result = execute_tool("lookup_db", {"table": "countries", "key": "Atlantis"})
        assert "not found" in result.lower()
