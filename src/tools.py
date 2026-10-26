"""Deterministic tool environment for RL training.

5 tools with fixed outputs so reward computation is reproducible.
Uses a seeded knowledge base — no external API calls.
"""
# 中文注释：本模块定义确定性的工具环境；固定数据让每次训练的工具结果和奖励都可复现。
import json

# --- Knowledge bases (deterministic) ---

# 中文注释：天气数据集是内置的只读知识库，不会发起外部网络请求。
WEATHER_DB = {
    "paris": {"temperature": 18, "condition": "partly cloudy", "humidity": 65},
    "london": {"temperature": 12, "condition": "rainy", "humidity": 80},
    "tokyo": {"temperature": 24, "condition": "sunny", "humidity": 55},
    "new york": {"temperature": 15, "condition": "cloudy", "humidity": 70},
    "sydney": {"temperature": 26, "condition": "sunny", "humidity": 45},
    "berlin": {"temperature": 10, "condition": "overcast", "humidity": 75},
    "mumbai": {"temperature": 32, "condition": "humid", "humidity": 85},
    "cairo": {"temperature": 35, "condition": "sunny", "humidity": 20},
    "seattle": {"temperature": 11, "condition": "rainy", "humidity": 82},
    "dubai": {"temperature": 38, "condition": "sunny", "humidity": 30},
}

# 中文注释：搜索数据集用关键词重叠模拟一个可控的检索服务。
SEARCH_DB = {
    "einstein birth year": [
        {"text": "Albert Einstein was born on March 14, 1879 in Ulm, Germany.", "score": 0.95},
        {"text": "Einstein published his theory of relativity in 1905.", "score": 0.7},
    ],
    "speed of light": [
        {"text": "The speed of light in a vacuum is 299,792,458 meters per second.", "score": 0.98},
        {"text": "Light travels at approximately 186,000 miles per second.", "score": 0.85},
    ],
    "python creator": [
        {"text": "Python was created by Guido van Rossum and first released in 1991.", "score": 0.96},
    ],
    "tallest mountain": [
        {"text": "Mount Everest is the tallest mountain at 8,849 meters (29,032 feet).", "score": 0.97},
    ],
    "water boiling point": [
        {"text": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard pressure.", "score": 0.99},
    ],
    "earth distance sun": [
        {"text": "The average distance from Earth to the Sun is about 149.6 million km (93 million miles).", "score": 0.94},
    ],
    "population france": [
        {"text": "France has a population of approximately 67.75 million people (2024).", "score": 0.93},
    ],
    "pi value": [
        {"text": "Pi is approximately 3.14159265358979.", "score": 0.99},
    ],
}

# 中文注释：国家数据集为数据库查询工具提供人口、面积、首都和 GDP 字段。
COUNTRIES_DB = {
    "france": {"population": 67750000, "capital": "Paris", "area_km2": 551695, "gdp_billion_usd": 2780},
    "japan": {"population": 125700000, "capital": "Tokyo", "area_km2": 377975, "gdp_billion_usd": 4230},
    "brazil": {"population": 216400000, "capital": "Brasilia", "area_km2": 8515767, "gdp_billion_usd": 2130},
    "germany": {"population": 84360000, "capital": "Berlin", "area_km2": 357114, "gdp_billion_usd": 4070},
    "india": {"population": 1428600000, "capital": "New Delhi", "area_km2": 3287263, "gdp_billion_usd": 3730},
}

# 中文注释：单位转换表把一对单位映射到纯函数，便于稳定地执行数值转换。
UNIT_CONVERSIONS = {
    ("celsius", "fahrenheit"): lambda v: round(v * 9 / 5 + 32, 4),
    ("fahrenheit", "celsius"): lambda v: round((v - 32) * 5 / 9, 4),
    ("km", "miles"): lambda v: round(v * 0.62137, 4),
    ("miles", "km"): lambda v: round(v * 1.60934, 4),
    ("kg", "lbs"): lambda v: round(v * 2.20462, 4),
    ("lbs", "kg"): lambda v: round(v * 0.45359, 4),
    ("meters", "feet"): lambda v: round(v * 3.28084, 4),
    ("feet", "meters"): lambda v: round(v * 0.3048, 4),
}

# --- Tool schemas (OpenAI function calling format) ---

# 中文注释：这些 schema 遵循 OpenAI function calling 格式，会直接传给模型。
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Returns the numeric result as a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression, e.g. '(3+5)*2'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city. Returns temperature, condition, and humidity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units. Default: celsius"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search a knowledge base for information. Returns a list of relevant text snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_units",
            "description": "Convert a value between units. Supports: celsius/fahrenheit, km/miles, kg/lbs, meters/feet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "The numeric value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit"},
                    "to_unit": {"type": "string", "description": "Target unit"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_db",
            "description": "Look up a record in a database table. Available tables: countries. Returns a JSON record.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Table name: 'countries'"},
                    "key": {"type": "string", "description": "Record key (e.g. country name)"},
                },
                "required": ["table", "key"],
            },
        },
    },
]

# --- Tool executors ---

# 中文注释：计算器先做字符白名单校验，再在禁用内置函数的环境中求值。
def _safe_calc(expression: str) -> str:
    """Evaluate math expression safely using character allowlist and no builtins.

    Note: eval() is used intentionally here for this RL training environment.
    The expression is strictly restricted to digits and math operators via
    character allowlisting, and __builtins__ is disabled. This is not
    user-facing code.
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters in expression. Only numbers and +-*/.() allowed."
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# 中文注释：天气执行器规范化城市和温度单位，并把记录序列化为 JSON。
def _get_weather(args: dict) -> str:
    city = args.get("city", "").lower().strip()
    units = args.get("units", "celsius").lower()
    if city not in WEATHER_DB:
        return json.dumps({"error": f"City '{args.get('city')}' not found. Available: {', '.join(WEATHER_DB.keys())}"})
    data = WEATHER_DB[city].copy()
    if units == "fahrenheit":
        data["temperature"] = round(data["temperature"] * 9 / 5 + 32, 1)
        data["units"] = "fahrenheit"
    else:
        data["units"] = "celsius"
    return json.dumps(data)


# 中文注释：搜索执行器按查询词与候选键的交集大小选择最相关记录。
def _search(args: dict) -> str:
    query = args.get("query", "").lower().strip()
    best_key = None
    best_overlap = 0
    query_words = set(query.split())
    for key in SEARCH_DB:
        key_words = set(key.split())
        overlap = len(query_words & key_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_key = key
    if best_key and best_overlap > 0:
        return json.dumps(SEARCH_DB[best_key])
    return json.dumps([{"text": f"No results found for '{args.get('query')}'.", "score": 0.0}])


# 中文注释：单位执行器检查输入类型和转换方向，再调用转换函数。
def _convert_units(args: dict) -> str:
    value = args.get("value")
    from_unit = args.get("from_unit", "").lower().strip()
    to_unit = args.get("to_unit", "").lower().strip()
    if not isinstance(value, (int, float)):
        return "Error: 'value' must be a number."
    key = (from_unit, to_unit)
    if key not in UNIT_CONVERSIONS:
        available = [f"{f}->{t}" for f, t in UNIT_CONVERSIONS]
        return f"Error: unsupported conversion {from_unit}->{to_unit}. Available: {', '.join(available)}"
    return str(UNIT_CONVERSIONS[key](value))


# 中文注释：数据库执行器先校验表名，再按规范化后的键查找国家记录。
def _lookup_db(args: dict) -> str:
    table = args.get("table", "").lower().strip()
    key = args.get("key", "").lower().strip()
    db_map = {"countries": COUNTRIES_DB}
    if table not in db_map:
        return f"Error: table '{args.get('table')}' not found. Available: {', '.join(db_map.keys())}"
    db = db_map[table]
    if key not in db:
        return f"Not found: '{args.get('key')}' not in {table}."
    return json.dumps(db[key])


# 中文注释：执行器注册表把模型生成的工具名路由到对应 Python 函数。
_EXECUTORS = {
    "calculator": lambda args: _safe_calc(args.get("expression", "")),
    "get_weather": _get_weather,
    "search": _search,
    "convert_units": _convert_units,
    "lookup_db": _lookup_db,
}


# 中文注释：统一入口负责处理未知工具，并把参数交给具体执行器。
def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with the given arguments. Returns a string result."""
    if name not in _EXECUTORS:
        return f"Error: unknown tool '{name}'. Available: {', '.join(_EXECUTORS.keys())}"
    return _EXECUTORS[name](arguments)
