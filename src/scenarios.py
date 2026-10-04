"""Multi-step tool-use scenario generator.

Each scenario defines:
- A natural language task
- The expected sequence of tool calls (name + arguments)
- Used for both training rollouts and reward computation

Scenarios are parameterized templates with deterministic seeding.
"""
# 中文注释：场景生成只依赖本地随机数生成器，因此给定 seed 后结果完全确定。
import random

# 中文注释：每个模板描述自然语言任务、工具序列和如何从参数构造期望调用。
SCENARIO_TEMPLATES = [
    # 2-step: search + calculator
    {
        "id": "age_calculation",
        "task": "How old was {person} in {year}?",
        "params": lambda rng: {
            "person": rng.choice(["Einstein", "Guido van Rossum"]),
            "year": rng.choice([2020, 2024]),
            "_birth": {"Einstein": 1879, "Guido van Rossum": 1956},
        },
        "expected_tools": ["search", "calculator"],
        "expected_calls": [
            {"name": "search", "arguments": {"query": "__template__"}},
            {"name": "calculator", "arguments": {"expression": "__template__"}},
        ],
        "build_calls": lambda p: [
            {"name": "search", "arguments": {"query": f"{p['person']} birth year"}},
            {"name": "calculator", "arguments": {"expression": f"{p['year']}-{p['_birth'][p['person']]}"}},
        ],
    },
    # 2-step: weather + convert
    {
        "id": "weather_convert",
        "task": "What is the temperature in {city} in Fahrenheit?",
        "params": lambda rng: {"city": rng.choice(["Paris", "Tokyo", "London", "Berlin", "Mumbai"])},
        "expected_tools": ["get_weather", "convert_units"],
        "expected_calls": [
            {"name": "get_weather", "arguments": {"city": "__template__"}},
            {"name": "convert_units", "arguments": {"value": "__dynamic__", "from_unit": "celsius", "to_unit": "fahrenheit"}},
        ],
        "build_calls": lambda p: [
            {"name": "get_weather", "arguments": {"city": p["city"]}},
            {"name": "convert_units", "arguments": {"value": "__dynamic__", "from_unit": "celsius", "to_unit": "fahrenheit"}},
        ],
    },
    # 2-step: lookup + calculator
    {
        "id": "population_math",
        "task": "What is the population of {country} divided by its area in km2?",
        "params": lambda rng: {"country": rng.choice(["France", "Japan", "Brazil", "Germany", "India"])},
        "expected_tools": ["lookup_db", "calculator"],
        "expected_calls": [
            {"name": "lookup_db", "arguments": {"table": "countries", "key": "__template__"}},
            {"name": "calculator", "arguments": {"expression": "__dynamic__"}},
        ],
        "build_calls": lambda p: [
            {"name": "lookup_db", "arguments": {"table": "countries", "key": p["country"]}},
            {"name": "calculator", "arguments": {"expression": "__dynamic__"}},
        ],
    },
    # 1-step: calculator only
    {
        "id": "simple_math",
        "task": "What is {a} {op_word} {b}?",
        "params": lambda rng: (lambda op: {
            "a": rng.randint(10, 999),
            "b": rng.randint(2, 99),
            "op": op,
            "op_word": {"+": "plus", "-": "minus", "*": "times"}[op],
        })(rng.choice(["+", "-", "*"])),
        "expected_tools": ["calculator"],
        "expected_calls": [
            {"name": "calculator", "arguments": {"expression": "__template__"}},
        ],
        "build_calls": lambda p: [
            {"name": "calculator", "arguments": {"expression": f"{p['a']}{p['op']}{p['b']}"}},
        ],
    },
    # 1-step: weather only
    {
        "id": "simple_weather",
        "task": "What's the weather like in {city}?",
        "params": lambda rng: {"city": rng.choice(["Paris", "Tokyo", "London", "Sydney", "Cairo", "Seattle", "Dubai"])},
        "expected_tools": ["get_weather"],
        "expected_calls": [
            {"name": "get_weather", "arguments": {"city": "__template__"}},
        ],
        "build_calls": lambda p: [
            {"name": "get_weather", "arguments": {"city": p["city"]}},
        ],
    },
    # 1-step: search only
    {
        "id": "simple_search",
        "task": "What is {topic}?",
        "params": lambda rng: {"topic": rng.choice([
            "the speed of light", "the boiling point of water", "the value of pi",
            "the tallest mountain", "the population of France",
        ])},
        "expected_tools": ["search"],
        "expected_calls": [
            {"name": "search", "arguments": {"query": "__template__"}},
        ],
        "build_calls": lambda p: [
            {"name": "search", "arguments": {"query": p["topic"]}},
        ],
    },
    # 2-step: search + convert
    {
        "id": "search_and_convert",
        "task": "What is {topic} in {target_unit}?",
        "params": lambda rng: {
            "topic": "the distance from Earth to the Sun in km",
            "target_unit": "miles",
        },
        "expected_tools": ["search", "convert_units"],
        "expected_calls": [
            {"name": "search", "arguments": {"query": "earth distance sun"}},
            {"name": "convert_units", "arguments": {"value": 149600000, "from_unit": "km", "to_unit": "miles"}},
        ],
        "build_calls": lambda p: [
            {"name": "search", "arguments": {"query": "earth distance sun"}},
            {"name": "convert_units", "arguments": {"value": 149600000, "from_unit": "km", "to_unit": "miles"}},
        ],
    },
    # 3-step: lookup + calculator + convert
    {
        "id": "country_density_convert",
        "task": "What is {country}'s population density in people per square mile?",
        "params": lambda rng: {"country": rng.choice(["France", "Japan", "Germany", "India"])},
        "expected_tools": ["lookup_db", "calculator", "convert_units"],
        "expected_calls": [
            {"name": "lookup_db", "arguments": {"table": "countries", "key": "__template__"}},
            {"name": "calculator", "arguments": {"expression": "__dynamic__"}},
            {"name": "convert_units", "arguments": {"value": "__dynamic__", "from_unit": "km", "to_unit": "miles"}},
        ],
        "build_calls": lambda p: [
            {"name": "lookup_db", "arguments": {"table": "countries", "key": p["country"]}},
            {"name": "calculator", "arguments": {"expression": "__dynamic__"}},
            {"name": "convert_units", "arguments": {"value": "__dynamic__", "from_unit": "km", "to_unit": "miles"}},
        ],
    },
    # 2-step: weather comparison
    {
        "id": "weather_compare",
        "task": "Which is hotter right now, {city_a} or {city_b}?",
        "params": lambda rng: {
            "city_a": rng.choice(["Paris", "Tokyo", "London"]),
            "city_b": rng.choice(["Cairo", "Mumbai", "Dubai"]),
        },
        "expected_tools": ["get_weather", "get_weather"],
        "expected_calls": [
            {"name": "get_weather", "arguments": {"city": "__template_a__"}},
            {"name": "get_weather", "arguments": {"city": "__template_b__"}},
        ],
        "build_calls": lambda p: [
            {"name": "get_weather", "arguments": {"city": p["city_a"]}},
            {"name": "get_weather", "arguments": {"city": p["city_b"]}},
        ],
    },
    # 1-step: unit conversion only
    {
        "id": "simple_convert",
        "task": "Convert {value} {from_u} to {to_u}.",
        "params": lambda rng: {"value": rng.randint(1, 100), "from_u": "kg", "to_u": "lbs"},
        "expected_tools": ["convert_units"],
        "expected_calls": [
            {"name": "convert_units", "arguments": {"value": "__template__", "from_unit": "kg", "to_unit": "lbs"}},
        ],
        "build_calls": lambda p: [
            {"name": "convert_units", "arguments": {"value": p["value"], "from_unit": p["from_u"], "to_unit": p["to_u"]}},
        ],
    },
    # 1-step: db lookup only
    {
        "id": "simple_lookup",
        "task": "What is the GDP of {country}?",
        "params": lambda rng: {"country": rng.choice(["France", "Japan", "Brazil", "Germany", "India"])},
        "expected_tools": ["lookup_db"],
        "expected_calls": [
            {"name": "lookup_db", "arguments": {"table": "countries", "key": "__template__"}},
        ],
        "build_calls": lambda p: [
            {"name": "lookup_db", "arguments": {"table": "countries", "key": p["country"]}},
        ],
    },
    # 2-step: compare two countries
    {
        "id": "country_compare",
        "task": "Which country has a larger population, {country_a} or {country_b}?",
        "params": lambda rng: {
            "country_a": rng.choice(["France", "Japan"]),
            "country_b": rng.choice(["Brazil", "India"]),
        },
        "expected_tools": ["lookup_db", "lookup_db"],
        "expected_calls": [
            {"name": "lookup_db", "arguments": {"table": "countries", "key": "__template_a__"}},
            {"name": "lookup_db", "arguments": {"table": "countries", "key": "__template_b__"}},
        ],
        "build_calls": lambda p: [
            {"name": "lookup_db", "arguments": {"table": "countries", "key": p["country_a"]}},
            {"name": "lookup_db", "arguments": {"table": "countries", "key": p["country_b"]}},
        ],
    },
]


# 中文注释：该函数从模板中生成一个可用于训练或奖励计算的完整场景。
def generate_scenario(seed: int) -> dict:
    """Generate a single scenario from templates using the given seed."""
    rng = random.Random(seed)
    template = rng.choice(SCENARIO_TEMPLATES)
    params = template["params"](rng)

    public_params = {k: v for k, v in params.items() if not k.startswith("_") and not callable(v)}
    task = template["task"].format(**public_params)
    expected_calls = template["build_calls"](params)

    return {
        "id": template["id"],
        "task": task,
        "expected_tools": template["expected_tools"],
        "expected_calls": expected_calls,
    }


# 中文注释：通过连续 seed 批量生成数据集，训练集和验证集可使用不同起点。
def generate_dataset(count: int, seed: int = 0) -> list[dict]:
    """Generate a list of scenarios."""
    return [generate_scenario(seed=seed + i) for i in range(count)]
