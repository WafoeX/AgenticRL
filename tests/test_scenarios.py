import json
import pytest
from src.scenarios import SCENARIO_TEMPLATES, generate_scenario, generate_dataset


# 中文注释：检查模板数量以及每个模板的必需字段。
class TestScenarioTemplates:
    def test_templates_exist(self):
        assert len(SCENARIO_TEMPLATES) >= 10

    def test_each_template_has_required_fields(self):
        for t in SCENARIO_TEMPLATES:
            assert "task" in t, f"Missing 'task' in template: {t}"
            assert "expected_tools" in t, f"Missing 'expected_tools' in template: {t}"
            assert "expected_calls" in t, f"Missing 'expected_calls' in template: {t}"
            assert len(t["expected_tools"]) >= 1
            assert len(t["expected_calls"]) >= 1


# 中文注释：检查单场景结构、随机种子确定性和不同种子的差异。
class TestGenerateScenario:
    def test_returns_dict(self):
        scenario = generate_scenario(seed=42)
        assert isinstance(scenario, dict)
        assert "task" in scenario
        assert "expected_calls" in scenario

    def test_deterministic_with_seed(self):
        s1 = generate_scenario(seed=42)
        s2 = generate_scenario(seed=42)
        assert s1 == s2

    def test_different_seeds_different_scenarios(self):
        s1 = generate_scenario(seed=1)
        s2 = generate_scenario(seed=2)
        assert s1["task"] != s2["task"] or s1 != s2


# 中文注释：检查批量生成数量和每条数据的基本有效性。
class TestGenerateDataset:
    def test_generates_correct_count(self):
        dataset = generate_dataset(count=5, seed=0)
        assert len(dataset) == 5

    def test_all_entries_valid(self):
        dataset = generate_dataset(count=10, seed=0)
        for entry in dataset:
            assert "task" in entry
            assert "expected_calls" in entry
            assert len(entry["expected_calls"]) >= 1
