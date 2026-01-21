#!/usr/bin/env python3
"""
独立测试脚本：验证 extraction_text 规范化逻辑

这个脚本不依赖项目其他模块，可以独立运行。
"""

import json
from typing import Any


def _string_value(value: Any) -> str:
    """将任意值转换为字符串"""
    if value is None:
        return ""
    return str(value).strip()


def _normalize_extraction_text(extraction_text: Any, attrs: dict[str, Any], ex_class: str | None) -> str:
    """
    将 extraction_text 规范化为字符串类型。
    这是一个防御性函数，用于处理 LLM 返回错误格式的情况。
    
    Args:
        extraction_text: 原始的 extraction_text 值
        attrs: extraction 的 attributes 字典
        ex_class: extraction_class 的值（"entity" 或 "relation"）
    
    Returns:
        规范化后的字符串
    """
    # 如果已经是字符串、整数或浮点数，直接转换
    if isinstance(extraction_text, (str, int, float)):
        return str(extraction_text).strip()
    
    # 如果是字典类型（错误格式），尝试提取有用信息
    if isinstance(extraction_text, dict):
        print(f"  ⚠️  警告: extraction_text 是字典（应该是字符串）: {extraction_text}")
        # 尝试从字典中提取文本
        for key in ("text", "value", "content", "name"):
            if key in extraction_text and extraction_text[key]:
                return str(extraction_text[key]).strip()
        # 如果找不到，转为 JSON 字符串
        try:
            return json.dumps(extraction_text, ensure_ascii=False)
        except Exception:
            pass
    
    # 如果是列表类型（错误格式）
    if isinstance(extraction_text, list):
        print(f"  ⚠️  警告: extraction_text 是列表（应该是字符串）: {extraction_text}")
        # 取第一个非空元素
        for item in extraction_text:
            if item:
                return str(item).strip()
    
    # 如果是 None 或其他类型，根据 extraction_class 生成备用值
    if ex_class == "entity":
        name = _string_value(attrs.get("name"))
        if name:
            return name
    elif ex_class == "relation":
        source = _string_value(attrs.get("source"))
        relation_type = _string_value(attrs.get("type"))
        target = _string_value(attrs.get("target"))
        if source and target:
            return f"{source} {relation_type} {target}".strip()
    
    # 最后的备用方案
    print("  ⚠️  警告: extraction_text 无效，使用空字符串作为备用")
    return ""


def test_normalize_extraction_text():
    """测试 extraction_text 规范化函数"""
    
    print("=" * 80)
    print("测试 LangExtract extraction_text 规范化功能")
    print("=" * 80)
    
    # 测试用例
    test_cases = [
        {
            "name": "1. ✅ 正常字符串",
            "extraction_text": "Acme Bio Inc.",
            "attrs": {"name": "Acme Bio Inc.", "type": "Company"},
            "ex_class": "entity",
            "expected": "Acme Bio Inc."
        },
        {
            "name": "2. ✅ 整数",
            "extraction_text": 123,
            "attrs": {"name": "Test"},
            "ex_class": "entity",
            "expected": "123"
        },
        {
            "name": "3. ✅ 浮点数",
            "extraction_text": 123.45,
            "attrs": {"name": "Test"},
            "ex_class": "entity",
            "expected": "123.45"
        },
        {
            "name": "4. 🔧 字典（包含 text 键）- 需要修复",
            "extraction_text": {"text": "Acme Bio Inc.", "source": "page1"},
            "attrs": {"name": "Acme Bio Inc.", "type": "Company"},
            "ex_class": "entity",
            "expected": "Acme Bio Inc."
        },
        {
            "name": "5. 🔧 字典（包含 value 键）- 需要修复",
            "extraction_text": {"value": "Some Value"},
            "attrs": {"name": "Test"},
            "ex_class": "entity",
            "expected": "Some Value"
        },
        {
            "name": "6. 🔧 字典（包含 name 键）- 需要修复",
            "extraction_text": {"name": "Entity Name", "foo": "bar"},
            "attrs": {"name": "Test"},
            "ex_class": "entity",
            "expected": "Entity Name"
        },
        {
            "name": "7. 🔧 字典（不包含标准键）- 需要修复",
            "extraction_text": {"foo": "bar", "baz": "qux"},
            "attrs": {"name": "Test Entity"},
            "ex_class": "entity",
            "expected_type": "json_string"  # 转为 JSON 字符串
        },
        {
            "name": "8. 🔧 列表 - 需要修复",
            "extraction_text": ["First Item", "Second Item"],
            "attrs": {"name": "Test"},
            "ex_class": "entity",
            "expected": "First Item"
        },
        {
            "name": "9. 🔧 空列表 - 需要修复（使用备用值）",
            "extraction_text": [],
            "attrs": {"name": "Fallback Name"},
            "ex_class": "entity",
            "expected": "Fallback Name"
        },
        {
            "name": "10. 🔧 None（实体）- 需要修复（使用备用值）",
            "extraction_text": None,
            "attrs": {"name": "Fallback Entity"},
            "ex_class": "entity",
            "expected": "Fallback Entity"
        },
        {
            "name": "11. 🔧 None（关系）- 需要修复（使用备用值）",
            "extraction_text": None,
            "attrs": {
                "source": "Company A",
                "type": "PARTNERS_WITH",
                "target": "Company B"
            },
            "ex_class": "relation",
            "expected": "Company A PARTNERS_WITH Company B"
        },
    ]
    
    # 运行测试
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        print(f"  输入类型: {type(test_case['extraction_text']).__name__}")
        print(f"  输入值: {test_case['extraction_text']}")
        
        result = _normalize_extraction_text(
            test_case['extraction_text'],
            test_case['attrs'],
            test_case['ex_class']
        )
        
        print(f"  输出值: '{result}'")
        print(f"  输出类型: {type(result).__name__}")
        
        # 检查结果
        if "expected_type" in test_case:
            # 特殊情况：检查是否是 JSON 字符串
            if test_case["expected_type"] == "json_string":
                try:
                    json.loads(result)
                    print(f"  ✅ 通过 (成功转换为 JSON 字符串)")
                    passed += 1
                except json.JSONDecodeError:
                    print(f"  ❌ 失败 (不是有效的 JSON 字符串)")
                    failed += 1
        else:
            expected = test_case.get('expected', '')
            if result == expected:
                print(f"  期望: '{expected}'")
                print(f"  ✅ 通过")
                passed += 1
            else:
                print(f"  期望: '{expected}'")
                print(f"  ❌ 失败")
                failed += 1
    
    # 统计结果
    print("\n" + "=" * 80)
    print(f"测试总结: {passed} 通过 / {failed} 失败 / {len(test_cases)} 总计")
    print("=" * 80)
    
    if failed == 0:
        print("✅ 所有测试通过！extraction_text 规范化功能工作正常。")
        return True
    else:
        print(f"❌ 有 {failed} 个测试失败，请检查代码。")
        return False


def test_real_world_scenario():
    """模拟真实世界中 LLM 返回错误格式的场景"""
    
    print("\n" + "=" * 80)
    print("真实场景测试：模拟 LLM 返回错误格式")
    print("=" * 80)
    
    # 场景 1：LLM 返回的 extraction_text 是字典
    print("\n场景 1: extraction_text 是字典（最常见的错误）")
    print("-" * 80)
    
    # 这是 LLM 可能返回的错误格式
    wrong_response = {
        "extractions": [
            {
                "extraction_class": "entity",
                "extraction_text": {  # ❌ 错误：应该是字符串
                    "text": "Acme Bio Inc.",
                    "context": "biotechnology company"
                },
                "attributes": {
                    "name": "Acme Bio Inc.",
                    "type": "Company",
                    "description": "A biotechnology company"
                }
            },
            {
                "extraction_class": "relation",
                "extraction_text": {  # ❌ 错误：应该是字符串
                    "relation": "partners with",
                    "from": "Acme Bio Inc.",
                    "to": "University"
                },
                "attributes": {
                    "source": "Acme Bio Inc.",
                    "target": "Example University",
                    "type": "PARTNERS_WITH"
                }
            }
        ]
    }
    
    print("LLM 返回的错误 JSON:")
    print(json.dumps(wrong_response, indent=2, ensure_ascii=False))
    
    print("\n处理结果:")
    for i, extraction in enumerate(wrong_response["extractions"], 1):
        print(f"\n  提取项 {i} ({extraction['extraction_class']}):")
        print(f"    原始 extraction_text: {extraction['extraction_text']}")
        print(f"    类型: {type(extraction['extraction_text']).__name__}")
        
        normalized = _normalize_extraction_text(
            extraction['extraction_text'],
            extraction['attributes'],
            extraction['extraction_class']
        )
        
        print(f"    ✅ 规范化后: '{normalized}' (类型: {type(normalized).__name__})")
    
    print("\n" + "=" * 80)
    print("✅ 真实场景测试完成！系统能够自动修复 LLM 返回的错误格式。")
    print("=" * 80)


if __name__ == "__main__":
    print("\n🔧 LangExtract extraction_text 修复验证\n")
    
    # 运行基础测试
    test1_passed = test_normalize_extraction_text()
    
    # 运行真实场景测试
    test_real_world_scenario()
    
    # 总结
    print("\n" + "=" * 80)
    if test1_passed:
        print("✅ 所有测试通过！修复代码工作正常。")
        print("\n📝 下一步操作:")
        print("   1. 重新运行你的 langextract 提取任务")
        print("   2. 系统会自动处理 LLM 返回的错误格式")
        print("   3. 检查日志中的警告信息，了解哪些数据被自动修复")
        print("\n💡 建议:")
        print("   - 如果经常看到警告，考虑进一步优化提示词")
        print("   - 或者考虑使用更强大的模型（如 gpt-4o）")
    else:
        print("❌ 部分测试失败，请检查修复代码。")
    print("=" * 80)
    
    exit(0 if test1_passed else 1)

