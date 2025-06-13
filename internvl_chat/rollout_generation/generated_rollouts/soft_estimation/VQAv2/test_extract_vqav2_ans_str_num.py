import pytest
import sys

sys.path.append('/data/users/brandon/ob1-projects/InternVL/internvl_chat/tools')

from reasoning_data_pipeline.utils.accuracy_reward import extract_dvqa_answer_int_from_xml, check_answer

def test_valid_boxed_integer():
    """Test extraction of valid boxed integer answer"""
    input_text = """Some text
<correct_answer>
$\\boxed{42}$
</correct_answer>
more text"""
    assert extract_dvqa_answer_int_from_xml(input_text) == "42"

def test_multiple_boxed_integers():
    """Test extraction when multiple boxed integers exist"""
    input_text = """$\\boxed{123}$
<correct_answer>
$\\boxed{456}$
</correct_answer>"""
    assert extract_dvqa_answer_int_from_xml(input_text) == "456"

def test_no_boxed_integer():
    """Test when no boxed integer pattern is found"""
    input_text = """<correct_answer>
plain text answer
</correct_answer>"""
    assert extract_dvqa_answer_int_from_xml(input_text) == "plain text answer"

def test_missing_tags():
    """Test when XML tags are missing"""
    input_text = "Just some text without tags"
    assert extract_dvqa_answer_int_from_xml(input_text) == input_text

def test_missing_closing_tag():
    """Test when closing tag is missing"""
    input_text = """<correct_answer>
$\\boxed{42}$"""
    assert extract_dvqa_answer_int_from_xml(input_text) == input_text

def test_empty_content():
    """Test with empty content between tags"""
    with pytest.raises(AssertionError, match="Empty answer content between XML tags"):
        input_text = """<correct_answer>

</correct_answer>"""
        extract_dvqa_answer_int_from_xml(input_text)

def test_mismatched_tags():
    """Test with mismatched number of opening and closing tags"""
    with pytest.raises(AssertionError, match="Mismatched XML tags"):
        input_text = """<correct_answer>
$\\boxed{42}$
</correct_answer>
</correct_answer>"""
        extract_dvqa_answer_int_from_xml(input_text)

def test_complex_boxed_expression():
    """Test with complex boxed expression"""
    input_text = """<correct_answer>
$\\boxed{12345}$
</correct_answer>"""
    assert extract_dvqa_answer_int_from_xml(input_text) == "12345"

def test_multiple_correct_answers():
    """Test with multiple correct_answer tags"""
    input_text = """<correct_answer>
first
</correct_answer>
<correct_answer>
$\\boxed{789}$
</correct_answer>"""
    assert extract_dvqa_answer_int_from_xml(input_text) == "789"

def test_boxed_integer_with_newlines():
    """Test extraction with newlines between tags and content"""
    input_text = """<correct_answer>
$\\boxed{1}$
</correct_answer>"""
    assert extract_dvqa_answer_int_from_xml(input_text) == "1"

def test_boxed_integer_with_multiple_newlines():
    """Test extraction with multiple newlines and spaces"""
    input_text = """<correct_answer>

    $\\boxed{42}$

</correct_answer>"""
    assert extract_dvqa_answer_int_from_xml(input_text) == "42"

def test_boxed_integer_with_mixed_format():
    """Test extraction with mixed format (newlines and inline)"""
    input_text = """Some text
<correct_answer>
$\\boxed{123}$
</correct_answer>
More text"""
    assert extract_dvqa_answer_int_from_xml(input_text) == "123"

def test_dvqa_int_only_score_exact_match():
    """Test exact integer match for DVQA scoring"""
    assert check_answer("42", "42", "vqav2_num_str_only_score") == 1
    assert check_answer("0", "0", "vqav2_num_str_only_score") == 1
    assert check_answer("-1", "-1", "vqav2_num_str_only_score") == 1

def test_dvqa_int_only_score_mismatch():
    """Test integer mismatch for DVQA scoring"""
    assert check_answer("42", "43", "vqav2_num_str_only_score") == 0
    assert check_answer("0", "1", "vqav2_num_str_only_score") == 0
    assert check_answer("-1", "1", "vqav2_num_str_only_score") == 0

def test_dvqa_int_only_score_invalid_input():
    """Test invalid input handling for DVQA scoring"""
    assert check_answer("not a number", "42", "vqav2_num_str_only_score") == 0
    assert check_answer("42", "not a number", "vqav2_num_str_only_score") == 0
    assert check_answer("abc", "def", "vqav2_num_str_only_score") == 0

def test_dvqa_int_only_score_with_whitespace():
    """Test DVQA scoring with whitespace handling"""
    input_text = """<correct_answer>
 $\\boxed{42}$
</correct_answer>"""
    extracted = extract_dvqa_answer_int_from_xml(input_text)
    assert check_answer(extracted, "42", "vqav2_num_str_only_score") == 1

    input_text = """<correct_answer>
$\\boxed{42}$
</correct_answer>"""
    extracted = extract_dvqa_answer_int_from_xml(input_text)
    assert check_answer(extracted, "42", "vqav2_num_str_only_score") == 1

    input_text = """<correct_answer>
 $\\boxed{42}$
</correct_answer>"""
    extracted = extract_dvqa_answer_int_from_xml(input_text)
    assert check_answer(extracted, "42", "vqav2_num_str_only_score") == 1


def test_dvqa_int_only_score_with_leading_zeros():
    """Test DVQA scoring with leading zeros handling"""
    input_text = """<correct_answer>
 $\\boxed{0042}$
</correct_answer>"""
    extracted = extract_dvqa_answer_int_from_xml(input_text)
    assert check_answer(extracted, "0042", "vqav2_num_str_only_score") == 1

    input_text = """<correct_answer>
$\\boxed{42000}$
</correct_answer>"""
    extracted = extract_dvqa_answer_int_from_xml(input_text)
    assert check_answer(extracted, "42000", "vqav2_num_str_only_score") == 1

    input_text = """<correct_answer>
 $\\boxed{004200}$
</correct_answer>"""
    extracted = extract_dvqa_answer_int_from_xml(input_text)
    assert check_answer(extracted, "004200", "vqav2_num_str_only_score") == 1

def test_dvqa_int_only_score_negative_integers():
    """Test DVQA scoring with negative integers"""
    assert check_answer("-42", "-42", "vqav2_num_str_only_score") == 1
    assert check_answer("-0", "-0", "vqav2_num_str_only_score") == 1
    assert check_answer("-1", "-1", "vqav2_num_str_only_score") == 1

def test_dvqa_int_only_score_random_integers():
    """Test DVQA scoring with random integer strings"""
    assert check_answer("02939", "02939", "vqav2_num_str_only_score") == 1
    assert check_answer("00042", "00042", "vqav2_num_str_only_score") == 1
    assert check_answer("123456789", "123456789", "vqav2_num_str_only_score") == 1
    assert check_answer("02939", "2939", "vqav2_num_str_only_score") == 0  # Leading zeros matter
    assert check_answer("00042", "42", "vqav2_num_str_only_score") == 0  # Leading zeros matter

def test_boxed_integer_with_leading_zeros():
    """Test extraction of boxed integer with leading zeros"""
    input_text = """<correct_answer>
$\\boxed{09123}$
</correct_answer>"""
    assert extract_dvqa_answer_int_from_xml(input_text) == "09123"

def test_boxed_integer_with_trailing_zeros():
    """Test extraction of boxed integer with trailing zeros"""
    input_text = """<correct_answer>
$\\boxed{09123000}$
</correct_answer>"""
    assert extract_dvqa_answer_int_from_xml(input_text) == "09123000"