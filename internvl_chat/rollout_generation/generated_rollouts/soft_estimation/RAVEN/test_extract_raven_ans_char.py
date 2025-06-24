import pytest
import sys

sys.path.append('/data/users/brandon/ob1-projects/InternVL/internvl_chat/tools')

from reasoning_data_pipeline.utils.accuracy_reward import extract_raven_choices_answer_from_xml_v2, check_answer

def test_valid_boxed_alphabet():
    """Test extraction of valid boxed alphabet answer"""
    input_text = """Some text
<correct_answer>
$\\boxed{C}$
</correct_answer>
more text"""
    assert extract_raven_choices_answer_from_xml_v2(input_text) == "C"

def test_valid_boxed_alphabet_with_parentheses():
    """Test extraction of valid boxed alphabet answer with parentheses"""
    input_text = """Some text
<correct_answer>
$\\boxed{(C)}$
</correct_answer>
more text"""
    assert extract_raven_choices_answer_from_xml_v2(input_text) == "C"

def test_valid_boxed_alphabet_with_period():
    """Test extraction of valid boxed alphabet answer with period"""
    input_text = """Some text
<correct_answer>
$\\boxed{C.}$
</correct_answer>
more text"""
    assert extract_raven_choices_answer_from_xml_v2(input_text) == "C"

def test_standalone_alphabet_answers():
    """Test extraction of boxed alphabet answers (updated to use boxed format)"""
    test_cases = [
        ("<correct_answer>$\\boxed{A}$</correct_answer>", "A"),
        ("<correct_answer>$\\boxed{B}$</correct_answer>", "B"),
        ("<correct_answer>$\\boxed{C}$</correct_answer>", "C"),
        ("<correct_answer>$\\boxed{D}$</correct_answer>", "D"),
        ("<correct_answer>$\\boxed{E}$</correct_answer>", "E"),
        ("<correct_answer>$\\boxed{F}$</correct_answer>", "F"),
        ("<correct_answer>$\\boxed{G}$</correct_answer>", "G"),
    ]
    
    for input_text, expected in test_cases:
        assert extract_raven_choices_answer_from_xml_v2(input_text) == expected

def test_multiple_boxed_alphabets():
    """Test extraction when multiple boxed alphabet answers exist"""
    input_text = """$\\boxed{A}$
<correct_answer>
$\\boxed{F}$
</correct_answer>"""
    assert extract_raven_choices_answer_from_xml_v2(input_text) == "F"

def test_mixed_format_multiple_answers():
    """Test extraction with mixed formats in multiple answers"""
    input_text = """<correct_answer>
$\\boxed{A}$
</correct_answer>
<correct_answer>
$\\boxed{E}$
</correct_answer>"""
    assert extract_raven_choices_answer_from_xml_v2(input_text) == "E"

def test_no_boxed_alphabet():
    """Test when no boxed alphabet pattern is found (fallback case)"""
    input_text = """<correct_answer>
$\\boxed{plain text answer}$
</correct_answer>"""
    assert extract_raven_choices_answer_from_xml_v2(input_text) == "plain text answer"

def test_missing_tags():
    """Test when XML tags are missing (edge case)"""
    input_text = "$\\boxed{A}$"
    assert extract_raven_choices_answer_from_xml_v2(input_text) == input_text

def test_missing_closing_tag():
    """Test when closing tag is missing (edge case)"""
    input_text = """<correct_answer>
$\\boxed{D}$"""
    assert extract_raven_choices_answer_from_xml_v2(input_text) == input_text

def test_empty_content():
    """Test with empty content between tags"""
    with pytest.raises(AssertionError, match="Empty answer content between XML tags"):
        input_text = """<correct_answer>

</correct_answer>"""
        extract_raven_choices_answer_from_xml_v2(input_text)

def test_mismatched_tags():
    """Test with mismatched number of opening and closing tags"""
    with pytest.raises(AssertionError, match="Mismatched XML tags"):
        input_text = """<correct_answer>
$\\boxed{B.}$
</correct_answer>
</correct_answer>"""
        extract_raven_choices_answer_from_xml_v2(input_text)

def test_all_format_variations():
    """Test all format variations for each letter using boxed format"""
    letters = ["A", "B", "C", "D", "E", "F", "G"]
    formats = [
        "$\\boxed{{{letter}}}$",     # Boxed standalone (primary format)
        "$\\boxed{{({letter})}}$",   # Boxed parentheses
        "$\\boxed{{{letter}.}}$",    # Boxed period
    ]
    
    for letter in letters:
        for format_template in formats:
            formatted_answer = format_template.format(letter=letter)
            input_text = f"""<correct_answer>
{formatted_answer}
</correct_answer>"""
            assert extract_raven_choices_answer_from_xml_v2(input_text) == letter

def test_boxed_alphabet_with_newlines():
    """Test extraction with newlines between tags and content"""
    input_text = """<correct_answer>
$\\boxed{A}$
</correct_answer>"""
    assert extract_raven_choices_answer_from_xml_v2(input_text) == "A"

def test_boxed_alphabet_with_multiple_newlines():
    """Test extraction with multiple newlines and spaces"""
    input_text = """<correct_answer>

    $\\boxed{D}$

</correct_answer>"""
    assert extract_raven_choices_answer_from_xml_v2(input_text) == "D"

def test_boxed_alphabet_with_mixed_format():
    """Test extraction with mixed format (newlines and inline)"""
    input_text = """Some text
<correct_answer>
$\\boxed{C}$
</correct_answer>
More text"""
    assert extract_raven_choices_answer_from_xml_v2(input_text) == "C"

def test_raven_alphabet_score_exact_match():
    """Test exact alphabet match for RAVEN scoring"""
    assert check_answer("A", "A", "raven_score_alphabet_only") == 1
    assert check_answer("B", "B", "raven_score_alphabet_only") == 1
    assert check_answer("C", "C", "raven_score_alphabet_only") == 1
    assert check_answer("D", "D", "raven_score_alphabet_only") == 1
    assert check_answer("E", "E", "raven_score_alphabet_only") == 1
    assert check_answer("F", "F", "raven_score_alphabet_only") == 1
    assert check_answer("G", "G", "raven_score_alphabet_only") == 1

def test_raven_alphabet_score_format_variations():
    """Test scoring across different boxed format variations"""

    # Test different letters still don't match regardless of format
    # First extract, then check
    input_text_a = """<correct_answer>
$\\boxed{A}$
</correct_answer>"""
    pred_a = extract_raven_choices_answer_from_xml_v2(input_text_a)
    gt_b = "B"  # Ground truth is raw single letter
    assert check_answer(pred_a, gt_b, "raven_score_alphabet_only") == 0
    
    input_text_c = """<correct_answer>
$\\boxed{C}$
</correct_answer>"""
    pred_c = extract_raven_choices_answer_from_xml_v2(input_text_c)
    gt_d = "D"  # Ground truth is raw single letter
    assert check_answer(pred_c, gt_d, "raven_score_alphabet_only") == 0
    
    input_text_e = """<correct_answer>
$\\boxed{E}$
</correct_answer>"""
    pred_e = extract_raven_choices_answer_from_xml_v2(input_text_e)
    gt_f = "F"  # Ground truth is raw single letter
    assert check_answer(pred_e, gt_f, "raven_score_alphabet_only") == 0

def test_raven_alphabet_score_mismatch():
    """Test alphabet mismatch for RAVEN scoring"""
    assert check_answer("A", "B", "raven_score_alphabet_only") == 0
    assert check_answer("C", "D", "raven_score_alphabet_only") == 0
    assert check_answer("F", "G", "raven_score_alphabet_only") == 0
    assert check_answer("G", "A", "raven_score_alphabet_only") == 0

def test_raven_alphabet_score_invalid_input():
    """Test invalid input handling for RAVEN scoring"""
    assert check_answer("not a letter", "A", "raven_score_alphabet_only") == 0
    assert check_answer("A", "not a letter", "raven_score_alphabet_only") == 0
    assert check_answer("H", "A", "raven_score_alphabet_only") == 0  # H is outside A-G range
    assert check_answer("(H)", "(A)", "raven_score_alphabet_only") == 0  # H is outside A-G range
    assert check_answer("123", "A.", "raven_score_alphabet_only") == 0

def test_raven_alphabet_score_with_whitespace():
    """Test RAVEN scoring with whitespace handling"""
    assert check_answer(" A ", "A", "raven_score_alphabet_only") == 1
    assert check_answer("B", " B ", "raven_score_alphabet_only") == 1
    assert check_answer(" C ", " C ", "raven_score_alphabet_only") == 1

def test_raven_all_valid_options():
    """Test all valid RAVEN options A through G"""
    valid_options = ["A", "B", "C", "D", "E", "F", "G"]
    for option in valid_options:
        assert check_answer(option, option, "raven_score_alphabet_only") == 1
        # Test that each option doesn't match others
        for other_option in valid_options:
            if option != other_option:
                assert check_answer(option, other_option, "raven_score_alphabet_only") == 0

def test_raven_comprehensive_format_matrix():
    """Test comprehensive matrix of all boxed format combinations"""
    letters = ["A", "B", "C", "D", "E", "F", "G"]
    
    for letter in letters:
        boxed_formats = [
            f"$\\boxed{{{letter}}}$",        # Boxed standalone
            f"$\\boxed{{({letter})}}$",      # Boxed parentheses
            f"$\\boxed{{{letter}.}}$",       # Boxed period
        ]
        
        # Test same letter in different boxed formats should match raw ground truth
        for format1 in boxed_formats:
            # First extract, then check against raw ground truth
            input_text_pred = f"""<correct_answer>
{format1}
</correct_answer>"""
            pred = extract_raven_choices_answer_from_xml_v2(input_text_pred)
            gt = letter  # Ground truth is always raw single letter
            assert check_answer(pred, gt, "raven_score_alphabet_only") == 1
        
        # Test different letters should not match raw ground truth
        for other_letter in letters:
            if letter != other_letter:
                # Test various boxed formats against raw ground truth
                input_text_pred1 = f"""<correct_answer>
$\\boxed{{{letter}}}$
</correct_answer>"""
                pred1 = extract_raven_choices_answer_from_xml_v2(input_text_pred1)
                gt1 = other_letter  # Ground truth is raw single letter
                assert check_answer(pred1, gt1, "raven_score_alphabet_only") == 0
                
                input_text_pred2 = f"""<correct_answer>
$\\boxed{{({letter})}}$
</correct_answer>"""
                pred2 = extract_raven_choices_answer_from_xml_v2(input_text_pred2)
                gt2 = other_letter  # Ground truth is raw single letter
                assert check_answer(pred2, gt2, "raven_score_alphabet_only") == 0
                
                input_text_pred3 = f"""<correct_answer>
$\\boxed{{{letter}.}}$
</correct_answer>"""
                pred3 = extract_raven_choices_answer_from_xml_v2(input_text_pred3)
                gt3 = other_letter  # Ground truth is raw single letter
                assert check_answer(pred3, gt3, "raven_score_alphabet_only") == 0

def test_prompt_template_boxed_format():
    """Test extraction using the exact boxed format from the rollout prompt template"""
    # Test the exact format: $\boxed{Your answer here}$ -> should extract single letter
    test_cases = [
        ("$\\boxed{A}$", "A"),
        ("$\\boxed{B}$", "B"), 
        ("$\\boxed{C}$", "C"),
        ("$\\boxed{D}$", "D"),
        ("$\\boxed{E}$", "E"),
        ("$\\boxed{F}$", "F"),
        ("$\\boxed{G}$", "G"),
    ]
    
    for boxed_answer, expected in test_cases:
        input_text = f"""<correct_answer>
{boxed_answer}
</correct_answer>"""
        assert extract_raven_choices_answer_from_xml_v2(input_text) == expected

def test_prompt_template_format_with_text():
    """Test extraction when the boxed format contains placeholder text"""
    # Test cases where the model might include extra text in the boxed format
    input_text = """<correct_answer>
$\\boxed{The answer is C}$
</correct_answer>"""
    # This should extract "The answer is C" since it doesn't match the single letter pattern
    assert extract_raven_choices_answer_from_xml_v2(input_text) == "C"

def test_malformed_backslash_variations():
    """Test extraction of answers with malformed backslash patterns"""
    # Test {\A} pattern - backslash before letter
    test_cases_before = [
        ("$\\boxed{\\A}$", "A"),
        ("$\\boxed{\\B}$", "B"),
        ("$\\boxed{\\C}$", "C"),
        ("$\\boxed{\\D}$", "D"),
        ("$\\boxed{\\E}$", "E"),
        ("$\\boxed{\\F}$", "F"),
        ("$\\boxed{\\G}$", "G"),
    ]
    
    for boxed_answer, expected in test_cases_before:
        input_text = f"""<correct_answer>
{boxed_answer}
</correct_answer>"""
        assert extract_raven_choices_answer_from_xml_v2(input_text) == expected

    # Test {A\} pattern - backslash after letter
    test_cases_after = [
        ("$\\boxed{A\\}$", "A"),
        ("$\\boxed{B\\}$", "B"),
        ("$\\boxed{C\\}$", "C"),
        ("$\\boxed{D\\}$", "D"),
        ("$\\boxed{E\\}$", "E"),
        ("$\\boxed{F\\}$", "F"),
        ("$\\boxed{G\\}$", "G"),
    ]
    
    for boxed_answer, expected in test_cases_after:
        input_text = f"""<correct_answer>
{boxed_answer}
</correct_answer>"""
        assert extract_raven_choices_answer_from_xml_v2(input_text) == expected

    # Test {\A\} pattern - backslashes before and after letter
    test_cases_both = [
        ("$\\boxed{\\A\\}$", "A"),
        ("$\\boxed{\\B\\}$", "B"),
        ("$\\boxed{\\C\\}$", "C"),
        ("$\\boxed{\\D\\}$", "D"),
        ("$\\boxed{\\E\\}$", "E"),
        ("$\\boxed{\\F\\}$", "F"),
        ("$\\boxed{\\G\\}$", "G"),
    ]
    
    for boxed_answer, expected in test_cases_both:
        input_text = f"""<correct_answer>
{boxed_answer}
</correct_answer>"""
        assert extract_raven_choices_answer_from_xml_v2(input_text) == expected

def test_malformed_backslash_with_parentheses():
    """Test malformed backslash patterns combined with parentheses"""
    test_cases = [
        ("$\\boxed{(\\A)}$", "A"),
        ("$\\boxed{(B\\)}$", "B"),
        ("$\\boxed{(\\C\\)}$", "C"),
        ("$\\boxed{\\(D)}$", "D"),
        ("$\\boxed{(E)\\}$", "E"),
    ]
    
    for boxed_answer, expected in test_cases:
        input_text = f"""<correct_answer>
{boxed_answer}
</correct_answer>"""
        assert extract_raven_choices_answer_from_xml_v2(input_text) == expected

def test_malformed_backslash_with_periods():
    """Test malformed backslash patterns combined with periods"""
    test_cases = [
        ("$\\boxed{\\A.}$", "A"),
        ("$\\boxed{B\\.}$", "B"),
        ("$\\boxed{\\C\\.}$", "C"),
        ("$\\boxed{\\.D}$", "D"),
        ("$\\boxed{E.\\}$", "E"),
    ]
    
    for boxed_answer, expected in test_cases:
        input_text = f"""<correct_answer>
{boxed_answer}
</correct_answer>"""
        assert extract_raven_choices_answer_from_xml_v2(input_text) == expected

def test_mixed_malformed_patterns():
    """Test mixed malformed patterns in the same input"""
    input_text = """Some initial text
<correct_answer>
$\\boxed{\\A}$
</correct_answer>
<correct_answer>
$\\boxed{B\\}$
</correct_answer>
<correct_answer>
$\\boxed{\\C\\}$
</correct_answer>"""
    # Should extract the last valid pattern
    assert extract_raven_choices_answer_from_xml_v2(input_text) == "C"

def test_malformed_backslash_scoring():
    """Test scoring with malformed backslash patterns"""
    # Test that malformed patterns still score correctly after extraction
    malformed_inputs = [
        "$\\boxed{\\A}$",
        "$\\boxed{B\\}$", 
        "$\\boxed{\\C\\}$",
        "$\\boxed{(\\D)}$",
        "$\\boxed{E\\.}$"
    ]
    
    expected_letters = ["A", "B", "C", "D", "E"]
    
    for malformed_input, expected_letter in zip(malformed_inputs, expected_letters):
        input_text = f"""<correct_answer>
{malformed_input}
</correct_answer>"""
        pred = extract_raven_choices_answer_from_xml_v2(input_text)
        # Should match the expected letter
        assert check_answer(pred, expected_letter, "raven_score_alphabet_only") == 1
        # Should not match other letters
        for other_letter in ["A", "B", "C", "D", "E", "F", "G"]:
            if other_letter != expected_letter:
                assert check_answer(pred, other_letter, "raven_score_alphabet_only") == 0
