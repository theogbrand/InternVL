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
    """Test extraction of standalone alphabet answers without boxing"""
    test_cases = [
        ("<correct_answer>A</correct_answer>", "A"),
        ("<correct_answer>(B)</correct_answer>", "B"),
        ("<correct_answer>C.</correct_answer>", "C"),
        ("<correct_answer>D</correct_answer>", "D"),
        ("<correct_answer>(E)</correct_answer>", "E"),
        ("<correct_answer>F.</correct_answer>", "F"),
        ("<correct_answer>G</correct_answer>", "G"),
    ]
    
    for input_text, expected in test_cases:
        assert extract_raven_choices_answer_from_xml_v2(input_text) == expected

def test_multiple_boxed_alphabets():
    """Test extraction when multiple boxed alphabet answers exist"""
    input_text = """$\\boxed{A}$
<correct_answer>
$\\boxed{(F)}$
</correct_answer>"""
    assert extract_raven_choices_answer_from_xml_v2(input_text) == "F"

def test_mixed_format_multiple_answers():
    """Test extraction with mixed formats in multiple answers"""
    input_text = """<correct_answer>
A.
</correct_answer>
<correct_answer>
$\\boxed{(E)}$
</correct_answer>"""
    assert extract_raven_choices_answer_from_xml_v2(input_text) == "E"

def test_no_boxed_alphabet():
    """Test when no boxed alphabet pattern is found"""
    input_text = """<correct_answer>
plain text answer
</correct_answer>"""
    assert extract_raven_choices_answer_from_xml_v2(input_text) == "plain text answer"

def test_missing_tags():
    """Test when XML tags are missing"""
    input_text = "Just some text without tags"
    assert extract_raven_choices_answer_from_xml_v2(input_text) == input_text

def test_missing_closing_tag():
    """Test when closing tag is missing"""
    input_text = """<correct_answer>
$\\boxed{(D)}$"""
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
    """Test all format variations for each letter"""
    letters = ["A", "B", "C", "D", "E", "F", "G"]
    formats = [
        "{letter}",           # Standalone
        "({letter})",         # Parentheses
        "{letter}.",          # Period
        "$\\boxed{{{letter}}}$",     # Boxed standalone
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
$\\boxed{(A)}$
</correct_answer>"""
    assert extract_raven_choices_answer_from_xml_v2(input_text) == "A"

def test_boxed_alphabet_with_multiple_newlines():
    """Test extraction with multiple newlines and spaces"""
    input_text = """<correct_answer>

    $\\boxed{D.}$

</correct_answer>"""
    assert extract_raven_choices_answer_from_xml_v2(input_text) == "D"

def test_boxed_alphabet_with_mixed_format():
    """Test extraction with mixed format (newlines and inline)"""
    input_text = """Some text
<correct_answer>
$\\boxed{(C)}$
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
    """Test scoring across different format variations"""

    # Test different letters still don't match regardless of format
    # First extract, then check
    input_text_a = """<correct_answer>
A
</correct_answer>"""
    pred_a = extract_raven_choices_answer_from_xml_v2(input_text_a)
    gt_b = "B"  # Ground truth is raw single letter
    assert check_answer(pred_a, gt_b, "raven_score_alphabet_only") == 0
    
    input_text_c = """<correct_answer>
(C)
</correct_answer>"""
    pred_c = extract_raven_choices_answer_from_xml_v2(input_text_c)
    gt_d = "D"  # Ground truth is raw single letter
    assert check_answer(pred_c, gt_d, "raven_score_alphabet_only") == 0
    
    input_text_e = """<correct_answer>
E.
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
    """Test comprehensive matrix of all format combinations"""
    letters = ["A", "B", "C", "D", "E", "F", "G"]
    
    for letter in letters:
        base_formats = [
            letter,                    # A
            f"({letter})",            # (A)
            f"{letter}.",             # A.
        ]
        
        # Test same letter in different formats should match raw ground truth
        for format1 in base_formats:
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
                # Test various formats against raw ground truth
                input_text_pred1 = f"""<correct_answer>
{letter}
</correct_answer>"""
                pred1 = extract_raven_choices_answer_from_xml_v2(input_text_pred1)
                gt1 = other_letter  # Ground truth is raw single letter
                assert check_answer(pred1, gt1, "raven_score_alphabet_only") == 0
                
                input_text_pred2 = f"""<correct_answer>
({letter})
</correct_answer>"""
                pred2 = extract_raven_choices_answer_from_xml_v2(input_text_pred2)
                gt2 = other_letter  # Ground truth is raw single letter
                assert check_answer(pred2, gt2, "raven_score_alphabet_only") == 0
                
                input_text_pred3 = f"""<correct_answer>
{letter}.
</correct_answer>"""
                pred3 = extract_raven_choices_answer_from_xml_v2(input_text_pred3)
                gt3 = other_letter  # Ground truth is raw single letter
                assert check_answer(pred3, gt3, "raven_score_alphabet_only") == 0
