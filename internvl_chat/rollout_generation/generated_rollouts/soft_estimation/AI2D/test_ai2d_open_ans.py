import pytest
import sys
import os
import base64
import json
from PIL import Image
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass
from typing import List

sys.path.append('/data/users/brandon/ob1-projects/InternVL/internvl_chat/tools')

from reasoning_data_pipeline.utils.accuracy_reward import (
    extract_dvqa_answer_int_from_xml, 
    check_answer,
    ai2d_open_answer_score,
    AnswerAcceptability
)

# def test_valid_boxed_integer():
#     """Test extraction of valid boxed integer answer"""
#     input_text = """Some text
# <correct_answer>
# $\boxed{\text{none of the above}}$
# </correct_answer>
# more text"""
#     assert extract_dvqa_answer_int_from_xml(input_text) == "42"

# def test_valid_boxed_integer():
#     """Test extraction of valid boxed integer answer"""
#     input_text = """Some text
# <correct_answer>
# $\boxed{man}$
# </correct_answer>
# more text"""
#     assert extract_dvqa_answer_int_from_xml(input_text) == "42"

# def test_valid_boxed_integer():
#     """Test extraction of valid boxed integer answer"""
#     input_text = """Some text
# <correct_answer>
# $\\boxed{42}$
# </correct_answer>
# more text"""
#     assert extract_dvqa_answer_int_from_xml(input_text) == "42"

# def test_multiple_boxed_integers():
#     """Test extraction when multiple boxed integers exist"""
#     input_text = """$\\boxed{123}$
# <correct_answer>
# $\\boxed{456}$
# </correct_answer>"""
#     assert extract_dvqa_answer_int_from_xml(input_text) == "456"

# def test_no_boxed_integer():
#     """Test when no boxed integer pattern is found"""
#     input_text = """<correct_answer>
# plain text answer
# </correct_answer>"""
#     assert extract_dvqa_answer_int_from_xml(input_text) == "plain text answer"

# def test_missing_tags():
#     """Test when XML tags are missing"""
#     input_text = "Just some text without tags"
#     assert extract_dvqa_answer_int_from_xml(input_text) == input_text

# def test_missing_closing_tag():
#     """Test when closing tag is missing"""
#     input_text = """<correct_answer>
# $\\boxed{42}$"""
#     assert extract_dvqa_answer_int_from_xml(input_text) == input_text

# def test_empty_content():
#     """Test with empty content between tags"""
#     with pytest.raises(AssertionError, match="Empty answer content between XML tags"):
#         input_text = """<correct_answer>

# </correct_answer>"""
#         extract_dvqa_answer_int_from_xml(input_text)

# def test_mismatched_tags():
#     """Test with mismatched number of opening and closing tags"""
#     with pytest.raises(AssertionError, match="Mismatched XML tags"):
#         input_text = """<correct_answer>
# $\\boxed{42}$
# </correct_answer>
# </correct_answer>"""
#         extract_dvqa_answer_int_from_xml(input_text)

# def test_complex_boxed_expression():
#     """Test with complex boxed expression"""
#     input_text = """<correct_answer>
# $\\boxed{12345}$
# </correct_answer>"""
#     assert extract_dvqa_answer_int_from_xml(input_text) == "12345"

# def test_multiple_correct_answers():
#     """Test with multiple correct_answer tags"""
#     input_text = """<correct_answer>
# first
# </correct_answer>
# <correct_answer>
# $\\boxed{789}$
# </correct_answer>"""
#     assert extract_dvqa_answer_int_from_xml(input_text) == "789"

# def test_boxed_integer_with_newlines():
#     """Test extraction with newlines between tags and content"""
#     input_text = """<correct_answer>
# $\\boxed{1}$
# </correct_answer>"""
#     assert extract_dvqa_answer_int_from_xml(input_text) == "1"

# def test_boxed_integer_with_multiple_newlines():
#     """Test extraction with multiple newlines and spaces"""
#     input_text = """<correct_answer>

#     $\\boxed{42}$

# </correct_answer>"""
#     assert extract_dvqa_answer_int_from_xml(input_text) == "42"

# def test_boxed_integer_with_mixed_format():
#     """Test extraction with mixed format (newlines and inline)"""
#     input_text = """Some text
# <correct_answer>
# $\\boxed{123}$
# </correct_answer>
# More text"""
#     assert extract_dvqa_answer_int_from_xml(input_text) == "123"

# def test_dvqa_int_only_score_exact_match():
#     """Test exact integer match for DVQA scoring"""
#     assert check_answer("42", "42", "dvqa_int_only_score") == 1
#     assert check_answer("0", "0", "dvqa_int_only_score") == 1
#     assert check_answer("-1", "-1", "dvqa_int_only_score") == 1

# def test_dvqa_int_only_score_mismatch():
#     """Test integer mismatch for DVQA scoring"""
#     assert check_answer("42", "43", "dvqa_int_only_score") == 0
#     assert check_answer("0", "1", "dvqa_int_only_score") == 0
#     assert check_answer("-1", "1", "dvqa_int_only_score") == 0

# def test_dvqa_int_only_score_invalid_input():
#     """Test invalid input handling for DVQA scoring"""
#     assert check_answer("not a number", "42", "dvqa_int_only_score") == 0
#     assert check_answer("42", "not a number", "dvqa_int_only_score") == 0
#     assert check_answer("abc", "def", "dvqa_int_only_score") == 0

# def test_dvqa_int_only_score_with_whitespace():
#     """Test DVQA scoring with whitespace handling"""
#     assert check_answer(" 42 ", "42", "dvqa_int_only_score") == 1
#     assert check_answer("42", " 42 ", "dvqa_int_only_score") == 1
#     assert check_answer(" 42 ", " 42 ", "dvqa_int_only_score") == 1

# def test_dvqa_int_only_score_negative_integers():
#     """Test DVQA scoring with negative integers"""
#     assert check_answer("-42", "-42", "dvqa_int_only_score") == 1
#     assert check_answer("-0", "-0", "dvqa_int_only_score") == 1
#     assert check_answer("-1", "-1", "dvqa_int_only_score") == 1

@pytest.fixture
def sample_image():
    img = Image.new('RGB', (10, 10), color='red')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.mark.skipif(
    not os.getenv("AZURE_CORRECTNESSJUDGE_API_KEY"),
    reason="AZURE_CORRECTNESSJUDGE_API_KEY environment variable not set"
)
def test_ai2d_basic_text_only():
    """Test basic text-only scoring without image"""
    result = ai2d_open_answer_score("The answer is 42", "The answer is 42", question="What is the answer?")
    assert result==1

@pytest.mark.skipif(
    not os.getenv("AZURE_CORRECTNESSJUDGE_API_KEY"),
    reason="AZURE_CORRECTNESSJUDGE_API_KEY environment variable not set"
)
def test_ai2d_with_image(sample_image, tmp_path):
    """Test scoring with image input"""
    image_path = tmp_path / "test.png"
    with open(image_path, 'wb') as f:
        f.write(sample_image.getvalue())
    
    result = ai2d_open_answer_score("The answer is 42", "The answer is 42", str(image_path), question="What is the answer?")
    assert result==1

def test_ai2d_invalid_image_path():
    """Test handling of invalid image path"""
    with pytest.raises(FileNotFoundError):
        ai2d_open_answer_score("answer", "answer", "nonexistent.png", question="What is the answer?")

@pytest.mark.skipif(
    not os.getenv("AZURE_CORRECTNESSJUDGE_API_KEY"),
    reason="AZURE_CORRECTNESSJUDGE_API_KEY environment variable not set"
)
def test_ai2d_empty_inputs():
    """Test handling of empty inputs"""
    result = ai2d_open_answer_score("", "", question="What is the answer?")
    assert result==0

@pytest.mark.skipif(
    not os.getenv("AZURE_CORRECTNESSJUDGE_API_KEY"),
    reason="AZURE_CORRECTNESSJUDGE_API_KEY environment variable not set"
)
def test_ai2d_special_characters():
    """Test handling of special characters in answers"""
    result = ai2d_open_answer_score("Answer: 42%", "Answer: 42%", question="What is the answer?")
    assert result==1

@pytest.mark.skipif(
    not os.getenv("AZURE_CORRECTNESSJUDGE_API_KEY"),
    reason="AZURE_CORRECTNESSJUDGE_API_KEY environment variable not set"
)
def test_ai2d_long_inputs():
    """Test handling of long input strings"""
    long_answer = "a" * 1000
    result = ai2d_open_answer_score(long_answer, long_answer, question="What is the answer?")
    assert result==1

@pytest.mark.skipif(
    not os.getenv("AZURE_CORRECTNESSJUDGE_API_KEY"),
    reason="AZURE_CORRECTNESSJUDGE_API_KEY environment variable not set"
)
def test_ai2d_semantic_equivalence():
    """Test semantic equivalence of different phrasings"""
    test_cases = [
        ("The answer is 42", "It's 42"),
        ("The result is 42", "The answer is 42"),
        ("42 is the answer", "The answer is 42"),
        ("The answer is forty-two", "The answer is 42"),
    ]
    
    for pred, gt in test_cases:
        result = ai2d_open_answer_score(pred, gt, question="What is the answer?")
        assert result==1

# AI2D Test Cases with Variations
TEST_CASES = [
    {
        "image_path": "/data/users/brandon/ob1-projects/InternVL/internvl_chat/rollout_generation/preprocessed_prompts/preprocessing_scripts/AI2D/subset_images/1453.png",
        "question": "Which letter shows the Spring Tides?\nB\nA\nC\nD\nPlease answer the question based on the options mentioned before.",
        "correct_answer": "A",
        "correct_variations": [
            "A",
            "Letter A",
            "Option A",
            "The answer is A",
            "It's A",
            "A shows the Spring Tides"
        ],
        "wrong_answers": ["B", "C", "D"],
        "uid": "a09c0052-097b-4527-a260-22acfc0e86dd"
    },
    {
        "image_path": "/data/users/brandon/ob1-projects/InternVL/internvl_chat/rollout_generation/preprocessed_prompts/preprocessing_scripts/AI2D/subset_images/1889.png",
        "question": "If the producers died off in this community, the first group to be affected would be:\nTop level carnivores\nTertiary consumers\nSecondary consumers\nPrimary conusmers\nPlease answer the question based on the options mentioned before.",
        "correct_answer": "Primary conusmers",
        "correct_variations": [
            "Primary conusmers",
            "Primary consumers",
            "The primary consumers",
            "Primary conusmers would be affected first",
            "The first group affected would be primary conusmers",
            "Primary conusmers are the first to be affected"
        ],
        "wrong_answers": [
            "Top level carnivores",
            "Tertiary consumers",
            "Secondary consumers"
        ],
        "uid": "da33bbc3-b628-45be-afec-ad3e274f899b"
    },
    {
        "image_path": "/data/users/brandon/ob1-projects/InternVL/internvl_chat/rollout_generation/preprocessed_prompts/preprocessing_scripts/AI2D/subset_images/480.png",
        "question": "How will it most likely affect the ecosystem if the population of vole decreases in the above representation of the ecosystem?\nThe population of fox will increase\nPopulation of brown hare would decrease\nPopulation of red grouse would increase\nThe population of short-eared owl will decrease.\nPlease answer the question based on the options mentioned before.",
        "correct_answer": "The population of short-eared owl will decrease.",
        "correct_variations": [
            "The population of short-eared owl will decrease",
            "Short-eared owl population will decrease",
            "The short-eared owl's population will decrease",
            "Short-eared owls will decrease in population",
            "The short-eared owl population would decrease",
            "Decrease in short-eared owl population"
        ],
        "wrong_answers": [
            "The population of fox will increase",
            "Population of brown hare would decrease",
            "Population of red grouse would increase"
        ],
        "uid": "7db6f6d9-047c-4b94-a7b5-149d28532d0e"
    }
]

def calculate_pass_rate(results):
    """Calculate the pass rate from a list of boolean results"""
    return sum(results) / len(results) if results else 0

@dataclass
class FailedCase:
    test_name: str
    predicted: str
    ground_truth: str
    image_path: str
    score: float

failed_cases: List[FailedCase] = []

def print_failed_cases():
    if failed_cases:
        print("\n=== Failed Cases ===")
        for case in failed_cases:
            print(f"\nTest: {case.test_name}")
            print(f"Predicted: {case.predicted}")
            print(f"Ground Truth: {case.ground_truth}")
            print(f"Image: {case.image_path}")
            print(f"Score: {case.score}")
        print("\n===================")

@pytest.fixture(autouse=True)
def cleanup_failed_cases():
    failed_cases.clear()
    yield
    print_failed_cases()

@pytest.mark.skipif(
    not os.getenv("AZURE_CORRECTNESSJUDGE_API_KEY"),
    reason="AZURE_CORRECTNESSJUDGE_API_KEY environment variable not set"
)
def test_ai2d_spring_tides():
    """Test spring tides question with correct and incorrect answers"""
    case = TEST_CASES[0]
    image_path = case["image_path"]
    question = case["question"]
    
    # Test correct answer variations
    correct_results = []
    for variation in case["correct_variations"]:
        result = ai2d_open_answer_score(variation, case["correct_answer"], image_path, question)
        if result == 0:
            failed_cases.append(FailedCase(
                "Spring Tides - Correct Variation",
                variation,
                case["correct_answer"],
                image_path,
                result
            ))
        correct_results.append(result)
    
    # Calculate pass rate for correct variations
    pass_rate = calculate_pass_rate(correct_results)
    assert pass_rate >= 0.8, f"Pass rate for correct variations too low: {pass_rate}"
    
    # Test incorrect answers
    wrong_results = []
    for wrong_answer in case["wrong_answers"]:
        result = ai2d_open_answer_score(wrong_answer, case["correct_answer"], image_path, question)
        if result == 1:
            failed_cases.append(FailedCase(
                "Spring Tides - Wrong Answer",
                wrong_answer,
                case["correct_answer"],
                image_path,
                result
            ))
        wrong_results.append(result)
    
    # Calculate pass rate for wrong answers (should be low)
    wrong_pass_rate = calculate_pass_rate(wrong_results)
    assert wrong_pass_rate <= 0.2, f"Pass rate for wrong answers too high: {wrong_pass_rate}"

@pytest.mark.skipif(
    not os.getenv("AZURE_CORRECTNESSJUDGE_API_KEY"),
    reason="AZURE_CORRECTNESSJUDGE_API_KEY environment variable not set"
)
def test_ai2d_producers_ecosystem():
    """Test ecosystem question with correct and incorrect answers"""
    case = TEST_CASES[1]
    image_path = case["image_path"]
    question = case["question"]
    
    # Test correct answer variations
    correct_results = []
    for variation in case["correct_variations"]:
        result = ai2d_open_answer_score(variation, case["correct_answer"], image_path, question)
        if result == 0:
            failed_cases.append(FailedCase(
                "Producers Ecosystem - Correct Variation",
                variation,
                case["correct_answer"],
                image_path,
                result
            ))
        correct_results.append(result)
    
    # Calculate pass rate for correct variations
    pass_rate = calculate_pass_rate(correct_results)
    assert pass_rate >= 0.8, f"Pass rate for correct variations too low: {pass_rate}"
    
    # Test incorrect answers
    wrong_results = []
    for wrong_answer in case["wrong_answers"]:
        result = ai2d_open_answer_score(wrong_answer, case["correct_answer"], image_path, question)
        if result == 1:
            failed_cases.append(FailedCase(
                "Producers Ecosystem - Wrong Answer",
                wrong_answer,
                case["correct_answer"],
                image_path,
                result
            ))
        wrong_results.append(result)
    
    # Calculate pass rate for wrong answers (should be low)
    wrong_pass_rate = calculate_pass_rate(wrong_results)
    assert wrong_pass_rate <= 0.2, f"Pass rate for wrong answers too high: {wrong_pass_rate}"

@pytest.mark.skipif(
    not os.getenv("AZURE_CORRECTNESSJUDGE_API_KEY"),
    reason="AZURE_CORRECTNESSJUDGE_API_KEY environment variable not set"
)
def test_ai2d_vole_ecosystem():
    """Test vole ecosystem question with correct and incorrect answers"""
    case = TEST_CASES[2]
    image_path = case["image_path"]
    question = case["question"]
    
    # Test correct answer variations
    correct_results = []
    for variation in case["correct_variations"]:
        result = ai2d_open_answer_score(variation, case["correct_answer"], image_path, question)
        if result == 0:
            failed_cases.append(FailedCase(
                "Vole Ecosystem - Correct Variation",
                variation,
                case["correct_answer"],
                image_path,
                result
            ))
        correct_results.append(result)
    
    # Calculate pass rate for correct variations
    pass_rate = calculate_pass_rate(correct_results)
    assert pass_rate >= 0.8, f"Pass rate for correct variations too low: {pass_rate}"
    
    # Test incorrect answers
    wrong_results = []
    for wrong_answer in case["wrong_answers"]:
        result = ai2d_open_answer_score(wrong_answer, case["correct_answer"], image_path, question)
        if result == 1:
            failed_cases.append(FailedCase(
                "Vole Ecosystem - Wrong Answer",
                wrong_answer,
                case["correct_answer"],
                image_path,
                result
            ))
        wrong_results.append(result)
    
    # Calculate pass rate for wrong answers (should be low)
    wrong_pass_rate = calculate_pass_rate(wrong_results)
    assert wrong_pass_rate <= 0.2, f"Pass rate for wrong answers too high: {wrong_pass_rate}"

@pytest.mark.skipif(
    not os.getenv("AZURE_CORRECTNESSJUDGE_API_KEY"),
    reason="AZURE_CORRECTNESSJUDGE_API_KEY environment variable not set"
)
def test_ai2d_overall_pass_rates():
    """Test overall pass rates across all test cases"""
    all_correct_results = []
    all_wrong_results = []
    
    for case in TEST_CASES:
        image_path = case["image_path"]
        question = case["question"]
        
        # Test correct variations
        for variation in case["correct_variations"]:
            result = ai2d_open_answer_score(variation, case["correct_answer"], image_path, question)
            if result == 0:
                failed_cases.append(FailedCase(
                    "Overall - Correct Variation",
                    variation,
                    case["correct_answer"],
                    image_path,
                    result
                ))
            all_correct_results.append(result)
        
        # Test wrong answers
        for wrong_answer in case["wrong_answers"]:
            result = ai2d_open_answer_score(wrong_answer, case["correct_answer"], image_path, question)
            if result == 1:
                failed_cases.append(FailedCase(
                    "Overall - Wrong Answer",
                    wrong_answer,
                    case["correct_answer"],
                    image_path,
                    result
                ))
            all_wrong_results.append(result)
    
    # Calculate overall pass rates
    correct_pass_rate = calculate_pass_rate(all_correct_results)
    wrong_pass_rate = calculate_pass_rate(all_wrong_results)
    
    # Assert reasonable pass rates
    assert correct_pass_rate >= 0.8, f"Overall pass rate for correct variations too low: {correct_pass_rate}"
    assert wrong_pass_rate <= 0.2, f"Overall pass rate for wrong answers too high: {wrong_pass_rate}"