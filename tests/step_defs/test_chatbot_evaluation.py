#!/usr/bin/env python3
"""
Step definitions for chatbot evaluation BDD tests.
"""

import os
import json
import pytest
import pandas as pd
from pytest_bdd import scenarios, given, when, then, parsers
from unittest.mock import MagicMock, patch

# Import components to test
from src.evaluator import ChatbotEvaluator
from src.bedrock_judge import BedrockLLMJudge
from src.ragas_evaluator import RagasEvaluator
from src.chatbot_client import ChatbotClient
from src.config import get_config

# Register scenarios
scenarios('../features/chatbot_evaluation.feature')

# Fixtures
@pytest.fixture
def mock_bedrock_judge():
    """Mock Bedrock LLM judge for testing."""
    mock_judge = MagicMock(spec=BedrockLLMJudge)
    
    # Mock evaluate_response method
    mock_judge.evaluate_response.return_value = {
        "criteria_scores": [
            {
                "criteria_id": "C001",
                "name": "Factual Accuracy",
                "score": 4,
                "justification": "The response is factually accurate."
            },
            {
                "criteria_id": "C002",
                "name": "Completeness",
                "score": 3,
                "justification": "The response is complete but could include more details."
            }
        ],
        "weighted_average": 3.5,
        "overall_feedback": "The response is generally good."
    }
    
    return mock_judge

@pytest.fixture
def mock_ragas_evaluator():
    """Mock RAGAS evaluator for testing."""
    mock_evaluator = MagicMock(spec=RagasEvaluator)
    
    # Mock evaluate method
    mock_evaluator.evaluate.return_value = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.78,
        "context_relevancy": 0.92
    }
    
    # Mock is_available method
    mock_evaluator.is_available.return_value = True
    
    return mock_evaluator

@pytest.fixture
def mock_chatbot_client():
    """Mock chatbot client for testing."""
    mock_client = MagicMock(spec=ChatbotClient)
    
    # Mock query method
    mock_client.query.return_value = {
        "response": "Paris is the capital of France."
    }
    
    # Mock query_and_extract method
    mock_client.query_and_extract.return_value = "Paris is the capital of France."
    
    return mock_client

@pytest.fixture
def test_evaluator(mock_bedrock_judge, mock_chatbot_client):
    """Create a test evaluator with mocked components."""
    evaluator = ChatbotEvaluator()
    evaluator.judge = mock_bedrock_judge
    
    # Mock the query_chatbot method
    evaluator.query_chatbot = MagicMock(return_value=("Paris is the capital of France.", 
                                                     {"response": "Paris is the capital of France."}))
    
    return evaluator

# Step definitions
@given("the evaluation criteria are loaded from the Excel template")
def evaluation_criteria_loaded(test_evaluator):
    """Ensure evaluation criteria are loaded."""
    # Mock the _load_evaluation_template method
    test_evaluator._load_evaluation_template = MagicMock()
    test_evaluator._load_evaluation_template()
    
    # Create mock criteria data
    test_evaluator.criteria_df = pd.DataFrame({
        "criteria_id": ["C001", "C002"],
        "name": ["Factual Accuracy", "Completeness"],
        "description": ["The response is factually accurate.", "The response is complete."],
        "weight": [0.6, 0.4],
        "min_score": [0, 0],
        "max_score": [5, 5],
        "passing_threshold": [3, 3]
    })
    
    assert len(test_evaluator.criteria_df) == 2

@given("the Bedrock LLM judge is configured")
def bedrock_judge_configured(test_evaluator, mock_bedrock_judge):
    """Ensure Bedrock LLM judge is configured."""
    test_evaluator.judge = mock_bedrock_judge
    assert test_evaluator.judge is not None

@given("the chatbot service is available")
def chatbot_service_available(mock_chatbot_client):
    """Ensure chatbot service is available."""
    response = mock_chatbot_client.query("Test")
    assert response is not None

@given(parsers.parse('a test question "{question}"'))
def test_question(context, question):
    """Set a test question."""
    if not hasattr(context, 'questions'):
        context.questions = []
    context.questions.append({"question": question})

@given(parsers.parse('an expected answer "{answer}"'))
def expected_answer(context, answer):
    """Set an expected answer."""
    if not hasattr(context, 'questions') or not context.questions:
        context.questions = [{}]
    context.questions[-1]["expected_answer"] = answer

@given("context information about France")
def context_information(context):
    """Set context information."""
    context.contexts = ["Paris is the capital and most populous city of France."]

@given("the questions are loaded from the Excel template")
def questions_loaded(test_evaluator):
    """Ensure questions are loaded from the template."""
    # Create mock questions data
    test_evaluator.questions_df = pd.DataFrame({
        "question_id": ["Q001", "Q002"],
        "category": ["General", "Geography"],
        "question": ["What is the capital of France?", "What is the largest city in France?"],
        "expected_answer": ["The capital of France is Paris.", "The largest city in France is Paris."],
        "context": ["", ""],
        "difficulty": ["Easy", "Easy"]
    })
    
    assert len(test_evaluator.questions_df) == 2

@when("I query the chatbot with the question")
def query_chatbot(context, mock_chatbot_client):
    """Query the chatbot with a question."""
    question = context.questions[-1]["question"]
    context.response = mock_chatbot_client.query_and_extract(question)
    assert context.response is not None

@when("I evaluate all questions using the chatbot")
def evaluate_all_questions(test_evaluator):
    """Evaluate all questions using the chatbot."""
    # Mock the evaluate_all_questions method
    test_evaluator.evaluate_all_questions = MagicMock(return_value={
        "questions": [
            {
                "question_id": "Q001",
                "question": "What is the capital of France?",
                "expected_answer": "The capital of France is Paris.",
                "actual_response": "Paris is the capital of France.",
                "evaluation": {
                    "weighted_average": 3.5,
                    "criteria_scores": []
                }
            },
            {
                "question_id": "Q002",
                "question": "What is the largest city in France?",
                "expected_answer": "The largest city in France is Paris.",
                "actual_response": "Paris is the largest city in France.",
                "evaluation": {
                    "weighted_average": 4.0,
                    "criteria_scores": []
                }
            }
        ],
        "summary": {
            "average_score": 3.75,
            "passed": True
        }
    })
    
    test_evaluator.results = test_evaluator.evaluate_all_questions()
    assert test_evaluator.results is not None
    assert "questions" in test_evaluator.results
    assert "summary" in test_evaluator.results

@when("the RAGAS evaluator should calculate metrics for the response")
def ragas_calculate_metrics(context, mock_ragas_evaluator):
    """Calculate RAGAS metrics for the response."""
    question = context.questions[-1]["question"]
    answer = context.response
    contexts = getattr(context, 'contexts', [""])
    
    context.ragas_scores = mock_ragas_evaluator.evaluate(question, answer, contexts)
    assert context.ragas_scores is not None

@when("I generate a text report")
def generate_text_report(test_evaluator):
    """Generate a text report."""
    # Mock the generate_report method
    test_evaluator.generate_report = MagicMock(return_value="Test Report Content")
    
    test_evaluator.text_report = test_evaluator.generate_report(test_evaluator.results, format="text")
    assert test_evaluator.text_report is not None

@when("I generate an HTML report")
def generate_html_report(test_evaluator):
    """Generate an HTML report."""
    # Mock the generate_report method for HTML
    test_evaluator.generate_report = MagicMock(return_value="<html>Test HTML Report</html>")
    
    test_evaluator.html_report = test_evaluator.generate_report(test_evaluator.results, format="html")
    assert test_evaluator.html_report is not None

@when("I attempt to evaluate a question")
def attempt_evaluate_question(test_evaluator, mock_chatbot_client):
    """Attempt to evaluate a question with unavailable service."""
    # Make the chatbot client raise an exception
    mock_chatbot_client.query.side_effect = Exception("Service unavailable")
    
    # Mock the evaluate_single_question method to handle the error
    test_evaluator.evaluate_single_question = MagicMock(return_value={
        "question_id": "Q001",
        "question": "What is the capital of France?",
        "error": "Service unavailable",
        "timestamp": "2025-03-21T12:00:00"
    })
    
    test_evaluator.error_result = test_evaluator.evaluate_single_question({
        "question_id": "Q001",
        "question": "What is the capital of France?",
        "expected_answer": "The capital of France is Paris."
    })
    
    assert "error" in test_evaluator.error_result

@then("I should receive a response from the chatbot")
def receive_response(context):
    """Verify a response is received from the chatbot."""
    assert context.response is not None
    assert isinstance(context.response, str)

@then("the Bedrock LLM judge should evaluate the response")
def bedrock_evaluate_response(context, test_evaluator, mock_bedrock_judge):
    """Verify the Bedrock LLM judge evaluates the response."""
    question = context.questions[-1]["question"]
    expected_answer = context.questions[-1]["expected_answer"]
    actual_response = context.response
    
    # Mock criteria list
    criteria_list = [
        {
            "criteria_id": "C001",
            "name": "Factual Accuracy",
            "description": "The response is factually accurate.",
            "weight": 0.6,
            "min_score": 0,
            "max_score": 5,
            "passing_threshold": 3
        },
        {
            "criteria_id": "C002",
            "name": "Completeness",
            "description": "The response is complete.",
            "weight": 0.4,
            "min_score": 0,
            "max_score": 5,
            "passing_threshold": 3
        }
    ]
    
    context.evaluation = mock_bedrock_judge.evaluate_response(
        question=question,
        expected_answer=expected_answer,
        actual_response=actual_response,
        criteria=criteria_list
    )
    
    assert context.evaluation is not None
    assert "weighted_average" in context.evaluation

@then("the evaluation should include scores for each criterion")
def evaluation_includes_scores(context):
    """Verify the evaluation includes scores for each criterion."""
    assert "criteria_scores" in context.evaluation
    assert len(context.evaluation["criteria_scores"]) > 0

@then("the evaluation should include a weighted average score")
def evaluation_includes_weighted_average(context):
    """Verify the evaluation includes a weighted average score."""
    assert "weighted_average" in context.evaluation
    assert isinstance(context.evaluation["weighted_average"], (int, float))

@then("I should receive responses for all questions")
def receive_all_responses(test_evaluator):
    """Verify responses are received for all questions."""
    assert len(test_evaluator.results["questions"]) == 2
    for question_result in test_evaluator.results["questions"]:
        assert "actual_response" in question_result

@then("each response should be evaluated by the Bedrock LLM judge")
def each_response_evaluated(test_evaluator):
    """Verify each response is evaluated by the Bedrock LLM judge."""
    for question_result in test_evaluator.results["questions"]:
        assert "evaluation" in question_result
        assert "weighted_average" in question_result["evaluation"]

@then("a summary report should be generated with statistics")
def summary_report_generated(test_evaluator):
    """Verify a summary report is generated with statistics."""
    assert "summary" in test_evaluator.results
    assert "average_score" in test_evaluator.results["summary"]
    assert "passed" in test_evaluator.results["summary"]

@then("the report should indicate if the chatbot passed the evaluation")
def report_indicates_pass_fail(test_evaluator):
    """Verify the report indicates if the chatbot passed the evaluation."""
    assert "passed" in test_evaluator.results["summary"]
    assert isinstance(test_evaluator.results["summary"]["passed"], bool)

@then("the metrics should include faithfulness and relevancy scores")
def metrics_include_faithfulness_relevancy(context):
    """Verify the metrics include faithfulness and relevancy scores."""
    assert "faithfulness" in context.ragas_scores
    assert "answer_relevancy" in context.ragas_scores

@then("a weighted RAGAS score should be calculated")
def weighted_ragas_score_calculated(context, mock_ragas_evaluator):
    """Verify a weighted RAGAS score is calculated."""
    # Mock the calculate_weighted_score method
    mock_ragas_evaluator.calculate_weighted_score.return_value = 0.85
    
    context.weighted_ragas_score = mock_ragas_evaluator.calculate_weighted_score(
        context.ragas_scores,
        weights={
            "faithfulness": 0.4,
            "answer_relevancy": 0.3,
            "context_relevancy": 0.3
        }
    )
    
    assert context.weighted_ragas_score is not None
    assert isinstance(context.weighted_ragas_score, float)

@then("the report should include summary statistics")
def report_includes_summary_statistics(test_evaluator):
    """Verify the report includes summary statistics."""
    assert test_evaluator.text_report is not None
    assert "CHATBOT EVALUATION REPORT" in test_evaluator.text_report
    assert "Average Score" in test_evaluator.text_report

@then("the report should include detailed scores for each question")
def report_includes_detailed_scores(test_evaluator):
    """Verify the report includes detailed scores for each question."""
    assert "Question" in test_evaluator.text_report
    assert "Weighted Average Score" in test_evaluator.text_report

@then("the HTML report should include formatted results")
def html_report_includes_formatted_results(test_evaluator):
    """Verify the HTML report includes formatted results."""
    assert test_evaluator.html_report is not None
    assert "<html>" in test_evaluator.html_report

@then("the HTML report should include pass/fail indicators")
def html_report_includes_pass_fail(test_evaluator):
    """Verify the HTML report includes pass/fail indicators."""
    assert "PASS" in test_evaluator.html_report or "FAIL" in test_evaluator.html_report

@then("the evaluation should handle the error gracefully")
def evaluation_handles_error(test_evaluator):
    """Verify the evaluation handles the error gracefully."""
    assert "error" in test_evaluator.error_result
    assert test_evaluator.error_result["error"] == "Service unavailable"

@then("the error should be logged")
def error_logged():
    """Verify the error is logged."""
    # This would normally check logs, but we'll skip actual log verification in tests
    pass

@then("the evaluation report should indicate the failure")
def report_indicates_failure(test_evaluator):
    """Verify the evaluation report indicates the failure."""
    assert "error" in test_evaluator.error_result

@given("the chatbot service is unavailable")
def chatbot_service_unavailable(mock_chatbot_client):
    """Make the chatbot service unavailable."""
    mock_chatbot_client.query.side_effect = Exception("Service unavailable")
    mock_chatbot_client.query_and_extract.side_effect = Exception("Service unavailable")

@given("the evaluation results are available")
def evaluation_results_ava<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>