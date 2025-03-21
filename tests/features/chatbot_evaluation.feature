Feature: Chatbot Evaluation
  As a quality assurance engineer
  I want to evaluate a chatbot service using Bedrock LLM as judge
  So that I can assess the quality of chatbot responses

  Background:
    Given the evaluation criteria are loaded from the Excel template
    And the Bedrock LLM judge is configured
    And the chatbot service is available

  Scenario: Evaluate a single question with Bedrock LLM judge
    Given a test question "What is the capital of France?"
    And an expected answer "The capital of France is Paris."
    When I query the chatbot with the question
    Then I should receive a response from the chatbot
    And the Bedrock LLM judge should evaluate the response
    And the evaluation should include scores for each criterion
    And the evaluation should include a weighted average score

  Scenario: Evaluate multiple questions from the template
    Given the questions are loaded from the Excel template
    When I evaluate all questions using the chatbot
    Then I should receive responses for all questions
    And each response should be evaluated by the Bedrock LLM judge
    And a summary report should be generated with statistics
    And the report should indicate if the chatbot passed the evaluation

  Scenario: Evaluate chatbot response using RAGAS metrics
    Given a test question "What is the capital of France?"
    And an expected answer "The capital of France is Paris."
    And context information about France
    When I query the chatbot with the question
    Then I should receive a response from the chatbot
    And the RAGAS evaluator should calculate metrics for the response
    And the metrics should include faithfulness and relevancy scores
    And a weighted RAGAS score should be calculated

  Scenario: Generate evaluation report in different formats
    Given the evaluation results are available
    When I generate a text report
    Then the report should include summary statistics
    And the report should include detailed scores for each question
    When I generate an HTML report
    Then the HTML report should include formatted results
    And the HTML report should include pass/fail indicators

  Scenario: Handle errors during evaluation
    Given the chatbot service is unavailable
    When I attempt to evaluate a question
    Then the evaluation should handle the error gracefully
    And the error should be logged
    And the evaluation report should indicate the failure
