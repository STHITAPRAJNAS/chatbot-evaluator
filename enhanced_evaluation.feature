Feature: Enhanced Chatbot Evaluation with Test Client
  As a quality assurance engineer
  I want to evaluate a chatbot service using a test client and Bedrock LLM as judge
  So that I can assess the quality of chatbot responses in a test environment

  Background:
    Given the enhanced evaluation criteria are loaded from the Excel template
    And the Bedrock LLM judge is configured
    And a test client is available for the chatbot service

  Scenario: Evaluate a single question using test client
    Given a test question "What is the capital of France?"
    And an expected answer "The capital of France is Paris."
    When I query the chatbot with the test client
    Then I should receive a response from the chatbot
    And the Bedrock LLM judge should evaluate the response
    And the evaluation should include scores for each criterion
    And the evaluation should include a weighted average score

  Scenario: Evaluate a multi-turn conversation using test client
    Given the following conversation turns:
      | question                        | expected_answer                                                |
      | I want to plan a trip to Paris. | Paris is a wonderful destination! What would you like to know? |
      | What are the best attractions?  | Paris has many famous attractions including the Eiffel Tower.  |
      | How many days should I spend?   | For a first visit to Paris, 3-4 days is recommended.          |
    When I conduct the multi-turn conversation with the test client
    Then I should receive responses for all turns
    And each turn should be evaluated by the Bedrock LLM judge
    And the evaluation should consider conversation coherence
    And a conversation summary report should be generated

  Scenario: Evaluate a RAG-based response with context
    Given a test question "What is quantum computing?"
    And an expected answer "Quantum computing uses quantum mechanics to perform calculations."
    And context information about quantum computing
    When I query the RAG chatbot with the test client
    Then I should receive a response that utilizes the context
    And the evaluation should include RAG-specific criteria
    And the RAGAS evaluator should calculate metrics for the response
    And the metrics should include faithfulness and context relevancy

  Scenario: Load evaluation criteria directly from feature file
    Given the following evaluation criteria:
      | name              | description                                    | weight | min_score | max_score | threshold |
      | Factual Accuracy  | The response contains factually correct info.  | 0.4    | 0         | 5         | 3         |
      | Completeness      | The response addresses all aspects.            | 0.3    | 0         | 5         | 3         |
      | Clarity           | The response is clear and well-structured.     | 0.3    | 0         | 5         | 3         |
    And the following test questions:
      | question                       | expected_answer                                |
      | What is the capital of France? | The capital of France is Paris.                |
      | What is machine learning?      | Machine learning is a subset of AI that...     |
    When I evaluate the questions using criteria from the feature file
    Then the evaluation should use the provided criteria
    And the results should include scores for each criterion

  Scenario: Generate comprehensive evaluation report for RAG application
    Given a set of evaluated RAG responses
    When I generate an enhanced evaluation report
    Then the report should include RAG-specific metrics
    And the report should include context utilization scores
    And the report should include source attribution assessment
    And the report should provide an overall RAG quality score
