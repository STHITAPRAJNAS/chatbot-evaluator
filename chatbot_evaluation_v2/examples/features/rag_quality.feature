Feature: RAG System Quality
  As a developer
  I want to evaluate the quality of my RAG system
  So that I can ensure it meets quality standards

  Scenario: Evaluating context relevance
    Given a RAG system with test samples
    When I evaluate the system using the "context_relevance" metric
    Then the average score should be at least 0.6

  Scenario: Evaluating context utilization
    Given a RAG system with test samples
    When I evaluate the system using the "context_utilization" metric
    Then the average score should be at least 0.5

  Scenario: Evaluating answer similarity
    Given a RAG system with test samples
    When I evaluate the system using the "answer_similarity" metric
    Then the average score should be at least 0.6
