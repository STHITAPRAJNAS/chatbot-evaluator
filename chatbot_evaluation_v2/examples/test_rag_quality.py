"""
Example pytest-bdd integration for RAG Evaluator.

This file demonstrates how to use the RAG Evaluator framework with pytest-bdd
for automated quality checks during builds.
"""

import os
import sys
import json
import pytest
from pytest_bdd import scenarios, given, when, then, parsers

# Add the parent directory to the path so we can import the rag_evaluator package
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from rag_evaluator.core.config import EvaluationConfig
from rag_evaluator.core.data import DataManager, RAGEvaluationSample
from rag_evaluator.core.evaluation import MetricRegistry
from rag_evaluator.non_llm import register_non_llm_metrics


# Define fixtures
@pytest.fixture
def rag_evaluation():
    """Fixture for RAG evaluation."""
    # Create directories if they don't exist
    os.makedirs("examples/data", exist_ok=True)
    os.makedirs("examples/reports", exist_ok=True)
    
    # Create configuration
    config = EvaluationConfig.from_dict({
        "name": "pytest_bdd_evaluation",
        "description": "Evaluation using pytest-bdd",
        "metrics": [
            "context_relevance", 
            "context_utilization", 
            "answer_similarity"
        ],
        "thresholds": {
            "context_relevance": 0.6,
            "context_utilization": 0.5,
            "answer_similarity": 0.6
        }
    })
    
    # Create data manager
    data_manager = DataManager()
    
    # Create metric registry
    metric_registry = MetricRegistry()
    register_non_llm_metrics(metric_registry)
    
    # Return evaluation context
    return {
        "config": config,
        "data_manager": data_manager,
        "metric_registry": metric_registry,
        "results": {}
    }


# Load scenarios from feature file
scenarios('features/rag_quality.feature')


@given('a RAG system with test samples')
def rag_system_with_samples(rag_evaluation):
    """Set up RAG system with test samples."""
    # Create example samples
    sample1 = RAGEvaluationSample(
        id="test_sample_1",
        query="What are the main causes of climate change?",
        response="Climate change is primarily caused by greenhouse gas emissions from human activities such as burning fossil fuels, deforestation, and industrial processes.",
        contexts=[
            "Human activities are the main driver of climate change, primarily due to burning fossil fuels like coal, oil, and natural gas.",
            "Deforestation contributes to climate change by reducing the number of trees that can absorb carbon dioxide."
        ],
        reference_answer="The main causes of climate change are human activities that release greenhouse gases, primarily burning fossil fuels, deforestation, and industrial processes."
    )
    
    # Add sample to data manager
    rag_evaluation["data_manager"].add_sample(sample1)


@when(parsers.parse('I evaluate the system using the "{metric_name}" metric'))
def evaluate_system(rag_evaluation, metric_name):
    """Evaluate the system using the specified metric."""
    metric_fn = rag_evaluation["metric_registry"].get(metric_name)
    assert metric_fn is not None, f"Metric {metric_name} not found in registry"
    
    results = []
    for sample_id, sample in rag_evaluation["data_manager"].samples.items():
        try:
            result = metric_fn(sample)
            if result:
                results.append(result)
        except Exception as e:
            pytest.fail(f"Error evaluating {metric_name} for sample {sample_id}: {str(e)}")
    
    # Calculate average score
    if results:
        avg_score = sum(r.score for r in results) / len(results)
        rag_evaluation["results"][metric_name] = avg_score


@then(parsers.parse('the average score should be at least {threshold:f}'))
def check_threshold(rag_evaluation, threshold):
    """Check if the average score meets the threshold."""
    for metric_name, score in rag_evaluation["results"].items():
        assert score >= threshold, f"Average score for {metric_name} ({score:.4f}) is below threshold ({threshold:.4f})"
