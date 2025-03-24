"""
pytest-bdd integration for the RAG evaluator.

This module provides utilities for integrating the RAG evaluator with pytest-bdd
for automated quality checks during builds.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import os
import json
import logging
import pytest
from pytest_bdd import given, when, then, parsers, scenario
import pandas as pd

from rag_evaluator.core.config import ConfigManager, EvaluationConfig
from rag_evaluator.core.data import DataManager, EvaluationSample, RAGEvaluationSample, SQLEvaluationSample
from rag_evaluator.core.evaluation import EvaluationManager, MetricRegistry
from rag_evaluator.core.reporting import ReportGenerator
from rag_evaluator.llm_critique import OpenAIProvider, register_llm_critics
from rag_evaluator.non_llm import register_non_llm_metrics


class BDDEvaluationContext:
    """Context for BDD evaluation scenarios."""
    
    def __init__(self):
        """Initialize BDD evaluation context."""
        self.config_manager = ConfigManager()
        self.data_manager = DataManager()
        self.metric_registry = MetricRegistry()
        self.evaluation_manager = None
        self.results = None
        self.report_path = None
        self.logger = logging.getLogger(__name__)
    
    def setup_evaluation(self, config_path: str = None, config_dict: Dict[str, Any] = None):
        """Set up evaluation with configuration.
        
        Args:
            config_path: Path to configuration file.
            config_dict: Configuration dictionary.
        """
        if config_path and os.path.exists(config_path):
            self.config_manager.load_from_file(config_path)
        elif config_dict:
            self.config_manager.load_from_dict(config_dict)
        else:
            self.logger.warning("No configuration provided, using defaults")
            self.config_manager.set_default_config()
        
        # Register metrics based on configuration
        config = self.config_manager.get_config()
        
        # Register non-LLM metrics
        register_non_llm_metrics(self.metric_registry)
        
        # Register LLM critics if enabled
        if config.use_llm_critique and "openai_api_key" in config.llm_config:
            llm_provider = OpenAIProvider(
                model_name=config.llm_config.get("model_name", "gpt-4"),
                api_key=config.llm_config.get("openai_api_key"),
                temperature=config.llm_config.get("temperature", 0.0),
                max_tokens=config.llm_config.get("max_tokens", 1000)
            )
            register_llm_critics(self.metric_registry, llm_provider)
        
        # Initialize evaluation manager
        self.evaluation_manager = EvaluationManager(
            metric_registry=self.metric_registry,
            config=config
        )
    
    def load_samples(self, samples_path: str = None, samples: List[Dict[str, Any]] = None):
        """Load evaluation samples.
        
        Args:
            samples_path: Path to samples file.
            samples: List of sample dictionaries.
        """
        if samples_path and os.path.exists(samples_path):
            self.data_manager.load_from_file(samples_path)
        elif samples:
            for sample_dict in samples:
                sample_type = sample_dict.get("type", "rag")
                
                if sample_type == "rag":
                    sample = RAGEvaluationSample(
                        id=sample_dict.get("id", ""),
                        query=sample_dict.get("query", ""),
                        response=sample_dict.get("response", ""),
                        contexts=sample_dict.get("contexts", []),
                        reference_answer=sample_dict.get("reference_answer")
                    )
                elif sample_type == "sql":
                    sample = SQLEvaluationSample(
                        id=sample_dict.get("id", ""),
                        query=sample_dict.get("query", ""),
                        generated_sql=sample_dict.get("generated_sql", ""),
                        reference_sql=sample_dict.get("reference_sql"),
                        execution_result=sample_dict.get("execution_result")
                    )
                else:
                    self.logger.warning(f"Unknown sample type: {sample_type}")
                    continue
                
                self.data_manager.add_sample(sample)
        else:
            self.logger.warning("No samples provided")
    
    def run_evaluation(self):
        """Run evaluation on loaded samples."""
        if not self.evaluation_manager:
            raise ValueError("Evaluation manager not initialized. Call setup_evaluation first.")
        
        if not self.data_manager.samples:
            raise ValueError("No samples loaded. Call load_samples first.")
        
        self.results = self.evaluation_manager.evaluate_all(self.data_manager.samples)
    
    def generate_report(self, report_title: str = "RAG Evaluation Report"):
        """Generate evaluation report.
        
        Args:
            report_title: Title for the report.
            
        Returns:
            Path to the generated report.
        """
        if not self.results:
            raise ValueError("No evaluation results. Call run_evaluation first.")
        
        config = self.config_manager.get_config()
        report_generator = ReportGenerator(output_dir=config.output_dir)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame([
            {
                "sample_id": result.sample_id,
                "metric_name": result.metric_name,
                "score": result.score
            }
            for result in self.results
        ])
        
        # Calculate summary statistics
        summary = {}
        for metric_name in results_df["metric_name"].unique():
            metric_df = results_df[results_df["metric_name"] == metric_name]
            summary[metric_name] = {
                "mean": metric_df["score"].mean(),
                "median": metric_df["score"].median(),
                "min": metric_df["score"].min(),
                "max": metric_df["score"].max(),
                "std": metric_df["score"].std()
            }
        
        # Check thresholds
        threshold_results = {}
        for metric_name, stats in summary.items():
            threshold = config.thresholds.get(metric_name, 0.0)
            threshold_results[metric_name] = stats["mean"] >= threshold
        
        # Generate plots
        plot_paths = report_generator.generate_summary_plots(
            summary, config.thresholds
        )
        
        if not results_df.empty:
            distribution_plots = report_generator.generate_distribution_plots(
                results_df, summary, config.thresholds
            )
            plot_paths.update(distribution_plots)
            
            heatmap_path = report_generator.generate_heatmap(results_df)
            if heatmap_path:
                plot_paths["heatmap"] = heatmap_path
        
        # Generate HTML report
        description = f"Evaluation of {len(self.data_manager.samples)} samples using {len(summary)} metrics."
        self.report_path = report_generator.generate_html_report(
            title=report_title,
            description=description,
            summary=summary,
            threshold_results=threshold_results,
            plot_paths=plot_paths,
            results_df=results_df
        )
        
        return self.report_path
    
    def check_thresholds(self) -> bool:
        """Check if all metrics meet their thresholds.
        
        Returns:
            True if all metrics meet their thresholds, False otherwise.
        """
        if not self.results:
            raise ValueError("No evaluation results. Call run_evaluation first.")
        
        config = self.config_manager.get_config()
        
        # Convert results to DataFrame
        results_df = pd.DataFrame([
            {
                "sample_id": result.sample_id,
                "metric_name": result.metric_name,
                "score": result.score
            }
            for result in self.results
        ])
        
        # Calculate mean scores for each metric
        metric_means = {}
        for metric_name in results_df["metric_name"].unique():
            metric_df = results_df[results_df["metric_name"] == metric_name]
            metric_means[metric_name] = metric_df["score"].mean()
        
        # Check thresholds
        all_passed = True
        failed_metrics = []
        
        for metric_name, mean_score in metric_means.items():
            threshold = config.thresholds.get(metric_name, 0.0)
            if mean_score < threshold:
                all_passed = False
                failed_metrics.append({
                    "metric": metric_name,
                    "score": mean_score,
                    "threshold": threshold
                })
        
        if not all_passed:
            self.logger.warning(f"Failed metrics: {failed_metrics}")
        
        return all_passed


# Create a shared context for BDD scenarios
_context = BDDEvaluationContext()


# BDD step definitions

@given(parsers.parse('a RAG evaluation configuration at "{config_path}"'))
def given_config_file(config_path):
    """Given step for loading configuration from file."""
    _context.setup_evaluation(config_path=config_path)


@given(parsers.parse('a RAG evaluation configuration with thresholds:\n{thresholds_table}'))
def given_config_with_thresholds(thresholds_table):
    """Given step for configuration with thresholds from table."""
    # Parse thresholds table
    thresholds = {}
    for line in thresholds_table.strip().split('\n'):
        parts = line.split('|')
        if len(parts) >= 3:
            metric_name = parts[1].strip()
            threshold_value = float(parts[2].strip())
            thresholds[metric_name] = threshold_value
    
    config_dict = {
        "thresholds": thresholds,
        "use_llm_critique": False  # Default to not using LLM critique in tests
    }
    
    _context.setup_evaluation(config_dict=config_dict)


@given(parsers.parse('evaluation samples at "{samples_path}"'))
def given_samples_file(samples_path):
    """Given step for loading samples from file."""
    _context.load_samples(samples_path=samples_path)


@given(parsers.parse('a RAG evaluation sample:\n{sample_json}'))
def given_rag_sample(sample_json):
    """Given step for a single RAG sample in JSON format."""
    try:
        sample_dict = json.loads(sample_json)
        _context.load_samples(samples=[sample_dict])
    except json.JSONDecodeError:
        pytest.fail(f"Invalid JSON: {sample_json}")


@when('the evaluation is run')
def when_evaluation_run():
    """When step for running the evaluation."""
    _context.run_evaluation()


@when(parsers.parse('the evaluation report is generated as "{report_title}"'))
def when_report_generated(report_title):
    """When step for generating the evaluation report."""
    _context.generate_report(report_title=report_title)


@then('all quality thresholds should pass')
def then_thresholds_pass():
    """Then step for checking if all thresholds pass."""
    assert _context.check_thresholds(), "Some metrics failed to meet their thresholds"


@then(parsers.parse('the "{metric_name}" score should be at least {threshold:f}'))
def then_metric_threshold(metric_name, threshold):
    """Then step for checking a specific metric threshold."""
    if not _context.results:
        pytest.fail("No evaluation results. Call run_evaluation first.")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame([
        {
            "sample_id": result.sample_id,
            "metric_name": result.metric_name,
            "score": result.score
        }
        for result in _context.results
    ])
    
    # Calculate mean score for the metric
    metric_df = results_df[results_df["metric_name"] == metric_name]
    
    if metric_df.empty:
        pytest.fail(f"Metric '{metric_name}' not found in results")
    
    mean_score = metric_df["score"].mean()
    
    assert mean_score >= threshold, f"Metric '{metric_name}' score {mean_score} is below threshold {threshold}"


@then('the evaluation report should be available')
def then_report_available():
    """Then step for checking if the report is available."""
    assert _context.report_path is not None, "Report not generated"
    assert os.path.exists(_context.report_path), f"Report file not found at {_context.report_path}"


# Example BDD scenario function
@scenario('features/rag_evaluation.feature', 'Basic RAG quality evaluation')
def test_basic_rag_evaluation():
    """Test basic RAG quality evaluation scenario."""
    pass


# Helper function to create feature directory and example feature file
def create_example_feature_file(base_dir: str):
    """Create example feature file for pytest-bdd.
    
    Args:
        base_dir: Base directory for feature files.
    """
    features_dir = os.path.join(base_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    
    feature_path = os.path.join(features_dir, "rag_evaluation.feature")
    
    feature_content = """
Feature: RAG System Quality Evaluation
  As a developer of a RAG-based system
  I want to automatically evaluate the quality of my system
  So that I can ensure it meets quality standards during builds

  Background:
    Given a RAG evaluation configuration with thresholds:
      | context_relevance    | 0.7 |
      | context_utilization  | 0.6 |
      | answer_similarity    | 0.7 |
      | keyword_overlap      | 0.5 |

  Scenario: Basic RAG quality evaluation
    Given a RAG evaluation sample:
      '''
      {
        "id": "sample1",
        "type": "rag",
        "query": "What are the benefits of renewable energy?",
        "response": "Renewable energy offers numerous benefits including reduced greenhouse gas emissions, decreased air pollution, energy independence, job creation in the green sector, and long-term cost savings. Solar, wind, and hydroelectric power are sustainable alternatives to fossil fuels.",
        "contexts": [
          "Renewable energy sources such as solar, wind, and hydroelectric power produce minimal greenhouse gas emissions and air pollution compared to fossil fuels. This makes them essential for combating climate change and improving air quality.",
          "The renewable energy sector has created millions of jobs worldwide. According to the International Renewable Energy Agency, the industry employed 11.5 million people globally in 2019, with potential for further growth.",
          "While initial installation costs can be high, renewable energy systems typically offer long-term savings. Solar panels, for instance, can significantly reduce electricity bills and may pay for themselves within 5-10 years, while lasting 25+ years."
        ],
        "reference_answer": "Renewable energy benefits include reduced emissions, lower pollution, energy security, job creation, and cost effectiveness over time. Major renewable sources include solar, wind, and hydroelectric power."
      }
      '''
    When the evaluation is run
    Then the "context_relevance" score should be at least 0.7
    And the "context_utilization" score should be at least 0.6
    And the "answer_similarity" score should be at least 0.7
    And the "keyword_overlap" score should be at least 0.5

  Scenario: Generate evaluation report
    Given a RAG evaluation sample:
      '''
      {
        "id": "sample1",
        "type": "rag",
        "query": "What are the benefits of renewable energy?",
        "response": "Renewable energy offers numerous benefits including reduced greenhouse gas emissions, decreased air pollution, energy independence, job creation in the green sector, and long-term cost savings. Solar, wind, and hydroelectric power are sustainable altern<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>