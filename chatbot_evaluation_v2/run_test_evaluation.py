"""
Modified run_test_evaluation.py to fix EvaluationManager usage.

This script runs a test evaluation using the example configuration and samples.
"""

import os
import sys
import logging
import json
from datetime import datetime

# Add the parent directory to the path so we can import the rag_evaluator package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rag_evaluator.core.config import ConfigManager, EvaluationConfig
from rag_evaluator.core.data import DataManager, RAGEvaluationSample, SQLEvaluationSample, EvaluationResult
from rag_evaluator.core.evaluation import MetricRegistry, EvaluationManager
from rag_evaluator.core.reporting import ReportGenerator
from rag_evaluator.non_llm import register_non_llm_metrics


def run_test_evaluation():
    """Run a test evaluation using the example configuration and samples."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    # Create directories if they don't exist
    os.makedirs("config", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Check if config and samples files exist, create them if not
    if not os.path.exists("config/test_config.json"):
        logger.info("Creating test configuration...")
        os.system("python3 create_test_config.py")
    
    if not os.path.exists("samples/test_samples.json"):
        logger.info("Creating test samples...")
        os.system("python3 create_test_samples.py")
    
    # Load configuration
    logger.info("Loading configuration...")
    with open("config/test_config.json", 'r') as f:
        config_dict = json.load(f)
    
    # Create config object
    config = EvaluationConfig.from_dict({
        "name": "test_config",
        "description": "Test configuration",
        "metrics": ["context_relevance", "context_utilization", "answer_similarity", "keyword_overlap", "sql_similarity"],
        "thresholds": config_dict.get("thresholds", {})
    })
    
    # Set output directory
    output_dir = config_dict.get("output_dir", "reports")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load samples
    logger.info("Loading samples...")
    data_manager = DataManager()
    with open("samples/test_samples.json", 'r') as f:
        samples_data = json.load(f)
    
    # Manually add each sample to the data manager
    for sample_dict in samples_data:
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
                reference_sql=sample_dict.get("reference_sql")
            )
        else:
            logger.warning(f"Unknown sample type: {sample_type}")
            continue
        
        data_manager.add_sample(sample)
    
    # Set up metrics
    logger.info("Setting up metrics...")
    metric_registry = MetricRegistry()
    
    # Register non-LLM metrics
    register_non_llm_metrics(metric_registry)
    
    # Initialize evaluation manager
    logger.info("Initializing evaluation manager...")
    evaluation_manager = EvaluationManager(
        metric_registry=metric_registry,
        data_manager=data_manager
    )
    
    # Create a custom evaluate_all function since we don't have a proper config_manager
    def evaluate_all_samples(samples):
        results = []
        for sample_id, sample in samples.items():
            for metric_name in metric_registry.list_metrics():
                metric_fn = metric_registry.get(metric_name)
                if metric_fn:
                    try:
                        result = metric_fn(sample)
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error evaluating {metric_name} for sample {sample_id}: {str(e)}")
        return results
    
    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluate_all_samples(data_manager.samples)
    
    # Generate report
    logger.info("Generating report...")
    report_generator = ReportGenerator(output_dir=output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_title = f"Test Evaluation Report - {timestamp}"
    
    # Convert results to DataFrame for reporting
    import pandas as pd
    results_df = pd.DataFrame([
        {
            "sample_id": result.sample_id,
            "metric_name": result.metric_name,
            "score": result.score
        }
        for result in results
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
    description = f"Test evaluation of {len(data_manager.samples)} samples using {len(summary)} metrics."
    report_path = report_generator.generate_html_report(
        title=report_title,
        description=description,
        summary=summary,
        threshold_results=threshold_results,
        plot_paths=plot_paths,
        results_df=results_df
    )
    
    logger.info(f"Evaluation complete. Report saved to: {report_path}")
    
    # Print summary results
    print("\nEvaluation Summary:")
    print("=" * 50)
    for metric_name, stats in summary.items():
        threshold = config.thresholds.get(metric_name, 0.0)
        status = "PASS" if stats["mean"] >= threshold else "FAIL"
        print(f"{metric_name}: {stats['mean']:.4f} (threshold: {threshold:.2f}) - {status}")
    
    # Check if all thresholds passed
    all_passed = all(threshold_results.values())
    print("\nOverall Result:", "PASS" if all_passed else "FAIL")
    
    return report_path, all_passed


if __name__ == "__main__":
    report_path, all_passed = run_test_evaluation()
    print(f"\nReport saved to: {report_path}")
    sys.exit(0 if all_passed else 1)
