"""
Example usage of the RAG Evaluator framework.

This script demonstrates how to use the RAG Evaluator framework to evaluate
a RAG-based chatbot system with both non-LLM and LLM critique methods.
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add the parent directory to the path so we can import the rag_evaluator package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rag_evaluator.core.config import EvaluationConfig
from rag_evaluator.core.data import DataManager, RAGEvaluationSample, SQLEvaluationSample
from rag_evaluator.core.evaluation import MetricRegistry, EvaluationManager
from rag_evaluator.core.reporting import ReportGenerator
from rag_evaluator.non_llm import register_non_llm_metrics
from rag_evaluator.llm_critique import register_llm_critique_metrics
from rag_evaluator.llm_critique.providers import OpenAIProvider


def run_example_evaluation(use_llm_critique=False, api_key=None):
    """Run an example evaluation using the RAG Evaluator framework.
    
    Args:
        use_llm_critique: Whether to use LLM critique methods.
        api_key: OpenAI API key for LLM critique (required if use_llm_critique is True).
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    # Create directories if they don't exist
    os.makedirs("examples", exist_ok=True)
    os.makedirs("examples/data", exist_ok=True)
    os.makedirs("examples/reports", exist_ok=True)
    
    # Create example configuration
    logger.info("Creating example configuration...")
    config = EvaluationConfig.from_dict({
        "name": "example_evaluation",
        "description": "Example evaluation of a RAG system",
        "metrics": [
            "context_relevance", 
            "context_utilization", 
            "answer_similarity", 
            "keyword_overlap", 
            "entity_recall"
        ],
        "thresholds": {
            "context_relevance": 0.6,
            "context_utilization": 0.5,
            "answer_similarity": 0.6,
            "keyword_overlap": 0.4,
            "entity_recall": 0.5
        }
    })
    
    # Create example samples
    logger.info("Creating example samples...")
    data_manager = DataManager()
    
    # Example RAG samples
    rag_sample1 = RAGEvaluationSample(
        id="rag_example_1",
        query="What are the main causes of climate change?",
        response="Climate change is primarily caused by greenhouse gas emissions from human activities such as burning fossil fuels, deforestation, and industrial processes. These activities release carbon dioxide, methane, and other gases that trap heat in the atmosphere, leading to global warming and climate disruption.",
        contexts=[
            "Human activities are the main driver of climate change, primarily due to burning fossil fuels like coal, oil, and natural gas. These activities add greenhouse gases to the atmosphere, trapping heat and raising global temperatures.",
            "Deforestation contributes to climate change by reducing the number of trees that can absorb carbon dioxide, while also often involving burning that releases stored carbon back into the atmosphere.",
            "Industrial processes and agricultural practices also release significant amounts of greenhouse gases, including methane from livestock and rice paddies, and nitrous oxide from fertilizer use."
        ],
        reference_answer="The main causes of climate change are human activities that release greenhouse gases, primarily burning fossil fuels (coal, oil, natural gas), deforestation, and industrial/agricultural processes. These activities increase atmospheric concentrations of carbon dioxide, methane, and other heat-trapping gases."
    )
    
    rag_sample2 = RAGEvaluationSample(
        id="rag_example_2",
        query="How do electric vehicles help reduce carbon emissions?",
        response="Electric vehicles reduce carbon emissions by eliminating tailpipe emissions that come from conventional vehicles with internal combustion engines. When powered by renewable energy sources like solar or wind, they can offer a significantly lower carbon footprint compared to gasoline or diesel vehicles.",
        contexts=[
            "Electric vehicles (EVs) produce zero direct emissions, eliminating tailpipe pollutants including greenhouse gases, particulate matter, volatile organic compounds, and other air pollutants.",
            "The overall emissions reduction from EVs depends on the electricity source. When powered by renewable energy, EVs offer dramatically lower lifecycle emissions compared to conventional vehicles.",
            "Studies show that even when powered by electricity from the current grid mix (which includes fossil fuels), EVs typically have a lower carbon footprint than conventional vehicles due to their higher efficiency."
        ],
        reference_answer="Electric vehicles help reduce carbon emissions by eliminating tailpipe emissions from burning fossil fuels. Their overall environmental impact depends on the electricity source, with renewable energy providing the greatest benefits. Even when charged using the current electricity grid mix, EVs typically produce fewer lifecycle emissions than conventional vehicles due to their higher energy efficiency."
    )
    
    # Example SQL samples
    sql_sample1 = SQLEvaluationSample(
        id="sql_example_1",
        query="Find all customers who made purchases totaling more than $1000 in the last month",
        generated_sql="SELECT c.customer_id, c.name, SUM(o.amount) as total FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH) GROUP BY c.customer_id, c.name HAVING total > 1000",
        reference_sql="SELECT c.customer_id, c.name, SUM(o.amount) as total_amount FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH) GROUP BY c.customer_id, c.name HAVING total_amount > 1000"
    )
    
    # Add samples to data manager
    data_manager.add_sample(rag_sample1)
    data_manager.add_sample(rag_sample2)
    data_manager.add_sample(sql_sample1)
    
    # Set up metrics
    logger.info("Setting up metrics...")
    metric_registry = MetricRegistry()
    
    # Register non-LLM metrics
    register_non_llm_metrics(metric_registry)
    
    # Register LLM critique metrics if enabled
    if use_llm_critique:
        if not api_key:
            logger.warning("LLM critique enabled but no API key provided. Skipping LLM critique.")
        else:
            logger.info("Registering LLM critique metrics...")
            llm_provider = OpenAIProvider(
                api_key=api_key,
                model_name="gpt-4",
                temperature=0.0
            )
            register_llm_critique_metrics(metric_registry, llm_provider)
    
    # Initialize evaluation manager
    logger.info("Initializing evaluation manager...")
    evaluation_manager = EvaluationManager(
        metric_registry=metric_registry,
        data_manager=data_manager
    )
    
    # Create a custom evaluate_all function
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
    report_generator = ReportGenerator(output_dir="examples/reports")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_title = f"Example Evaluation Report - {timestamp}"
    
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
    description = f"Example evaluation of {len(data_manager.samples)} samples using {len(summary)} metrics."
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Run an example evaluation using the RAG Evaluator framework.")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM critique methods")
    parser.add_argument("--api-key", type=str, help="OpenAI API key for LLM critique")
    
    args = parser.parse_args()
    
    report_path, all_passed = run_example_evaluation(
        use_llm_critique=args.use_llm,
        api_key=args.api_key
    )
    
    print(f"\nReport saved to: {report_path}")
    sys.exit(0 if all_passed else 1)
