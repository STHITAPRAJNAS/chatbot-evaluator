"""
Main module for the RAG evaluator package.

This module provides the main entry point for the RAG evaluator package.
"""

from typing import Dict, Any, List, Optional, Union
import os
import sys
import json
import argparse
import logging
from datetime import datetime

from rag_evaluator.core.config import ConfigManager
from rag_evaluator.core.data import DataManager
from rag_evaluator.core.evaluation import MetricRegistry, EvaluationManager
from rag_evaluator.core.reporting import ReportGenerator
from rag_evaluator.llm_critique import OpenAIProvider, register_llm_critics
from rag_evaluator.non_llm import register_non_llm_metrics
from rag_evaluator.pytest_bdd import create_example_files


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration.
    
    Args:
        log_level: Logging level.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )


def run_evaluation(
    config_path: str,
    samples_path: str,
    output_dir: str = None,
    report_title: str = "RAG Evaluation Report"
):
    """Run evaluation with the specified configuration and samples.
    
    Args:
        config_path: Path to configuration file.
        samples_path: Path to samples file.
        output_dir: Directory to save output files.
        report_title: Title for the evaluation report.
        
    Returns:
        Path to the generated report.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running evaluation with config: {config_path}, samples: {samples_path}")
    
    # Load configuration
    config_manager = ConfigManager()
    config_manager.load_from_file(config_path)
    config = config_manager.get_config()
    
    # Override output directory if specified
    if output_dir:
        config.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # Load samples
    data_manager = DataManager()
    data_manager.load_from_file(samples_path)
    
    # Set up metrics
    metric_registry = MetricRegistry()
    
    # Register non-LLM metrics
    register_non_llm_metrics(metric_registry)
    
    # Register LLM critics if enabled
    if config.use_llm_critique and "openai_api_key" in config.llm_config:
        llm_provider = OpenAIProvider(
            model_name=config.llm_config.get("model_name", "gpt-4"),
            api_key=config.llm_config.get("openai_api_key"),
            temperature=config.llm_config.get("temperature", 0.0),
            max_tokens=config.llm_config.get("max_tokens", 1000)
        )
        register_llm_critics(metric_registry, llm_provider)
    
    # Initialize evaluation manager
    evaluation_manager = EvaluationManager(
        metric_registry=metric_registry,
        config=config
    )
    
    # Run evaluation
    results = evaluation_manager.evaluate_all(data_manager.samples)
    
    # Generate report
    report_generator = ReportGenerator(output_dir=config.output_dir)
    report_path = evaluation_manager.generate_report(
        results=results,
        report_title=report_title,
        report_generator=report_generator
    )
    
    logger.info(f"Evaluation complete. Report saved to: {report_path}")
    
    return report_path


def main():
    """Main entry point for the RAG evaluator."""
    parser = argparse.ArgumentParser(description="RAG Evaluator")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    evaluate_parser.add_argument("--config", required=True, help="Path to configuration file")
    evaluate_parser.add_argument("--samples", required=True, help="Path to samples file")
    evaluate_parser.add_argument("--output-dir", help="Directory to save output files")
    evaluate_parser.add_argument("--report-title", default="RAG Evaluation Report", help="Title for the evaluation report")
    evaluate_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    
    # Create examples command
    examples_parser = subparsers.add_parser("create-examples", help="Create example files")
    examples_parser.add_argument("--output-dir", required=True, help="Directory to save example files")
    
    args = parser.parse_args()
    
    if args.command == "evaluate":
        setup_logging(args.log_level)
        run_evaluation(
            config_path=args.config,
            samples_path=args.samples,
            output_dir=args.output_dir,
            report_title=args.report_title
        )
    elif args.command == "create-examples":
        os.makedirs(args.output_dir, exist_ok=True)
        create_example_files(args.output_dir)
        print(f"Example files created in {args.output_dir}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
