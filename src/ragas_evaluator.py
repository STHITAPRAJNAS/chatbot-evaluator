#!/usr/bin/env python3
"""
RAGAS evaluation module for chatbot evaluation.
This module provides functionality to use RAGAS metrics for evaluating
chatbot responses.
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RagasEvaluator:
    """
    A class to evaluate chatbot responses using RAGAS metrics.
    """
    
    def __init__(self, metrics_config: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the RagasEvaluator.
        
        Args:
            metrics_config: Optional list of RAGAS metrics configuration
        """
        self.metrics_config = metrics_config or []
        self._initialize_ragas()
        
    def _initialize_ragas(self):
        """Initialize RAGAS evaluation components."""
        try:
            # Import RAGAS components
            from ragas.metrics import (
                faithfulness, 
                answer_relevancy,
                context_relevancy,
                context_precision,
                context_recall
            )
            
            # Store available metrics
            self.available_metrics = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_relevancy": context_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall
            }
            
            logger.info("RAGAS evaluation components initialized successfully")
        except ImportError as e:
            logger.warning(f"Could not import RAGAS components: {str(e)}")
            logger.warning("RAGAS evaluation will be unavailable")
            self.available_metrics = {}
    
    def is_available(self) -> bool:
        """
        Check if RAGAS evaluation is available.
        
        Returns:
            Boolean indicating if RAGAS can be used
        """
        return len(self.available_metrics) > 0
    
    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate a chatbot response using RAGAS metrics.
        
        Args:
            question: The question asked
            answer: The chatbot's answer
            contexts: List of context passages used for the answer
            metrics: Optional list of specific metrics to use
            
        Returns:
            Dictionary of metric names to scores
        """
        if not self.is_available():
            logger.warning("RAGAS evaluation is not available")
            return {}
        
        try:
            # Import RAGAS evaluation components
            from ragas.metrics import evaluate
            from datasets import Dataset
            
            # Prepare data for evaluation
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts]
            }
            
            # Create dataset
            dataset = Dataset.from_dict(data)
            
            # Determine which metrics to use
            metrics_to_use = []
            if metrics:
                for metric_name in metrics:
                    if metric_name in self.available_metrics:
                        metrics_to_use.append(self.available_metrics[metric_name])
                    else:
                        logger.warning(f"Metric '{metric_name}' not available")
            else:
                # Use all available metrics
                metrics_to_use = list(self.available_metrics.values())
            
            if not metrics_to_use:
                logger.warning("No valid metrics specified for evaluation")
                return {}
            
            # Run evaluation
            result = evaluate(dataset, metrics_to_use)
            
            # Convert result to dictionary
            scores = {}
            for metric_name, value in result.items():
                scores[metric_name] = float(value)
            
            return scores
        
        except Exception as e:
            logger.error(f"Error during RAGAS evaluation: {str(e)}")
            return {}
    
    def calculate_weighted_score(
        self,
        scores: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate a weighted score from individual metric scores.
        
        Args:
            scores: Dictionary of metric scores
            weights: Optional dictionary of metric weights
            
        Returns:
            Weighted average score
        """
        if not scores:
            return 0.0
        
        # Use provided weights or equal weights
        if weights is None:
            # Equal weights for all metrics
            weight_value = 1.0 / len(scores)
            weights = {metric: weight_value for metric in scores}
        
        # Calculate weighted sum
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, score in scores.items():
            if metric in weights:
                weight = weights[metric]
                weighted_sum += score * weight
                total_weight += weight
        
        # Return weighted average
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0
    
    def evaluate_from_template(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        template_path: str
    ) -> Dict[str, Any]:
        """
        Evaluate using metrics and weights from an Excel template.
        
        Args:
            question: The question asked
            answer: The chatbot's answer
            contexts: List of context passages used for the answer
            template_path: Path to the Excel template
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Load RAGAS metrics from template
            ragas_df = pd.read_excel(template_path, sheet_name="RAGAS_Metrics")
            
            # Extract metrics and weights
            metrics = []
            weights = {}
            
            for _, row in ragas_df.iterrows():
                metric_name = row["name"]
                metrics.append(metric_name)
                weights[metric_name] = float(row["weight"])
            
            # Run evaluation
            scores = self.evaluate(question, answer, contexts, metrics)
            
            # Calculate weighted score
            weighted_score = self.calculate_weighted_score(scores, weights)
            
            # Get passing thresholds
            thresholds = {}
            for _, row in ragas_df.iterrows():
                metric_name = row["name"]
                thresholds[metric_name] = float(row["passing_threshold"])
            
            # Determine which metrics passed their thresholds
            passed_metrics = {}
            for metric, score in scores.items():
                threshold = thresholds.get(metric, 0.7)  # Default threshold
                passed_metrics[metric] = score >= threshold
            
            # Prepare result
            result = {
                "scores": scores,
                "weighted_score": weighted_score,
                "thresholds": thresholds,
                "passed_metrics": passed_metrics,
                "overall_passed": weighted_score >= 0.7  # Default overall threshold
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error evaluating with template: {str(e)}")
            return {
                "error": str(e),
                "scores": {},
                "weighted_score": 0.0
            }


def get_ragas_evaluator(metrics_config: Optional[List[Dict[str, Any]]] = None) -> RagasEvaluator:
    """
    Factory function to create and return a RagasEvaluator instance.
    
    Args:
        metrics_config: Optional list of RAGAS metrics configuration
        
    Returns:
        RagasEvaluator instance
    """
    return RagasEvaluator(metrics_config)


if __name__ == "__main__":
    # Example usage
    evaluator = get_ragas_evaluator()
    
    # Check if RAGAS is available
    if evaluator.is_available():
        # Example evaluation
        question = "What is the capital of France?"
        answer = "The capital of France is Paris."
        contexts = ["Paris is the capital and most populous city of France."]
        
        scores = evaluator.evaluate(question, answer, contexts)
        print("RAGAS Scores:", scores)
        
        weighted_score = evaluator.calculate_weighted_score(scores)
        print("Weighted Score:", weighted_score)
    else:
        print("RAGAS evaluation is not available")
