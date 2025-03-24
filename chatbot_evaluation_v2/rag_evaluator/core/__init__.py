"""
Core module initialization for the RAG evaluator.

This module initializes the core components of the RAG evaluator.
"""

from rag_evaluator.core.config import (
    EvaluationConfig,
    RAGEvaluationConfig,
    SQLEvaluationConfig,
    ConfigManager
)

from rag_evaluator.core.data import (
    EvaluationSample,
    RAGEvaluationSample,
    SQLEvaluationSample,
    EvaluationResult,
    DataManager
)

from rag_evaluator.core.evaluation import (
    MetricRegistry,
    Evaluator,
    EvaluationManager
)

__all__ = [
    # Config classes
    'EvaluationConfig',
    'RAGEvaluationConfig',
    'SQLEvaluationConfig',
    'ConfigManager',
    
    # Data classes
    'EvaluationSample',
    'RAGEvaluationSample',
    'SQLEvaluationSample',
    'EvaluationResult',
    'DataManager',
    
    # Evaluation classes
    'MetricRegistry',
    'Evaluator',
    'EvaluationManager'
]
