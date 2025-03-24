"""
Data management for the RAG evaluator.

This module provides classes and functions for handling test datasets
and evaluation results.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import json
import os
import pandas as pd
from dataclasses import dataclass, field
import uuid


@dataclass
class EvaluationSample:
    """Base class for evaluation samples."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        return {
            "id": self.id,
            "query": self.query,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, sample_dict: Dict[str, Any]) -> 'EvaluationSample':
        """Create sample from dictionary."""
        return cls(
            id=sample_dict.get("id", str(uuid.uuid4())),
            query=sample_dict.get("query", ""),
            metadata=sample_dict.get("metadata", {})
        )


@dataclass
class RAGEvaluationSample(EvaluationSample):
    """Sample for RAG system evaluation."""
    
    response: str = ""
    contexts: List[str] = field(default_factory=list)
    reference_answer: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "response": self.response,
            "contexts": self.contexts,
            "reference_answer": self.reference_answer
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, sample_dict: Dict[str, Any]) -> 'RAGEvaluationSample':
        """Create sample from dictionary."""
        base_sample = super().from_dict(sample_dict)
        return cls(
            id=base_sample.id,
            query=base_sample.query,
            metadata=base_sample.metadata,
            response=sample_dict.get("response", ""),
            contexts=sample_dict.get("contexts", []),
            reference_answer=sample_dict.get("reference_answer")
        )


@dataclass
class SQLEvaluationSample(EvaluationSample):
    """Sample for text-to-SQL evaluation."""
    
    generated_sql: str = ""
    reference_sql: Optional[str] = None
    expected_result: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "generated_sql": self.generated_sql,
            "reference_sql": self.reference_sql,
            "expected_result": self.expected_result
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, sample_dict: Dict[str, Any]) -> 'SQLEvaluationSample':
        """Create sample from dictionary."""
        base_sample = super().from_dict(sample_dict)
        return cls(
            id=base_sample.id,
            query=base_sample.query,
            metadata=base_sample.metadata,
            generated_sql=sample_dict.get("generated_sql", ""),
            reference_sql=sample_dict.get("reference_sql"),
            expected_result=sample_dict.get("expected_result")
        )


@dataclass
class EvaluationResult:
    """Base class for evaluation results."""
    
    sample_id: str
    metric_name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "sample_id": self.sample_id,
            "metric_name": self.metric_name,
            "score": self.score,
            "details": self.details
        }
    
    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]) -> 'EvaluationResult':
        """Create result from dictionary."""
        return cls(
            sample_id=result_dict.get("sample_id", ""),
            metric_name=result_dict.get("metric_name", ""),
            score=result_dict.get("score", 0.0),
            details=result_dict.get("details", {})
        )


class DataManager:
    """Manager for evaluation data."""
    
    def __init__(self, data_dir: str = None):
        """Initialize data manager.
        
        Args:
            data_dir: Directory to store data files.
        """
        self.data_dir = data_dir or os.path.join(os.getcwd(), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.samples_dir = os.path.join(self.data_dir, "samples")
        self.results_dir = os.path.join(self.data_dir, "results")
        
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.samples = {}
        self.results = {}
    
    def add_sample(self, sample: EvaluationSample) -> None:
        """Add sample to manager."""
        self.samples[sample.id] = sample
    
    def get_sample(self, sample_id: str) -> Optional[EvaluationSample]:
        """Get sample by ID."""
        return self.samples.get(sample_id)
    
    def add_result(self, result: EvaluationResult) -> None:
        """Add result to manager."""
        if result.sample_id not in self.results:
            self.results[result.sample_id] = {}
        self.results[result.sample_id][result.metric_name] = result
    
    def get_result(self, sample_id: str, metric_name: str) -> Optional[EvaluationResult]:
        """Get result by sample ID and metric name."""
        if sample_id in self.results and metric_name in self.results[sample_id]:
            return self.results[sample_id][metric_name]
        return None
    
    def save_samples(self, dataset_name: str) -> None:
        """Save samples to file."""
        filepath = os.path.join(self.samples_dir, f"{dataset_name}.json")
        samples_dict = {sample_id: sample.to_dict() for sample_id, sample in self.samples.items()}
        with open(filepath, 'w') as f:
            json.dump(samples_dict, f, indent=2)
    
    def load_samples(self, dataset_name: str) -> None:
        """Load samples from file."""
        filepath = os.path.join(self.samples_dir, f"{dataset_name}.json")
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            samples_dict = json.load(f)
        
        self.samples = {}
        for sample_id, sample_data in samples_dict.items():
            if "generated_sql" in sample_data:
                sample = SQLEvaluationSample.from_dict(sample_data)
            elif "contexts" in sample_data:
                sample = RAGEvaluationSample.from_dict(sample_data)
            else:
                sample = EvaluationSample.from_dict(sample_data)
            self.samples[sample_id] = sample
    
    def save_results(self, evaluation_name: str) -> None:
        """Save results to file."""
        filepath = os.path.join(self.results_dir, f"{evaluation_name}.json")
        results_dict = {
            sample_id: {
                metric_name: result.to_dict()
                for metric_name, result in sample_results.items()
            }
            for sample_id, sample_results in self.results.items()
        }
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    def load_results(self, evaluation_name: str) -> None:
        """Load results from file."""
        filepath = os.path.join(self.results_dir, f"{evaluation_name}.json")
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            results_dict = json.load(f)
        
        self.results = {}
        for sample_id, sample_results in results_dict.items():
            self.results[sample_id] = {}
            for metric_name, result_data in sample_results.items():
                result = EvaluationResult.from_dict(result_data)
                self.results[sample_id][metric_name] = result
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as pandas DataFrame."""
        data = []
        for sample_id, sample_results in self.results.items():
            for metric_name, result in sample_results.items():
                row = {
                    "sample_id": sample_id,
                    "metric_name": metric_name,
                    "score": result.score
                }
                # Add sample query if available
                if sample_id in self.samples:
                    row["query"] = self.samples[sample_id].query
                
                # Add details as separate columns
                for key, value in result.details.items():
                    if isinstance(value, (str, int, float, bool)):
                        row[f"detail_{key}"] = value
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for evaluation results."""
        df = self.get_results_dataframe()
        if df.empty:
            return {}
        
        summary = {}
        for metric_name in df["metric_name"].unique():
            metric_df = df[df["metric_name"] == metric_name]
            summary[metric_name] = {
                "mean": metric_df["score"].mean(),
                "median": metric_df["score"].median(),
                "min": metric_df["score"].min(),
                "max": metric_df["score"].max(),
                "std": metric_df["score"].std()
            }
        
        return summary
    
    def create_sample_dataset(self, dataset_type: str, num_samples: int = 5) -> None:
        """Create a sample dataset for testing.
        
        Args:
            dataset_type: Type of dataset to create ('rag' or 'sql').
            num_samples: Number of samples to create.
        """
        self.samples = {}
        
        if dataset_type.lower() == 'rag':
            for i in range(num_samples):
                sample = RAGEvaluationSample(
                    query=f"Sample RAG query {i+1}",
                    response=f"Sample RAG response {i+1}",
                    contexts=[f"Sample context {j+1} for query {i+1}" for j in range(3)],
                    reference_answer=f"Sample reference answer {i+1}" if i % 2 == 0 else None
                )
                self.add_sample(sample)
            
            self.save_samples("sample_rag_dataset")
        
        elif dataset_type.lower() == 'sql':
            for i in range(num_samples):
                sample = SQLEvaluationSample(
                    query=f"Sample SQL query {i+1}",
                    generated_sql=f"SELECT * FROM table_{i+1}",
                    reference_sql=f"SELECT id, name FROM table_{i+1}" if i % 2 == 0 else None
                )
                self.add_sample(sample)
            
            self.save_samples("sample_sql_dataset")