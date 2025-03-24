"""
Core configuration management for the RAG evaluator.

This module provides classes and functions for managing configuration settings
for different evaluation scenarios.
"""

from typing import Dict, Any, List, Optional, Union
import json
import os
from dataclasses import dataclass, field


@dataclass
class EvaluationConfig:
    """Base configuration for evaluation scenarios."""
    
    name: str
    description: str = ""
    metrics: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "metrics": self.metrics,
            "thresholds": self.thresholds
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EvaluationConfig':
        """Create configuration from dictionary."""
        return cls(
            name=config_dict.get("name", ""),
            description=config_dict.get("description", ""),
            metrics=config_dict.get("metrics", []),
            thresholds=config_dict.get("thresholds", {})
        )
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'EvaluationConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class RAGEvaluationConfig(EvaluationConfig):
    """Configuration specific to RAG system evaluation."""
    
    retrieval_metrics: List[str] = field(default_factory=list)
    generation_metrics: List[str] = field(default_factory=list)
    llm_critique_enabled: bool = False
    llm_critique_model: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "retrieval_metrics": self.retrieval_metrics,
            "generation_metrics": self.generation_metrics,
            "llm_critique_enabled": self.llm_critique_enabled,
            "llm_critique_model": self.llm_critique_model
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGEvaluationConfig':
        """Create configuration from dictionary."""
        base_config = super().from_dict(config_dict)
        return cls(
            name=base_config.name,
            description=base_config.description,
            metrics=base_config.metrics,
            thresholds=base_config.thresholds,
            retrieval_metrics=config_dict.get("retrieval_metrics", []),
            generation_metrics=config_dict.get("generation_metrics", []),
            llm_critique_enabled=config_dict.get("llm_critique_enabled", False),
            llm_critique_model=config_dict.get("llm_critique_model", "")
        )


@dataclass
class SQLEvaluationConfig(EvaluationConfig):
    """Configuration specific to text-to-SQL evaluation."""
    
    sql_equivalence_check: bool = True
    execution_check: bool = True
    database_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "sql_equivalence_check": self.sql_equivalence_check,
            "execution_check": self.execution_check,
            "database_path": self.database_path
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SQLEvaluationConfig':
        """Create configuration from dictionary."""
        base_config = super().from_dict(config_dict)
        return cls(
            name=base_config.name,
            description=base_config.description,
            metrics=base_config.metrics,
            thresholds=base_config.thresholds,
            sql_equivalence_check=config_dict.get("sql_equivalence_check", True),
            execution_check=config_dict.get("execution_check", True),
            database_path=config_dict.get("database_path", "")
        )


class ConfigManager:
    """Manager for evaluation configurations."""
    
    def __init__(self, config_dir: str = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory to store configuration files.
        """
        self.config_dir = config_dir or os.path.join(os.getcwd(), "configs")
        os.makedirs(self.config_dir, exist_ok=True)
        self.configs = {}
    
    def add_config(self, config: EvaluationConfig) -> None:
        """Add configuration to manager."""
        self.configs[config.name] = config
    
    def get_config(self, name: str) -> Optional[EvaluationConfig]:
        """Get configuration by name."""
        return self.configs.get(name)
    
    def save_all(self) -> None:
        """Save all configurations to files."""
        for name, config in self.configs.items():
            filepath = os.path.join(self.config_dir, f"{name}.json")
            config.save(filepath)
    
    def load_all(self) -> None:
        """Load all configurations from files."""
        for filename in os.listdir(self.config_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.config_dir, filename)
                name = os.path.splitext(filename)[0]
                
                # Determine config type based on content
                with open(filepath, 'r') as f:
                    config_dict = json.load(f)
                
                if "sql_equivalence_check" in config_dict:
                    config = SQLEvaluationConfig.load(filepath)
                elif "retrieval_metrics" in config_dict:
                    config = RAGEvaluationConfig.load(filepath)
                else:
                    config = EvaluationConfig.load(filepath)
                
                self.configs[name] = config
    
    def create_default_configs(self) -> None:
        """Create default configurations."""
        # Default RAG evaluation config
        rag_config = RAGEvaluationConfig(
            name="default_rag",
            description="Default configuration for RAG evaluation",
            metrics=["context_precision", "faithfulness", "answer_relevancy"],
            thresholds={"context_precision": 0.7, "faithfulness": 0.8, "answer_relevancy": 0.7},
            retrieval_metrics=["context_precision", "context_recall"],
            generation_metrics=["faithfulness", "answer_relevancy"],
            llm_critique_enabled=True,
            llm_critique_model="gpt-4"
        )
        self.add_config(rag_config)
        
        # Default SQL evaluation config
        sql_config = SQLEvaluationConfig(
            name="default_sql",
            description="Default configuration for text-to-SQL evaluation",
            metrics=["sql_equivalence", "execution_accuracy"],
            thresholds={"sql_equivalence": 0.8, "execution_accuracy": 0.9},
            sql_equivalence_check=True,
            execution_check=True
        )
        self.add_config(sql_config)