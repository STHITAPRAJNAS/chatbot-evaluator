#!/usr/bin/env python3
"""
Configuration module for the chatbot evaluation framework.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS Bedrock Configuration
BEDROCK_CONFIG = {
    "model_id": os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"),
    "region_name": os.getenv("AWS_REGION", "us-east-1"),
    "temperature": float(os.getenv("BEDROCK_TEMPERATURE", "0.0")),
    "max_tokens": int(os.getenv("BEDROCK_MAX_TOKENS", "4096"))
}

# Chatbot API Configuration
CHATBOT_API_CONFIG = {
    "endpoint": os.getenv("CHATBOT_API_ENDPOINT", "http://localhost:8000/chat"),
    "timeout": int(os.getenv("CHATBOT_API_TIMEOUT", "30")),
    "headers": {
        "Content-Type": "application/json",
        "Authorization": os.getenv("CHATBOT_API_KEY", "")
    }
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "template_path": os.getenv(
        "EVALUATION_TEMPLATE_PATH", 
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                    "data", "chatbot_evaluation_template.xlsx")
    ),
    "results_dir": os.getenv(
        "EVALUATION_RESULTS_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "results")
    ),
    "use_ragas": os.getenv("USE_RAGAS", "True").lower() in ("true", "1", "yes")
}

# Create results directory if it doesn't exist
os.makedirs(EVALUATION_CONFIG["results_dir"], exist_ok=True)

def get_config(section: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration settings.
    
    Args:
        section: Optional section name to retrieve specific config
        
    Returns:
        Dict containing configuration settings
    """
    config = {
        "bedrock": BEDROCK_CONFIG,
        "chatbot_api": CHATBOT_API_CONFIG,
        "evaluation": EVALUATION_CONFIG
    }
    
    if section:
        return config.get(section, {})
    
    return config
