"""
Configuration for the example test.

This file contains the configuration for the example test.
"""

import os
import json

# Create the example configuration directory
os.makedirs("config", exist_ok=True)

# Create the example configuration file
config = {
    "output_dir": "reports",
    "use_llm_critique": False,  # Set to False for testing without API key
    "llm_config": {
        "model_name": "gpt-4",
        "temperature": 0.0,
        "max_tokens": 1000
    },
    "thresholds": {
        "context_relevance": 0.6,
        "context_utilization": 0.5,
        "answer_similarity": 0.6,
        "keyword_overlap": 0.4,
        "sql_similarity": 0.7
    }
}

# Write the configuration to a file
with open("config/test_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Created example configuration file: config/test_config.json")
