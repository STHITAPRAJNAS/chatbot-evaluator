# Enhanced Chatbot Evaluator

A professional framework for evaluating chatbot services using AWS Bedrock LLM as a judge, with pytest-BDD based testing. Now with support for multi-turn conversations, test client integration, and RAG application evaluation.

## Overview

This project provides a comprehensive evaluation framework for assessing the quality of chatbot responses. It uses AWS Bedrock LLM models as judges to evaluate responses based on customizable criteria defined in an Excel template or directly in feature files. The framework supports both single-turn and multi-turn conversations, and includes specialized metrics for evaluating RAG (Retrieval-Augmented Generation) applications.

Key features:
- AWS Bedrock LLM integration for response evaluation
- Multi-turn conversation support for evaluating conversation coherence
- Test client support for direct integration testing
- RAG-specific evaluation metrics and criteria
- RAGAS metrics integration for comprehensive RAG evaluation
- Flexible input sources (Excel templates or feature files)
- Pytest-BDD based testing framework
- Detailed evaluation reports in text and HTML formats
- Customizable evaluation criteria and thresholds

## Architecture

The framework consists of the following main components:

1. **Enhanced Chatbot Client**: Connects to the chatbot service API or test client to send queries and receive responses, with support for multi-turn conversations
2. **Bedrock LLM Judge**: Uses AWS Bedrock models to evaluate chatbot responses based on defined criteria
3. **RAGAS Evaluator**: Provides specialized metrics for evaluating RAG applications
4. **Enhanced Evaluator**: Coordinates the evaluation process with support for multi-turn conversations and RAG-specific evaluation
5. **BDD Testing Framework**: Provides behavior-driven testing capabilities with support for test client integration

## Installation

### Prerequisites

- Python 3.10+
- AWS account with Bedrock access
- Access to a chatbot service endpoint or test client

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chatbot-evaluator.git
cd chatbot-evaluator
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure AWS credentials:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=your_region
```

4. Create a `.env` file with configuration:
```
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
AWS_REGION=us-east-1
CHATBOT_API_ENDPOINT=http://your-chatbot-endpoint/chat
CHATBOT_API_KEY=your_api_key
```

## Usage

### Basic Usage

To run a complete evaluation using the default template:

```python
from src.enhanced_evaluator import run_enhanced_evaluation

# Run evaluation with default settings
results = run_enhanced_evaluation()
```

### Using a Test Client

To evaluate a chatbot using a test client instead of a real endpoint:

```python
from src.enhanced_evaluator import run_enhanced_evaluation
from your_app import create_test_client

# Create a test client for your application
test_client = create_test_client()

# Run evaluation with the test client
results = run_enhanced_evaluation(test_client=test_client)
```

### Evaluating Multi-turn Conversations

To evaluate multi-turn conversations:

```python
from src.enhanced_evaluator import EnhancedChatbotEvaluator

# Initialize evaluator
evaluator = EnhancedChatbotEvaluator()

# Evaluate a specific conversation from the template
conversation_id = "CONV001"
result = evaluator.evaluate_conversation(conversation_id)

# Save the result
evaluator.save_results(result, "conversation_evaluation.json")
```

### Evaluating RAG Applications

To evaluate a RAG application with context:

```python
from src.enhanced_evaluator import EnhancedChatbotEvaluator

# Initialize evaluator
evaluator = EnhancedChatbotEvaluator()

# Define a RAG question with context
question_data = {
    "question_id": "RAG001",
    "category": "RAG",
    "question": "What is quantum computing?",
    "expected_answer": "Quantum computing uses quantum mechanics to perform calculations.",
    "context": "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations. The devices that perform quantum computations are known as quantum computers.",
    "difficulty": "Medium"
}

# Evaluate the RAG question
result = evaluator.evaluate_single_question(question_data)

# Print the result
print(result)
```

### Using Criteria from Feature Files

To evaluate using criteria defined directly in feature files:

```python
from src.enhanced_evaluator import EnhancedChatbotEvaluator

# Initialize evaluator
evaluator = EnhancedChatbotEvaluator()

# Define criteria
criteria = [
    {
        "name": "Factual Accuracy",
        "description": "The response contains factually correct information.",
        "weight": 0.4,
        "min_score": 0,
        "max_score": 5,
        "passing_threshold": 3
    },
    {
        "name": "Completeness",
        "description": "The response addresses all aspects of the question.",
        "weight": 0.3,
        "min_score": 0,
        "max_score": 5,
        "passing_threshold": 3
    },
    {
        "name": "Clarity",
        "description": "The response is clear and well-structured.",
        "weight": 0.3,
        "min_score": 0,
        "max_score": 5,
        "passing_threshold": 3
    }
]

# Define questions
questions = [
    {
        "question": "What is the capital of France?",
        "expected_answer": "The capital of France is Paris."
    },
    {
        "question": "What is machine learning?",
        "expected_answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
    }
]

# Evaluate using criteria from feature file
result = evaluator.evaluate_from_feature_file(questions)

# Save the result
evaluator.save_results(result, "feature_file_evaluation.json")
```

## Evaluation Templates

The framework supports two types of evaluation templates:

1. **Standard Template**: For basic chatbot evaluation
2. **Enhanced Template**: For multi-turn conversations and RAG evaluation

To create an enhanced template:

```python
from src.templates.create_enhanced_template import create_enhanced_evaluation_template

# Create enhanced template
template_path = create_enhanced_evaluation_template()
```

The enhanced template includes:

- **Questions Sheet**: With additional fields for conversation_id, turn_number, and previous_turns
- **Evaluation_Criteria Sheet**: With RAG-specific and multi-turn specific criteria
- **RAGAS_Metrics Sheet**: With metrics specifically for RAG evaluation
- **Thresholds Sheet**: With thresholds for different evaluation types

## Running Tests

To run the enhanced BDD tests:

```bash
cd chatbot-evaluator
pytest tests/step_defs/test_enhanced_evaluation.py -v
```

To run specific scenarios:

```bash
pytest tests/step_defs/test_enhanced_evaluation.py -v -k "test_client"
```

## Configuration

The enhanced framework can be configured using environment variables or a `.env` file:

- `BEDROCK_MODEL_ID`: The AWS Bedrock model ID to use (default: `anthropic.claude-3-sonnet-20240229-v1:0`)
- `AWS_REGION`: The AWS region for Bedrock (default: `us-east-1`)
- `CHATBOT_API_ENDPOINT`: The chatbot API endpoint URL
- `CHATBOT_API_KEY`: API key for chatbot authentication
- `CHATBOT_API_TIMEOUT`: Request timeout in seconds (default: `30`)
- `EVALUATION_TEMPLATE_PATH`: Path to the evaluation template Excel file
- `CONVERSATION_ID_FIELD`: Field name for conversation ID in requests/responses (default: `conversation_id`)
- `HISTORY_FIELD`: Field name for conversation history in requests/responses (default: `history`)
- `USE_RAGAS`: Whether to use RAGAS evaluation (default: `True`)

## Project Structure

```
chatbot-evaluator/
├── data/
│   ├── chatbot_evaluation_template.xlsx
│   ├── enhanced_chatbot_evaluation_template.xlsx
│   └── results/
├── docs/
├── src/
│   ├── bedrock_judge.py
│   ├── chatbot_client.py
│   ├── config.py
│   ├── evaluator.py
│   ├── enhanced_evaluator.py
│   ├── ragas_evaluator.py
│   └── templates/
│       ├── create_evaluation_template.py
│       └── create_enhanced_template.py
├── tests/
│   ├── features/
│   │   ├── chatbot_evaluation.feature
│   │   └── enhanced_evaluation.feature
│   └── step_defs/
│       ├── test_chatbot_evaluation.py
│       └── test_enhanced_evaluation.py
├── examples/
│   └── usage_examples.py
├── .env
├── README.md
├── requirements.txt
└── todo.md
```

## Key Enhancements

### Multi-turn Conversation Support
- Track conversation state across multiple turns
- Evaluate conversation coherence and context retention
- Support for conversation history management

### Test Client Integration
- Direct integration with application test clients
- Support for both real endpoints and test clients
- Simplified testing workflow

### RAG-specific Evaluation
- Specialized criteria for RAG applications
- Context utilization assessment
- Source attribution evaluation
- RAGAS metrics for comprehensive RAG evaluation

### Flexible Input Sources
- Support for Excel templates
- Support for criteria defined in feature files
- Mix-and-match approach for maximum flexibility

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
