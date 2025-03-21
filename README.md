# Chatbot Evaluator

A professional framework for evaluating chatbot services using AWS Bedrock LLM as a judge, with pytest-BDD based testing.

## Overview

This project provides a comprehensive evaluation framework for assessing the quality of chatbot responses. It uses AWS Bedrock LLM models as judges to evaluate responses based on customizable criteria defined in an Excel template. The framework also integrates RAGAS metrics for additional evaluation capabilities.

Key features:
- AWS Bedrock LLM integration for response evaluation
- Customizable evaluation criteria via Excel templates
- RAGAS metrics integration for additional evaluation metrics
- Pytest-BDD based testing framework
- Detailed evaluation reports in text and HTML formats
- Flexible configuration options

## Architecture

The framework consists of the following main components:

1. **Bedrock LLM Judge**: Uses AWS Bedrock models to evaluate chatbot responses based on defined criteria
2. **RAGAS Evaluator**: Provides additional evaluation metrics using the RAGAS library
3. **Chatbot Client**: Connects to the chatbot service API to send queries and receive responses
4. **Evaluation Framework**: Coordinates the evaluation process and generates reports
5. **BDD Testing Framework**: Provides behavior-driven testing capabilities

## Installation

### Prerequisites

- Python 3.10+
- AWS account with Bedrock access
- Access to a chatbot service endpoint

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
from src.evaluator import run_evaluation

# Run evaluation with default settings
results = run_evaluation()
```

### Custom Evaluation Template

You can create a custom evaluation template by modifying the Excel file in `data/chatbot_evaluation_template.xlsx` or creating a new one with the same structure.

To run an evaluation with a custom template:

```python
from src.evaluator import run_evaluation

# Run evaluation with custom template
results = run_evaluation(template_path="/path/to/your/template.xlsx")
```

### Evaluating a Single Question

To evaluate a single question:

```python
from src.evaluator import ChatbotEvaluator

# Initialize evaluator
evaluator = ChatbotEvaluator()

# Evaluate a single question
result = evaluator.evaluate_single_question({
    "question_id": "Q001",
    "question": "What is the capital of France?",
    "expected_answer": "The capital of France is Paris."
})

# Print the result
print(result)
```

### Generating Reports

To generate evaluation reports:

```python
from src.evaluator import ChatbotEvaluator

# Initialize evaluator
evaluator = ChatbotEvaluator()

# Run evaluation
results = evaluator.evaluate_all_questions()

# Generate text report
text_report = evaluator.generate_report(results, format="text")
with open("report.txt", "w") as f:
    f.write(text_report)

# Generate HTML report
html_report = evaluator.generate_report(results, format="html")
with open("report.html", "w") as f:
    f.write(html_report)
```

## Evaluation Criteria

The evaluation criteria are defined in the Excel template with the following sheets:

1. **Questions**: Defines the test questions, expected answers, and related metadata
2. **Evaluation_Criteria**: Defines the criteria used by the Bedrock LLM to evaluate responses
3. **RAGAS_Metrics**: Configures RAGAS evaluation metrics and their weights
4. **Thresholds**: Sets passing thresholds for the overall evaluation

### Example Criteria

The framework includes the following default evaluation criteria:

- **Factual Accuracy**: The response contains factually correct information
- **Completeness**: The response addresses all aspects of the question
- **Relevance**: The response is directly relevant to the question asked
- **Clarity**: The response is clear and well-structured
- **Conciseness**: The response is appropriately concise
- **Helpfulness**: The response provides value to the user

## Running Tests

To run the BDD tests:

```bash
cd chatbot-evaluator
pytest tests/step_defs/test_chatbot_evaluation.py -v
```

To generate a test report:

```bash
pytest tests/step_defs/test_chatbot_evaluation.py -v --html=test_report.html
```

## Configuration

The framework can be configured using environment variables or a `.env` file:

- `BEDROCK_MODEL_ID`: The AWS Bedrock model ID to use (default: `anthropic.claude-3-sonnet-20240229-v1:0`)
- `AWS_REGION`: The AWS region for Bedrock (default: `us-east-1`)
- `BEDROCK_TEMPERATURE`: Temperature for model inference (default: `0.0`)
- `BEDROCK_MAX_TOKENS`: Maximum tokens in the response (default: `4096`)
- `CHATBOT_API_ENDPOINT`: The chatbot API endpoint URL
- `CHATBOT_API_KEY`: API key for chatbot authentication
- `CHATBOT_API_TIMEOUT`: Request timeout in seconds (default: `30`)
- `EVALUATION_TEMPLATE_PATH`: Path to the evaluation template Excel file
- `EVALUATION_RESULTS_DIR`: Directory to store evaluation results
- `USE_RAGAS`: Whether to use RAGAS evaluation (default: `True`)

## Project Structure

```
chatbot-evaluator/
├── data/
│   ├── chatbot_evaluation_template.xlsx
│   └── results/
├── docs/
├── src/
│   ├── bedrock_judge.py
│   ├── chatbot_client.py
│   ├── config.py
│   ├── evaluator.py
│   ├── ragas_evaluator.py
│   └── templates/
│       └── create_evaluation_template.py
├── tests/
│   ├── features/
│   │   └── chatbot_evaluation.feature
│   └── step_defs/
│       └── test_chatbot_evaluation.py
├── .env
├── README.md
├── requirements.txt
└── todo.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
