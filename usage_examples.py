#!/usr/bin/env python3
"""
Example script demonstrating how to use the chatbot evaluation framework.
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator import ChatbotEvaluator, run_evaluation
from src.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def example_single_question():
    """Example of evaluating a single question."""
    logger.info("Running single question evaluation example...")
    
    # Initialize evaluator
    evaluator = ChatbotEvaluator()
    
    # Define a test question
    question_data = {
        "question_id": "Q001",
        "category": "General Knowledge",
        "question": "What is the capital of France?",
        "expected_answer": "The capital of France is Paris.",
        "difficulty": "Easy"
    }
    
    # Evaluate the question
    result = evaluator.evaluate_single_question(question_data)
    
    # Save the result
    output_file = os.path.join(
        evaluator.results_dir,
        f"single_question_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Single question evaluation result saved to {output_file}")
    
    return result

def example_full_evaluation():
    """Example of running a full evaluation."""
    logger.info("Running full evaluation example...")
    
    # Run the evaluation
    results = run_evaluation()
    
    logger.info("Full evaluation completed")
    
    return results

def example_custom_template():
    """Example of using a custom evaluation template."""
    logger.info("Running custom template evaluation example...")
    
    # Path to the default template
    default_template = get_config("evaluation")["template_path"]
    
    # Run the evaluation with the template
    results = run_evaluation(template_path=default_template)
    
    logger.info("Custom template evaluation completed")
    
    return results

def example_generate_reports():
    """Example of generating evaluation reports."""
    logger.info("Running report generation example...")
    
    # Initialize evaluator
    evaluator = ChatbotEvaluator()
    
    # Run a simple evaluation
    results = evaluator.evaluate_all_questions()
    
    # Generate text report
    text_report = evaluator.generate_report(results, format="text")
    text_report_path = os.path.join(
        evaluator.results_dir,
        f"text_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    
    with open(text_report_path, 'w') as f:
        f.write(text_report)
    
    logger.info(f"Text report saved to {text_report_path}")
    
    # Generate HTML report
    html_report = evaluator.generate_report(results, format="html")
    html_report_path = os.path.join(
        evaluator.results_dir,
        f"html_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    
    with open(html_report_path, 'w') as f:
        f.write(html_report)
    
    logger.info(f"HTML report saved to {html_report_path}")
    
    return {
        "text_report_path": text_report_path,
        "html_report_path": html_report_path
    }

if __name__ == "__main__":
    print("Chatbot Evaluator Usage Examples")
    print("=" * 40)
    
    # Check if a specific example was requested
    if len(sys.argv) > 1:
        example = sys.argv[1]
        
        if example == "single":
            example_single_question()
        elif example == "full":
            example_full_evaluation()
        elif example == "template":
            example_custom_template()
        elif example == "reports":
            example_generate_reports()
        else:
            print(f"Unknown example: {example}")
            print("Available examples: single, full, template, reports")
    else:
        # Run all examples
        print("Running all examples...")
        
        try:
            # Example 1: Single question evaluation
            print("\n1. Single Question Evaluation")
            print("-" * 40)
            example_single_question()
            
            # Example 2: Full evaluation
            print("\n2. Full Evaluation")
            print("-" * 40)
            example_full_evaluation()
            
            # Example 3: Custom template
            print("\n3. Custom Template Evaluation")
            print("-" * 40)
            example_custom_template()
            
            # Example 4: Generate reports
            print("\n4. Report Generation")
            print("-" * 40)
            example_generate_reports()
            
            print("\nAll examples completed successfully!")
        except Exception as e:
            print(f"Error running examples: {str(e)}")
