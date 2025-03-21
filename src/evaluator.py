#!/usr/bin/env python3
"""
Chatbot evaluation framework core module.
This module provides the main functionality for evaluating a chatbot service
using AWS Bedrock LLM as a judge.
"""

import os
import json
import time
import logging
import pandas as pd
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from src.config import get_config
from src.bedrock_judge import get_bedrock_judge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatbotEvaluator:
    """
    Main class for evaluating a chatbot service using AWS Bedrock LLM as a judge.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ChatbotEvaluator.
        
        Args:
            config_path: Optional path to a custom configuration file
        """
        # Load configuration
        self.config = get_config()
        
        # Initialize Bedrock judge
        bedrock_config = self.config["bedrock"]
        self.judge = get_bedrock_judge(
            model_id=bedrock_config["model_id"],
            region_name=bedrock_config["region_name"]
        )
        
        # Load evaluation template
        self.template_path = self.config["evaluation"]["template_path"]
        self._load_evaluation_template()
        
        # Initialize results storage
        self.results_dir = Path(self.config["evaluation"]["results_dir"])
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info("ChatbotEvaluator initialized successfully")
    
    def _load_evaluation_template(self):
        """Load the evaluation template from Excel file."""
        try:
            # Load sheets from Excel file
            self.questions_df = pd.read_excel(self.template_path, sheet_name="Questions")
            self.criteria_df = pd.read_excel(self.template_path, sheet_name="Evaluation_Criteria")
            self.thresholds_df = pd.read_excel(self.template_path, sheet_name="Thresholds")
            
            # Try to load RAGAS metrics if available
            try:
                self.ragas_df = pd.read_excel(self.template_path, sheet_name="RAGAS_Metrics")
            except Exception as e:
                logger.warning(f"RAGAS metrics sheet not found: {str(e)}")
                self.ragas_df = None
            
            logger.info(f"Loaded evaluation template from {self.template_path}")
            logger.info(f"Found {len(self.questions_df)} questions and {len(self.criteria_df)} evaluation criteria")
        
        except Exception as e:
            logger.error(f"Failed to load evaluation template: {str(e)}")
            raise
    
    def _prepare_criteria_list(self) -> List[Dict[str, Any]]:
        """
        Convert criteria DataFrame to list of dictionaries.
        
        Returns:
            List of criteria dictionaries
        """
        criteria_list = []
        
        for _, row in self.criteria_df.iterrows():
            criteria_list.append({
                "criteria_id": row["criteria_id"],
                "name": row["name"],
                "description": row["description"],
                "weight": float(row["weight"]),
                "min_score": float(row["min_score"]),
                "max_score": float(row["max_score"]),
                "passing_threshold": float(row["passing_threshold"])
            })
        
        return criteria_list
    
    def _prepare_questions_list(self) -> List[Dict[str, Any]]:
        """
        Convert questions DataFrame to list of dictionaries.
        
        Returns:
            List of question dictionaries
        """
        questions_list = []
        
        for _, row in self.questions_df.iterrows():
            questions_list.append({
                "question_id": row["question_id"],
                "category": row["category"],
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "context": row["context"] if "context" in row and not pd.isna(row["context"]) else "",
                "difficulty": row["difficulty"]
            })
        
        return questions_list
    
    def _get_thresholds(self) -> Dict[str, float]:
        """
        Get evaluation thresholds from the thresholds DataFrame.
        
        Returns:
            Dictionary of threshold values
        """
        thresholds = {}
        
        for _, row in self.thresholds_df.iterrows():
            thresholds[row["name"]] = float(row["value"])
        
        return thresholds
    
    def query_chatbot(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """
        Query the chatbot API with a question.
        
        Args:
            question: The question to ask the chatbot
            
        Returns:
            Tuple of (response_text, full_response_data)
        """
        api_config = self.config["chatbot_api"]
        endpoint = api_config["endpoint"]
        headers = api_config["headers"]
        timeout = api_config["timeout"]
        
        payload = {
            "message": question
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            response_time = time.time() - start_time
            
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the response text - adjust based on actual API response format
            if "response" in response_data:
                response_text = response_data["response"]
            elif "answer" in response_data:
                response_text = response_data["answer"]
            elif "message" in response_data:
                response_text = response_data["message"]
            else:
                response_text = str(response_data)
                logger.warning("Could not find response text in standard fields, using full response")
            
            # Add response time to the response data
            response_data["_response_time"] = response_time
            
            return response_text, response_data
        
        except requests.RequestException as e:
            logger.error(f"Error querying chatbot API: {str(e)}")
            raise
    
    def evaluate_single_question(
        self, 
        question_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate the chatbot on a single question.
        
        Args:
            question_data: Dictionary containing question information
            
        Returns:
            Dictionary with evaluation results
        """
        question_id = question_data["question_id"]
        question_text = question_data["question"]
        expected_answer = question_data["expected_answer"]
        
        logger.info(f"Evaluating question {question_id}: {question_text}")
        
        try:
            # Query the chatbot
            response_text, response_data = self.query_chatbot(question_text)
            
            # Get evaluation criteria
            criteria_list = self._prepare_criteria_list()
            
            # Evaluate the response using Bedrock LLM judge
            evaluation_result = self.judge.evaluate_response(
                question=question_text,
                expected_answer=expected_answer,
                actual_response=response_text,
                criteria=criteria_list
            )
            
            # Add metadata to the result
            result = {
                "question_id": question_id,
                "question": question_text,
                "expected_answer": expected_answer,
                "actual_response": response_text,
                "response_time": response_data.get("_response_time", 0),
                "evaluation": evaluation_result,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error evaluating question {question_id}: {str(e)}")
            return {
                "question_id": question_id,
                "question": question_text,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def evaluate_all_questions(self) -> Dict[str, Any]:
        """
        Evaluate the chatbot on all questions in the template.
        
        Returns:
            Dictionary with evaluation results for all questions
        """
        questions_list = self._prepare_questions_list()
        thresholds = self._get_thresholds()
        
        results = {
            "questions": [],
            "summary": {},
            "thresholds": thresholds,
            "timestamp": datetime.now().isoformat()
        }
        
        total_score = 0
        total_questions = len(questions_list)
        successful_evaluations = 0
        
        for question_data in questions_list:
            try:
                result = self.evaluate_single_question(question_data)
                results["questions"].append(result)
                
                if "evaluation" in result and "weighted_average" in result["evaluation"]:
                    total_score += result["evaluation"]["weighted_average"]
                    successful_evaluations += 1
            
            except Exception as e:
                logger.error(f"Failed to evaluate question {question_data['question_id']}: {str(e)}")
                results["questions"].append({
                    "question_id": question_data["question_id"],
                    "question": question_data["question"],
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Calculate summary statistics
        if successful_evaluations > 0:
            average_score = total_score / successful_evaluations
        else:
            average_score = 0
        
        results["summary"] = {
            "total_questions": total_questions,
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": total_questions - successful_evaluations,
            "average_score": average_score,
            "pass_threshold": thresholds.get("overall_passing_score", 0.7),
            "passed": average_score >= thresholds.get("overall_passing_score", 0.7)
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Evaluation results dictionary
            filename: Optional custom filename
            
        Returns:
            Path to the saved results file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        file_path = self.results_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {file_path}")
        return str(file_path)
    
    def generate_report(self, results: Dict[str, Any], format: str = "text") -> str:
        """
        Generate a human-readable report from evaluation results.
        
        Args:
            results: Evaluation results dictionary
            format: Report format ('text' or 'html')
            
        Returns:
            Report content as string
        """
        if format == "html":
            return self._generate_html_report(results)
        else:
            return self._generate_text_report(results)
    
    def _generate_text_report(self, results: Dict[str, Any]) -> str:
        """Generate a text report from evaluation results."""
        summary = results["summary"]
        questions = results["questions"]
        thresholds = results["thresholds"]
        
        report = []
        report.append("=" * 80)
        report.append("CHATBOT EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Questions: {summary['total_questions']}")
        report.append(f"Successful Evaluations: {summary['successful_evaluations']}")
        report.append(f"Failed Evaluations: {summary['failed_evaluations']}")
        report.append(f"Average Score: {summary['average_score']:.2f}")
        report.append(f"Pass Threshold: {summary['pass_threshold']:.2f}")
        report.append(f"Overall Result: {'PASS' if summary['passed'] else 'FAIL'}")
        report.append("=" * 80)
        report.append("")
        
        for i, question in enumerate(questions, 1):
            report.append(f"Question {i}: {question['question_id']}")
            report.append("-" * 80)
            report.append(f"Category: {question.get('category', 'N/A')}")
            report.append(f"Question: {question['question']}")
            report.append(f"Expected Answer: {question.get('expected_answer', 'N/A')}")
            report.append(f"Actual Response: {question.get('actual_response', 'N/A')}")
            
            if "error" in question:
                report.append(f"Error: {question['error']}")
            elif "evaluation" in question:
                eval_data = question["evaluation"]
                report.append(f"Weighted Average Score: {eval_data.get('weighted_average', 0):.2f}")
                report.append("Criteria Scores:")
                
                for score in eval_data.get("criteria_scores", []):
                    report.append(f"  - {score['name']}: {score['score']} - {score['justification']}")
                
                report.append(f"Overall Feedback: {eval_data.get('overall_feedback', 'N/A')}")
            
            report.append("")
        
        return "\n".join(report)
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate an HTML report from evaluation results."""
        summary = results["summary"]
        questions = results["questions"]
        
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("  <title>Chatbot Evaluation Report</title>")
        html.append("  <style>")
        html.append("    body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("    h1 { color: #333; }")
        html.append("    .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }")
        html.append("    .pass { color: green; font-weight: bold; }")
        html.append("    .fail { color: red; font-weight: bold; }")
        html.append("    .question { border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 5px; }")
        html.append("    .criteria { margin-left: 20px; }")
        html.append("    table { border-collapse: collapse; width: 100%; }")
        html.append("    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("    th { background-color: #f2f2f2; }")
        html.append("  </style>")
        html.append("</head>")
        html.append("<body>")
        
        html.append("  <h1>Chatbot Evaluation Report</h1>")
        html.append("  <div class='summary'>")
        html.append(f"    <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html.append(f"    <p><strong>Total Questions:</strong> {summary['total_questions']}</p>")
        html.append(f"    <p><strong>Successful Evaluations:</strong> {summary['successful_evaluations']}</p>")
        <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>