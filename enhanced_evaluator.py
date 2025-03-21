#!/usr/bin/env python3
"""
Enhanced evaluator for RAG applications with multi-turn conversation support.
This module provides the main functionality for evaluating any RAG-based chatbot service
using AWS Bedrock LLM as a judge, with support for multi-turn conversations.
"""

import os
import json
import time
import logging
import pandas as pd
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

from src.config import get_config
from src.bedrock_judge import get_bedrock_judge
from src.ragas_evaluator import get_ragas_evaluator
from src.chatbot_client import get_chatbot_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedChatbotEvaluator:
    """
    Enhanced class for evaluating any RAG-based chatbot service using AWS Bedrock LLM as a judge,
    with support for multi-turn conversations and flexible input sources.
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        template_path: Optional[str] = None,
        test_client=None
    ):
        """
        Initialize the EnhancedChatbotEvaluator.
        
        Args:
            config_path: Optional path to a custom configuration file
            template_path: Optional path to a custom evaluation template
            test_client: Optional test client for direct integration testing
        """
        # Load configuration
        self.config = get_config()
        
        # Override template path if provided
        if template_path:
            self.config["evaluation"]["template_path"] = template_path
        
        # Initialize Bedrock judge
        bedrock_config = self.config["bedrock"]
        self.judge = get_bedrock_judge(
            model_id=bedrock_config["model_id"],
            region_name=bedrock_config["region_name"]
        )
        
        # Initialize RAGAS evaluator
        self.ragas_evaluator = get_ragas_evaluator()
        
        # Initialize chatbot client
        chatbot_config = self.config["chatbot_api"]
        self.chatbot_client = get_chatbot_client(
            endpoint=chatbot_config["endpoint"],
            api_key=chatbot_config.get("headers", {}).get("Authorization", ""),
            test_client=test_client
        )
        
        # Load evaluation template
        self.template_path = self.config["evaluation"]["template_path"]
        self._load_evaluation_template()
        
        # Initialize results storage
        self.results_dir = Path(self.config["evaluation"]["results_dir"])
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Track active conversations
        self.active_conversations = {}
        
        logger.info("EnhancedChatbotEvaluator initialized successfully")
    
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
            
            # Check for multi-turn conversations
            if 'conversation_id' in self.questions_df.columns:
                conversation_ids = self.questions_df['conversation_id'].unique()
                multi_turn_conversations = [id for id in conversation_ids if id and not pd.isna(id)]
                if multi_turn_conversations:
                    logger.info(f"Found {len(multi_turn_conversations)} multi-turn conversations in the template")
        
        except Exception as e:
            logger.error(f"Failed to load evaluation template: {str(e)}")
            raise
    
    def _prepare_criteria_list(self, evaluation_type: str = "all") -> List[Dict[str, Any]]:
        """
        Convert criteria DataFrame to list of dictionaries, filtered by evaluation type.
        
        Args:
            evaluation_type: Type of evaluation ('all', 'rag', or 'multi-turn')
            
        Returns:
            List of criteria dictionaries
        """
        criteria_list = []
        
        for _, row in self.criteria_df.iterrows():
            # Check if this criterion applies to the current evaluation type
            applies_to = row.get("applies_to", "all")
            if applies_to != "all" and applies_to != evaluation_type:
                continue
                
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
    
    def _prepare_questions_list(self, filter_conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Convert questions DataFrame to list of dictionaries, optionally filtered by conversation ID.
        
        Args:
            filter_conversation_id: Optional conversation ID to filter questions
            
        Returns:
            List of question dictionaries
        """
        questions_list = []
        
        for _, row in self.questions_df.iterrows():
            # If filtering by conversation ID and this question doesn't match, skip it
            if filter_conversation_id is not None:
                row_conv_id = row.get("conversation_id", "")
                if pd.isna(row_conv_id) or row_conv_id != filter_conversation_id:
                    continue
            
            question_data = {
                "question_id": row["question_id"],
                "category": row["category"],
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "context": row["context"] if "context" in row and not pd.isna(row["context"]) else "",
                "difficulty": row["difficulty"]
            }
            
            # Add multi-turn conversation fields if available
            if "conversation_id" in row and not pd.isna(row["conversation_id"]):
                question_data["conversation_id"] = row["conversation_id"]
                question_data["turn_number"] = int(row["turn_number"]) if "turn_number" in row else 1
                question_data["previous_turns"] = row["previous_turns"] if "previous_turns" in row and not pd.isna(row["previous_turns"]) else ""
            
            questions_list.append(question_data)
        
        # Sort by turn number if this is a multi-turn conversation
        if filter_conversation_id is not None:
            questions_list.sort(key=lambda q: q.get("turn_number", 1))
        
        return questions_list
    
    def _get_thresholds(self, evaluation_type: str = "all") -> Dict[str, float]:
        """
        Get evaluation thresholds from the thresholds DataFrame, filtered by evaluation type.
        
        Args:
            evaluation_type: Type of evaluation ('all', 'rag', or 'multi-turn')
            
        Returns:
            Dictionary of threshold values
        """
        thresholds = {}
        
        for _, row in self.thresholds_df.iterrows():
            # Check if this threshold applies to the current evaluation type
            applies_to = row.get("applies_to", "all")
            if applies_to != "all" and applies_to != evaluation_type:
                continue
                
            thresholds[row["name"]] = float(row["value"])
        
        return thresholds
    
    def _get_ragas_metrics(self, evaluation_type: str = "all") -> Dict[str, Dict[str, Any]]:
        """
        Get RAGAS metrics from the RAGAS metrics DataFrame, filtered by evaluation type.
        
        Args:
            evaluation_type: Type of evaluation ('all', 'rag', or 'multi-turn')
            
        Returns:
            Dictionary of RAGAS metrics
        """
        if self.ragas_df is None:
            return {}
            
        metrics = {}
        
        for _, row in self.ragas_df.iterrows():
            # Check if this metric applies to the current evaluation type
            applies_to = row.get("applies_to", "all")
            if applies_to != "all" and applies_to != evaluation_type:
                continue
                
            metrics[row["name"]] = {
                "metric_id": row["metric_id"],
                "description": row["description"],
                "weight": float(row["weight"]),
                "passing_threshold": float(row["passing_threshold"])
            }
        
        return metrics
    
    def evaluate_single_question(
        self, 
        question_data: Dict[str, Any],
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the chatbot on a single question.
        
        Args:
            question_data: Dictionary containing question information
            conversation_id: Optional conversation ID for multi-turn evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        question_id = question_data["question_id"]
        question_text = question_data["question"]
        expected_answer = question_data["expected_answer"]
        
        # Determine if this is part of a multi-turn conversation
        is_multi_turn = False
        turn_number = 1
        
        if "conversation_id" in question_data and question_data["conversation_id"]:
            is_multi_turn = True
            conversation_id = question_data["conversation_id"]
            turn_number = question_data.get("turn_number", 1)
            
        logger.info(f"Evaluating question {question_id}: {question_text}")
        if is_multi_turn:
            logger.info(f"This is turn {turn_number} of conversation {conversation_id}")
        
        try:
            # Query the chatbot
            if is_multi_turn:
                # For multi-turn, use the conversation_id
                response_text = self.chatbot_client.query_and_extract(
                    question_text, 
                    conversation_id=conversation_id
                )
            else:
                # For single-turn, just send the question
                response_text = self.chatbot_client.query_and_extract(question_text)
            
            # Determine evaluation type
            evaluation_type = "all"
            if is_multi_turn:
                evaluation_type = "multi-turn"
            if "context" in question_data and question_data["context"]:
                evaluation_type = "rag"
            
            # Get evaluation criteria based on evaluation type
            criteria_list = self._prepare_criteria_list(evaluation_type)
            
            # Evaluate the response using Bedrock LLM judge
            evaluation_result = self.judge.evaluate_response(
                question=question_text,
                expected_answer=expected_answer,
                actual_response=response_text,
                criteria=criteria_list
            )
            
            # If this is a RAG evaluation and we have context, evaluate with RAGAS
            ragas_result = None
            if evaluation_type == "rag" and self.ragas_evaluator.is_available():
                context = question_data.get("context", "")
                if context:
                    contexts = [context]
                    ragas_metrics = self._get_ragas_metrics(evaluation_type)
                    
                    if ragas_metrics:
                        ragas_result = self.ragas_evaluator.evaluate(
                            question=question_text,
                            answer=response_text,
                            contexts=contexts,
                            metrics=list(ragas_metrics.keys())
                        )
            
            # Add metadata to the result
            result = {
                "question_id": question_id,
                "question": question_text,
                "expected_answer": expected_answer,
                "actual_response": response_text,
                "evaluation": evaluation_result,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add multi-turn metadata if applicable
            if is_multi_turn:
                result["conversation_id"] = conversation_id
                result["turn_number"] = turn_number
                
                # Get conversation history
                history = self.chatbot_client.get_conversation_history(conversation_id)
                result["conversation_history"] = history
            
            # Add RAGAS results if available
            if ragas_result:
                result["ragas_evaluation"] = ragas_result
            
            return result
        
        except Exception as e:
            logger.error(f"Error evaluating question {question_id}: {str(e)}")
            return {
                "question_id": question_id,
                "question": question_text,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def evaluate_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Evaluate a multi-turn conversation.
        
        Args:
            conversation_id: The conversation ID to evaluate
            
        Returns:
            Dictionary with evaluation results for the conversation
        """
        # Get all questions for this conversation
        questions = self._prepare_questions_list(filter_conversation_id=conversation_id)
        
        if not questions:
            logger.warning(f"No questions found for conversation {conversation_id}")
            return {
                "conversation_id": conversation_id,
                "error": "No questions found for this conversation",
                "timestamp": datetime.now().isoformat()
            }
        
        logger.info(f"Evaluating conversation {conversation_id} with {len(questions)} turns")
        
        # Start a new conversation with the chatbot client
        client_conversation_id = self.chatbot_client.start_conversation()
        
        results = []
        total_score = 0
        successful_evaluations = 0
        
        for question_data in questions:
            try:
                # Evaluate this turn
                result = self.evaluate_single_question(
                    question_data=question_data,
                    conversation_id=client_conversation_id
                )
                
                results.append(result)
                
                if "evaluation" in result and "weighted_average" in result["evaluation"]:
                    total_score += result["evaluation"]["weighted_average"]
                    successful_evaluations += 1
            
            except Exception as e:
                logger.error(f"Failed to evaluate turn {question_data.get('turn_number', '?')} of conversation {conversation_id}: {str(e)}")
                results.append({
                    "question_id": question_data["question_id"],
                    "question": question_data["question"],
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Calculate average score
        average_score = total_score / successful_evaluations if successful_evaluations > 0 else 0
        
        # Get thresholds for multi-turn evaluation
        thresholds = self._get_thresholds(evaluation_type="multi-turn")
        passing_threshold = thresholds.get("overall_passing_score", 0.7)
        
        # Prepare conversation result
        conversation_result = {
            "conversation_id": conversation_id,
            "turns": results,
            "summary": {
                "total_turns": len(questions),
                "successful_evaluations": successful_evaluations,
                "failed_evaluations": len(questions) - successful_evaluations,
                "average_score": average_score,
                "pass_threshold": passing_threshold,
                "passed": average_score >= passing_threshold
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # End the conversation with the chatbot client
        self.chatbot_client.end_conversation(client_conversation_id)
        
        return conversation_result
    
    def evaluate_all_questions(self) -> Dict[str, Any]:
        """
        Evaluate the chatbot on all questions in the template.
        
        Returns:
            Dictionary with evaluation results for all questions
        """
        # Get all questions
        questions_list = self._prepare_questions_list()
        
        # Identify single questions and conversations
        single_questions = []
        conversation_ids = set()
        
        for question in questions_list:
            if "conversation_id" in question and question["conversation_id"]:
                conversation_ids.add(question["conversation_id"])
            else:
                single_questions.append(question)
        
        logger.info(f"Found {len(single_questions)} single questions and {len(conversation_ids)} conversations to evaluate")
        
        # Initialize results
        results = {
            "single_questions": [],
            "conversations": [],
            "summary": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Evaluate single questions
        total_score = 0
        successful_evaluations = 0
        
        for question_data in single_questions:
            try:
                result = self.evaluate_single_question(question_data)
                results["single_questions"].append(result)
                
                if "evaluation" in result and "weighted_average" in result["evaluation"]:
                    total_score += result["evaluation"]["weighted_average"]
                    successful_evaluations += 1
            
            except Exception as e:
                logger.error(f"Failed to evaluate question {question_data['question_id']}: {str(e)}")
                results["single_questions"].append({
                    "question_id": question_data["question_id"],
                    "question": question_data["question"],
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Evaluate conversations
        conversation_scores = []
        
        for conversation_id in conversation_ids:
            try:
                conversation_result = self.evaluate_conversation(conversation_id)
                results["conversations"].append(conversation_result)
                
                if "summary" in conversation_result and "average_score" in conversation_result["summary"]:
                    conversation_scores.append(conversation_result["summary"]["average_score"])
            
            except Exception as e:
                logger.error(f"Failed to evaluate conversation {conversation_id}: {str(e)}")
                results["conversations"].append({
                    "conversation_id": conversation_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Calculate overall statistics
        thresholds = self._get_thresholds()
        
        # Calculate average score for single questions
        single_question_avg = total_score / successful_evaluations if successful_evaluations > 0 else 0
        
        # Calculate average score for conversations
        conversation_avg = sum(conversation_scores) / len(conversation_scores) if conversation_scores else 0
        
        # Calculate overall average (weighted equally between single questions and conversations)
        if successful_evaluations > 0 and conversation_scores:
            overall_avg = (single_question_avg + conversation_avg) / 2
        elif successful_evaluations > 0:
            overall_avg = single_question_avg
        elif conversation_scores:
            overall_avg = conversation_avg
        else:
            overall_avg = 0
        
        # Determine if the evaluation passed
        passing_threshold = thresholds.get("overall_passing_score", 0.7)
        
        results["summary"] = {
            "single_questions": {
                "total": len(single_questions),
                "successful": successful_evaluations,
                "failed": len(single_questions) - successful_evaluations,
                "average_score": single_question_avg
            },
            "conversations": {
                "total": len(conversation_ids),
                "successful": len(conversation_scores),
                "failed": len(conversation_ids) - len(conversation_scores),
                "average_score": conversation_avg
            },
            "overall": {
                "average_score": overall_avg,
                "pass_threshold": passing_threshold,
                "passed": overall_avg >= passing_threshold
            }
        }
        
        return results
    
    def evaluate_from_feature_file(
        self, 
        questions: List[Dict[str, Any]],
        is_multi_turn: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate the chatbot using questions defined in a feature file.
        
        Args:
            questions: List of question dictionaries from feature file
            is_multi_turn: Whether this is a multi-turn conversation
            
        Returns:
            Dictionary with evaluation results
        """
        if is_multi_turn:
            # For multi-turn, treat as a conversation
            conversation_id = f"feature_file_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Start a new conversation with the chatbot client
            client_conversation_id = self.chatbot_client.start_conversation()
            
            results = []
            total_score = 0
            successful_evaluations = 0
            
            for i, question_data in enumerate(questions, 1):
                # Add turn number if not present
                if "turn_number" not in question_data:
                    question_data["turn_number"] = i
                
                # Add conversation ID if not present
                if "conversation_id" not in question_data:
                    question_data["conversation_id"] = conversation_id
                
                try:
                    # Evaluate this turn
                    result = self.evaluate_single_question(
                        question_data=question_data,
                        conversation_id=client_conversation_id
                    )
                    
                    results.append(result)
                    
                    if "evaluation" in result and "weighted_average" in result["evaluation"]:
                        total_score += result["evaluation"]["weighted_average"]
                        successful_evaluations += 1
                
                except Exception as e:
                    logger.error(f"Failed to evaluate turn {i} of feature file conversation: {str(e)}")
                    results.append({
                        "question_id": question_data.get("question_id", f"Q{i}"),
                        "question": question_data["question"],
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Calculate average score
            average_score = total_score / successful_evaluations if successful_evaluations > 0 else 0
            
            # Get thresholds for multi-turn evaluation
            thresholds = self._get_thresholds(evaluation_type="multi-turn")
            passing_threshold = thresholds.get("overall_passing_score", 0.7)
            
            # Prepare conversation result
            conversation_result = {
                "conversation_id": conversation_id,
                "turns": results,
                "summary": {
                    "total_turns": len(questions),
                    "successful_evaluations": successful_evaluations,
                    "failed_evaluations": len(questions) - successful_evaluations,
                    "average_score": average_score,
                    "pass_threshold": passing_threshold,
                    "passed": average_score >= passing_threshold
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # End the conversation with the chatbot client
            self.chatbot_client.end_conversation(client_conversation_id)
            
            return conversation_result
        
        else:
            # For single questions, evaluate each independently
            results = {
                "questions": [],
                "summary": {},
                "timestamp": datetime.now().isoformat()
            }
            
            total_score = 0
            successful_evaluations = 0
            
            for i, question_data in enumerate(questions, 1):
                # Add question_id if not present
                if "question_id" not in question_data:
                    question_data["question_id"] = f"F{i}"
                
                try:
                    result = self.evaluate_single_question(question_data)
                    results["questions"].append(result)
                    
                    if "evaluation" in result and "weighted_average" in result["evaluation"]:
                        total_score += result["evaluation"]["weighted_average"]
                        successful_evaluations += 1
                
                except Exception as e:
                    logger.error(f"Failed to evaluate question {question_data.get('question_id', f'F{i}')}: {str(e)}")
                    results["questions"].append({
                        "question_id": question_data.get("question_id", f"F{i}"),
                        "question": question_data["question"],
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Calculate summary statistics
            if successful_evaluations > 0:
                average_score = total_score / successful_evaluations
            else:
                average_score = 0
            
            thresholds = self._get_thresholds()
            passing_threshold = thresholds.get("overall_passing_score", 0.7)
            
            results["summary"] = {
                "total_questions": len(questions),
                "successful_evaluations": successful_evaluations,
                "failed_evaluations": len(questions) - successful_evaluations,
                "average_score": average_score,
                "pass_threshold": passing_threshold,
                "passed": average_score >= passing_threshold
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
        report = []
        report.append("=" * 80)
        report.append("ENHANCED CHATBOT EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Add summary section
        if "summary" in results:
            summary = results["summary"]
            
            # For enhanced evaluation with single questions and conversations
            if "single_questions" in summary and "conversations" in summary:
                report.append("\nSUMMARY:")
                report.append("-" * 40)
                
                # Single questions summary
                sq = summary["single_questions"]
                report.append(f"Single Questions: {sq['total']} total, {sq['successful']} successful, {sq['failed']} failed")
                report.append(f"Single Questions Average Score: {sq['average_score']:.2f}")
                
                # Conversations summary
                conv = summary["conversations"]
                report.append(f"Conversations: {conv['total']} total, {conv['successful']} successful, {conv['failed']} failed")
                report.append(f"Conversations Average Score: {conv['average_score']:.2f}")
                
                # Overall summary
                overall = summary["overall"]
                report.append(f"Overall Average Score: {overall['average_score']:.2f}")
                report.append(f"Pass Threshold: {overall['pass_threshold']:.2f}")
                report.append(f"Overall Result: {'PASS' if overall['passed'] else 'FAIL'}")
            
            # For single evaluation (from feature file or single conversation)
            else:
                report.append("\nSUMMARY:")
                report.append("-" * 40)
                report.append(f"Total Questions: {summary.get('total_questions', summary.get('total_turns', 0))}")
                report.append(f"Successful Evaluations: {summary.get('successful_evaluations', 0)}")
                report.append(f"Failed Evaluations: {summary.get('failed_evaluations', 0)}")
                report.append(f"Average Score: {summary.get('average_score', 0):.2f}")
                report.append(f"Pass Threshold: {summary.get('pass_threshold', 0):.2f}")
                report.append(f"Overall Result: {'PASS' if summary.get('passed', False) else 'FAIL'}")
        
        report.append("=" * 80)
        report.append("")
        
        # Add single questions section
        if "single_questions" in results:
            report.append("\nSINGLE QUESTIONS:")
            report.append("=" * 80)
            
            for i, question in enumerate(results["single_questions"], 1):
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
                
                # Add RAGAS evaluation if available
                if "ragas_evaluation" in question:
                    report.append("\nRAGAS Metrics:")
                    for metric, score in question["ragas_evaluation"].items():
                        report.append(f"  - {metric}: {score:.2f}")
                
                report.append("")
        
        # Add conversations section
        if "conversations" in results:
            report.append("\nCONVERSATIONS:")
            report.append("=" * 80)
            
            for i, conversation in enumerate(results["conversations"], 1):
                report.append(f"Conversation {i}: {conversation['conversation_id']}")
                report.append("-" * 80)
                
                if "error" in conversation:
                    report.append(f"Error: {conversation['error']}")
                    continue
                
                # Add conversation summary
                if "summary" in conversation:
                    summary = conversation["summary"]
                    report.append(f"Total Turns: {summary.get('total_turns', 0)}")
                    report.append(f"Successful Evaluations: {summary.get('successful_evaluations', 0)}")
                    report.append(f"Failed Evaluations: {summary.get('failed_evaluations', 0)}")
                    report.append(f"Average Score: {summary.get('average_score', 0):.2f}")
                    report.append(f"Pass Threshold: {summary.get('pass_threshold', 0):.2f}")
                    report.append(f"Overall Result: {'PASS' if summary.get('passed', False) else 'FAIL'}")
                
                # Add individual turns
                if "turns" in conversation:
                    for j, turn in enumerate(conversation["turns"], 1):
                        report.append(f"\nTurn {j}: {turn['question_id']}")
                        report.append(f"Question: {turn['question']}")
                        report.append(f"Expected Answer: {turn.get('expected_answer', 'N/A')}")
                        report.append(f"Actual Response: {turn.get('actual_response', 'N/A')}")
                        
                        if "error" in turn:
                            report.append(f"Error: {turn['error']}")
                        elif "evaluation" in turn:
                            eval_data = turn["evaluation"]
                            report.append(f"Weighted Average Score: {eval_data.get('weighted_average', 0):.2f}")
                            report.append("Criteria Scores:")
                            
                            for score in eval_data.get("criteria_scores", []):
                                report.append(f"  - {score['name']}: {score['score']} - {score['justification']}")
                
                report.append("")
        
        # Add questions section for feature file evaluation
        if "questions" in results:
            report.append("\nQUESTIONS:")
            report.append("=" * 80)
            
            for i, question in enumerate(results["questions"], 1):
                report.append(f"Question {i}: {question['question_id']}")
                report.append("-" * 80)
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
        
        # Add turns section for single conversation evaluation
        if "turns" in results:
            report.append("\nCONVERSATION TURNS:")
            report.append("=" * 80)
            
            for i, turn in enumerate(results["turns"], 1):
                report.append(f"Turn {i}: {turn['question_id']}")
                report.append("-" * 80)
                report.append(f"Question: {turn['question']}")
                report.append(f"Expected Answer: {turn.get('expected_answer', 'N/A')}")
                report.append(f"Actual Response: {turn.get('actual_response', 'N/A')}")
                
                if "error" in turn:
                    report.append(f"Error: {turn['error']}")
                elif "evaluation" in turn:
                    eval_data = turn["evaluation"]
                    report.append(f"Weighted Average Score: {eval_data.get('weighted_average', 0):.2f}")
                    report.append("Criteria Scores:")
                    
                    for score in eval_data.get("criteria_scores", []):
                        report.append(f"  - {score['name']}: {score['score']} - {score['justification']}")
                
                report.append("")
        
        return "\n".join(report)
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate an HTML report from evaluation results."""
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("  <title>Enhanced Chatbot Evaluation Report</title>")
        html.append("  <style>")
        html.append("    body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("    h1, h2, h3 { color: #333; }")
        html.append("    .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }")
        html.append("    .pass { color: green; font-weight: bold; }")
        html.append("    .fail { color: red; font-weight: bold; }")
        html.append("    .question, .turn, .conversation { border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 5px; }")
        html.append("    .criteria { margin-left: 20px; }")
        html.append("    table { border-collapse: collapse; width: 100%; }")
        html.append("    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("    th { background-color: #f2f2f2; }")
        html.append("    .conversation-header { background-color: #e6f2ff; padding: 10px; margin-bottom: 10px; border-radius: 5px; }")
        html.append("    .turn { background-color: #f9f9f9; }")
        html.append("    .ragas { background-color: #fff8e6; padding: 10px; margin-top: 10px; border-radius: 5px; }")
        html.append("  </style>")
        html.append("</head>")
        html.append("<body>")
        
        html.append("  <h1>Enhanced Chatbot Evaluation Report</h1>")
        html.append(f"  <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Add summary section
        if "summary" in results:
            summary = results["summary"]
            html.append("  <div class='summary'>")
            html.append("    <h2>Summary</h2>")
            
            # For enhanced evaluation with single questions and conversations
            if "single_questions" in summary and "conversations" in summary:
                # Single questions summary
                sq = summary["single_questions"]
                html.append("    <h3>Single Questions</h3>")
                html.append(f"    <p><strong>Total:</strong> {sq['total']}</p>")
                html.append(f"    <p><strong>Successful:</strong> {sq['successful']}</p>")
                html.append(f"    <p><strong>Failed:</strong> {sq['failed']}</p>")
                html.append(f"    <p><strong>Average Score:</strong> {sq['average_score']:.2f}</p>")
                
                # Conversations summary
                conv = summary["conversations"]
                html.append("    <h3>Conversations</h3>")
                html.append(f"    <p><strong>Total:</strong> {conv['total']}</p>")
                html.append(f"    <p><strong>Successful:</strong> {conv['successful']}</p>")
                html.append(f"    <p><strong>Failed:</strong> {conv['failed']}</p>")
                html.append(f"    <p><strong>Average Score:</strong> {conv['average_score']:.2f}</p>")
                
                # Overall summary
                overall = summary["overall"]
                html.append("    <h3>Overall</h3>")
                html.append(f"    <p><strong>Average Score:</strong> {overall['average_score']:.2f}</p>")
                html.append(f"    <p><strong>Pass Threshold:</strong> {overall['pass_threshold']:.2f}</p>")
                html.append(f"    <p><strong>Overall Result:</strong> <span class=\"{'pass' if overall['passed'] else 'fail'}\">{'PASS' if overall['passed'] else 'FAIL'}</span></p>")
            
            # For single evaluation (from feature file or single conversation)
            else:
                html.append(f"    <p><strong>Total Questions/Turns:</strong> {summary.get('total_questions', summary.get('total_turns', 0))}</p>")
                html.append(f"    <p><strong>Successful Evaluations:</strong> {summary.get('successful_evaluations', 0)}</p>")
                html.append(f"    <p><strong>Failed Evaluations:</strong> {summary.get('failed_evaluations', 0)}</p>")
                html.append(f"    <p><strong>Average Score:</strong> {summary.get('average_score', 0):.2f}</p>")
                html.append(f"    <p><strong>Pass Threshold:</strong> {summary.get('pass_threshold', 0):.2f}</p>")
                html.append(f"    <p><strong>Overall Result:</strong> <span class=\"{'pass' if summary.get('passed', False) else 'fail'}\">{'PASS' if summary.get('passed', False) else 'FAIL'}</span></p>")
            
            html.append("  </div>")
        
        # Add single questions section
        if "single_questions" in results:
            html.append("  <h2>Single Questions</h2>")
            
            for i, question in enumerate(results["single_questions"], 1):
                html.append(f"  <div class='question'>")
                html.append(f"    <h3>Question {i}: {question['question_id']}</h3>")
                html.append(f"    <p><strong>Category:</strong> {question.get('category', 'N/A')}</p>")
                html.append(f"    <p><strong>Question:</strong> {question['question']}</p>")
                html.append(f"    <p><strong>Expected Answer:</strong> {question.get('expected_answer', 'N/A')}</p>")
                html.append(f"    <p><strong>Actual Response:</strong> {question.get('actual_response', 'N/A')}</p>")
                
                if "error" in question:
                    html.append(f"    <p><strong>Error:</strong> <span class='fail'>{question['error']}</span></p>")
                elif "evaluation" in question:
                    eval_data = question["evaluation"]
                    html.append(f"    <p><strong>Weighted Average Score:</strong> {eval_data.get('weighted_average', 0):.2f}</p>")
                    html.append("    <h4>Criteria Scores:</h4>")
                    html.append("    <table class='criteria'>")
                    html.append("      <tr><th>Criterion</th><th>Score</th><th>Justification</th></tr>")
                    
                    for score in eval_data.get("criteria_scores", []):
                        html.append(f"      <tr><td>{score['name']}</td><td>{score['score']}</td><td>{score['justification']}</td></tr>")
                    
                    html.append("    </table>")
                    html.append(f"    <p><strong>Overall Feedback:</strong> {eval_data.get('overall_feedback', 'N/A')}</p>")
                
                # Add RAGAS evaluation if available
                if "ragas_evaluation" in question:
                    html.append("    <div class='ragas'>")
                    html.append("      <h4>RAGAS Metrics:</h4>")
                    html.append("      <table>")
                    html.append("        <tr><th>Metric</th><th>Score</th></tr>")
                    
                    for metric, score in question["ragas_evaluation"].items():
                        html.append(f"        <tr><td>{metric}</td><td>{score:.2f}</td></tr>")
                    
                    html.append("      </table>")
                    html.append("    </div>")
                
                html.append("  </div>")
        
        # Add conversations section
        if "conversations" in results:
            html.append("  <h2>Conversations</h2>")
            
            for i, conversation in enumerate(results["conversations"], 1):
                html.append(f"  <div class='conversation'>")
                html.append(f"    <div class='conversation-header'>")
                html.append(f"      <h3>Conversation {i}: {conversation['conversation_id']}</h3>")
                
                if "error" in conversation:
                    html.append(f"      <p><strong>Error:</strong> <span class='fail'>{conversation['error']}</span></p>")
                    html.append("    </div>")
                    html.append("  </div>")
                    continue
                
                # Add conversation summary
                if "summary" in conversation:
                    summary = conversation["summary"]
                    html.append(f"      <p><strong>Total Turns:</strong> {summary.get('total_turns', 0)}</p>")
                    html.append(f"      <p><strong>Successful Evaluations:</strong> {summary.get('successful_evaluations', 0)}</p>")
                    html.append(f"      <p><strong>Failed Evaluations:</strong> {summary.get('failed_evaluations', 0)}</p>")
                    html.append(f"      <p><strong>Average Score:</strong> {summary.get('average_score', 0):.2f}</p>")
                    html.append(f"      <p><strong>Pass Threshold:</strong> {summary.get('pass_threshold', 0):.2f}</p>")
                    html.append(f"      <p><strong>Overall Result:</strong> <span class=\"{'pass' if summary.get('passed', False) else 'fail'}\">{'PASS' if summary.get('passed', False) else 'FAIL'}</span></p>")
                
                html.append("    </div>")
                
                # Add individual turns
                if "turns" in conversation:
                    for j, turn in enumerate(conversation["turns"], 1):
                        html.append(f"    <div class='turn'>")
                        html.append(f"      <h4>Turn {j}: {turn['question_id']}</h4>")
                        html.append(f"      <p><strong>Question:</strong> {turn['question']}</p>")
                        html.append(f"      <p><strong>Expected Answer:</strong> {turn.get('expected_answer', 'N/A')}</p>")
                        html.append(f"      <p><strong>Actual Response:</strong> {turn.get('actual_response', 'N/A')}</p>")
                        
                        if "error" in turn:
                            html.append(f"      <p><strong>Error:</strong> <span class='fail'>{turn['error']}</span></p>")
                        elif "evaluation" in turn:
                            eval_data = turn["evaluation"]
                            html.append(f"      <p><strong>Weighted Average Score:</strong> {eval_data.get('weighted_average', 0):.2f}</p>")
                            html.append("      <h5>Criteria Scores:</h5>")
                            html.append("      <table class='criteria'>")
                            html.append("        <tr><th>Criterion</th><th>Score</th><th>Justification</th></tr>")
                            
                            for score in eval_data.get("criteria_scores", []):
                                html.append(f"        <tr><td>{score['name']}</td><td>{score['score']}</td><td>{score['justification']}</td></tr>")
                            
                            html.append("      </table>")
                        
                        html.append("    </div>")
                
                html.append("  </div>")
        
        # Add questions section for feature file evaluation
        if "questions" in results:
            html.append("  <h2>Questions</h2>")
            
            for i, question in enumerate(results["questions"], 1):
                html.append(f"  <div class='question'>")
                html.append(f"    <h3>Question {i}: {question['question_id']}</h3>")
                html.append(f"    <p><strong>Question:</strong> {question['question']}</p>")
                html.append(f"    <p><strong>Expected Answer:</strong> {question.get('expected_answer', 'N/A')}</p>")
                html.append(f"    <p><strong>Actual Response:</strong> {question.get('actual_response', 'N/A')}</p>")
                
                if "error" in question:
                    html.append(f"    <p><strong>Error:</strong> <span class='fail'>{question['error']}</span></p>")
                elif "evaluation" in question:
                    eval_data = question["evaluation"]
                    html.append(f"    <p><strong>Weighted Average Score:</strong> {eval_data.get('weighted_average', 0):.2f}</p>")
                    html.append("    <h4>Criteria Scores:</h4>")
                    html.append("    <table class='criteria'>")
                    html.append("      <tr><th>Criterion</th><th>Score</th><th>Justification</th></tr>")
                    
                    for score in eval_data.get("criteria_scores", []):
                        html.append(f"      <tr><td>{score['name']}</td><td>{score['score']}</td><td>{score['justification']}</td></tr>")
                    
                    html.append("    </table>")
                    html.append(f"    <p><strong>Overall Feedback:</strong> {eval_data.get('overall_feedback', 'N/A')}</p>")
                
                html.append("  </div>")
        
        # Add turns section for single conversation evaluation
        if "turns" in results:
            html.append("  <h2>Conversation Turns</h2>")
            
            for i, turn in enumerate(results["turns"], 1):
                html.append(f"  <div class='turn'>")
                html.append(f"    <h3>Turn {i}: {turn['question_id']}</h3>")
                html.append(f"    <p><strong>Question:</strong> {turn['question']}</p>")
                html.append(f"    <p><strong>Expected Answer:</strong> {turn.get('expected_answer', 'N/A')}</p>")
                html.append(f"    <p><strong>Actual Response:</strong> {turn.get('actual_response', 'N/A')}</p>")
                
                if "error" in turn:
                    html.append(f"    <p><strong>Error:</strong> <span class='fail'>{turn['error']}</span></p>")
                elif "evaluation" in turn:
                    eval_data = turn["evaluation"]
                    html.append(f"    <p><strong>Weighted Average Score:</strong> {eval_data.get('weighted_average', 0):.2f}</p>")
                    html.append("    <h4>Criteria Scores:</h4>")
                    html.append("    <table class='criteria'>")
                    html.append("      <tr><th>Criterion</th><th>Score</th><th>Justification</th></tr>")
                    
                    for score in eval_data.get("criteria_scores", []):
                        html.append(f"      <tr><td>{score['name']}</td><td>{score['score']}</td><td>{score['justification']}</td></tr>")
                    
                    html.append("    </table>")
                
                html.append("  </div>")
        
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)


def run_enhanced_evaluation(
    template_path: Optional[str] = None,
    test_client=None
) -> Dict[str, Any]:
    """
    Run a complete evaluation using the enhanced chatbot evaluator.
    
    Args:
        template_path: Optional path to a custom evaluation template
        test_client: Optional test client for direct integration testing
        
    Returns:
        Evaluation results dictionary
    """
    # Create evaluator
    evaluator = EnhancedChatbotEvaluator(
        template_path=template_path,
        test_client=test_client
    )
    
    # Run evaluation
    results = evaluator.evaluate_all_questions()
    
    # Save results
    results_path = evaluator.save_results(results)
    
    # Generate report
    report = evaluator.generate_report(results)
    report_path = os.path.join(evaluator.results_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Evaluation complete. Results saved to {results_path}")
    logger.info(f"Report saved to {report_path}")
    
    return results


if __name__ == "__main__":
    # Run evaluation with default settings
    run_enhanced_evaluation()
