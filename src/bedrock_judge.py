#!/usr/bin/env python3
"""
AWS Bedrock LLM integration for chatbot evaluation.
This module provides functionality to use AWS Bedrock models as judges
for evaluating chatbot responses.
"""

import os
import json
import boto3
import logging
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class BedrockLLMJudge:
    """
    A class to interact with AWS Bedrock models for evaluating chatbot responses.
    """
    
    def __init__(
        self, 
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0", 
        region_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ):
        """
        Initialize the BedrockLLMJudge.
        
        Args:
            model_id: The Bedrock model ID to use
            region_name: AWS region name (defaults to environment variable or 'us-east-1')
            temperature: Temperature for model inference (lower for more deterministic outputs)
            max_tokens: Maximum tokens in the response
        """
        self.model_id = model_id
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Bedrock client
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the Bedrock Runtime client."""
        try:
            self.bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.region_name
            )
            logger.info(f"Initialized Bedrock client in region {self.region_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {str(e)}")
            raise
    
    def _prepare_anthropic_prompt(
        self, 
        system_prompt: str, 
        user_message: str
    ) -> Dict[str, Any]:
        """
        Prepare the prompt for Anthropic Claude models.
        
        Args:
            system_prompt: The system prompt to guide the model
            user_message: The user message content
            
        Returns:
            Dict containing the formatted request body
        """
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        }
    
    def _prepare_amazon_prompt(
        self, 
        system_prompt: str, 
        user_message: str
    ) -> Dict[str, Any]:
        """
        Prepare the prompt for Amazon Titan models.
        
        Args:
            system_prompt: The system prompt to guide the model
            user_message: The user message content
            
        Returns:
            Dict containing the formatted request body
        """
        return {
            "inputText": f"<system>{system_prompt}</system>\n\n<user>{user_message}</user>",
            "textGenerationConfig": {
                "maxTokenCount": self.max_tokens,
                "temperature": self.temperature,
                "topP": 0.9
            }
        }
    
    def _prepare_prompt(
        self, 
        system_prompt: str, 
        user_message: str
    ) -> Dict[str, Any]:
        """
        Prepare the appropriate prompt based on the model provider.
        
        Args:
            system_prompt: The system prompt to guide the model
            user_message: The user message content
            
        Returns:
            Dict containing the formatted request body
        """
        if self.model_id.startswith("anthropic."):
            return self._prepare_anthropic_prompt(system_prompt, user_message)
        elif self.model_id.startswith("amazon."):
            return self._prepare_amazon_prompt(system_prompt, user_message)
        else:
            raise ValueError(f"Unsupported model provider in model_id: {self.model_id}")
    
    def invoke_model(
        self, 
        system_prompt: str, 
        user_message: str
    ) -> Dict[str, Any]:
        """
        Invoke the Bedrock model with the given prompts.
        
        Args:
            system_prompt: The system prompt to guide the model
            user_message: The user message content
            
        Returns:
            Dict containing the model response
        """
        try:
            request_body = self._prepare_prompt(system_prompt, user_message)
            
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response.get("body").read())
            logger.debug(f"Model response: {response_body}")
            
            return response_body
        except Exception as e:
            logger.error(f"Error invoking Bedrock model: {str(e)}")
            raise
    
    def extract_response_content(self, response: Dict[str, Any]) -> str:
        """
        Extract the content from the model response.
        
        Args:
            response: The raw response from the model
            
        Returns:
            The extracted content as a string
        """
        if self.model_id.startswith("anthropic."):
            return response.get("content", [{}])[0].get("text", "")
        elif self.model_id.startswith("amazon."):
            return response.get("results", [{}])[0].get("outputText", "")
        else:
            raise ValueError(f"Unsupported model provider in model_id: {self.model_id}")
    
    def evaluate_response(
        self, 
        question: str, 
        expected_answer: str, 
        actual_response: str, 
        criteria: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate a chatbot response using the Bedrock LLM as judge.
        
        Args:
            question: The original question asked to the chatbot
            expected_answer: The expected answer or ground truth
            actual_response: The actual response from the chatbot
            criteria: List of evaluation criteria with weights and descriptions
            
        Returns:
            Dict containing evaluation scores and feedback
        """
        # Construct the system prompt
        system_prompt = """
        You are an expert evaluator of AI assistant responses. Your task is to evaluate the quality of a response 
        given by an AI assistant to a user question. You will be provided with:
        
        1. The original question
        2. The expected answer (ground truth)
        3. The actual response from the AI assistant
        4. A set of evaluation criteria
        
        For each criterion, provide:
        1. A score on the specified scale
        2. A brief justification for the score
        
        Be objective, fair, and consistent in your evaluation. Focus on the quality of the response 
        in relation to the question and expected answer.
        """
        
        # Construct the user message with the evaluation request
        criteria_text = "\n".join([
            f"- {c['name']} (weight: {c['weight']}, scale: {c['min_score']}-{c['max_score']}): {c['description']}"
            for c in criteria
        ])
        
        user_message = f"""
        # Question
        {question}
        
        # Expected Answer
        {expected_answer}
        
        # Actual Response
        {actual_response}
        
        # Evaluation Criteria
        {criteria_text}
        
        Please evaluate the actual response based on each criterion. For each criterion, provide:
        1. A numerical score within the specified scale
        2. A brief justification for the score
        
        Then, calculate a weighted average score based on the weights provided.
        
        Format your response as a JSON object with the following structure:
        {{
            "criteria_scores": [
                {{
                    "criteria_id": "C001",
                    "name": "Factual Accuracy",
                    "score": 4,
                    "justification": "The response contains mostly accurate information..."
                }},
                ...
            ],
            "weighted_average": 3.75,
            "overall_feedback": "The response is generally good but could improve in..."
        }}
        """
        
        # Invoke the model
        response = self.invoke_model(system_prompt, user_message)
        content = self.extract_response_content(response)
        
        # Parse the JSON response
        try:
            # Extract JSON from the response (handling potential text before/after the JSON)
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                evaluation_result = json.loads(json_str)
            else:
                raise ValueError("Could not find valid JSON in the response")
                
            return evaluation_result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation result as JSON: {str(e)}")
            logger.error(f"Raw content: {content}")
            raise ValueError(f"Failed to parse evaluation result: {str(e)}")

    def evaluate_batch(
        self,
        questions: List[Dict[str, Any]],
        responses: Dict[str, str],
        criteria: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of chatbot responses.
        
        Args:
            questions: List of question objects with id, question, and expected_answer
            responses: Dict mapping question_id to actual response
            criteria: List of evaluation criteria
            
        Returns:
            Dict containing evaluation results for each question and overall statistics
        """
        results = {}
        all_scores = []
        
        for q in questions:
            question_id = q["question_id"]
            if question_id not in responses:
                logger.warning(f"No response found for question {question_id}")
                continue
                
            try:
                eval_result = self.evaluate_response(
                    question=q["question"],
                    expected_answer=q["expected_answer"],
                    actual_response=responses[question_id],
                    criteria=criteria
                )
                
                results[question_id] = eval_result
                all_scores.append(eval_result["weighted_average"])
            except Exception as e:
                logger.error(f"Error evaluating response for question {question_id}: {str(e)}")
                results[question_id] = {"error": str(e)}
        
        # Calculate overall statistics
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            stats = {
                "average_score": avg_score,
                "min_score": min(all_scores),
                "max_score": max(all_scores),
                "total_evaluated": len(all_scores),
                "total_questions": len(questions)
            }
        else:
            stats = {
                "average_score": 0,
                "min_score": 0,
                "max_score": 0,
                "total_evaluated": 0,
                "total_questions": len(questions)
            }
        
        return {
            "question_results": results,
            "statistics": stats
        }


def get_bedrock_judge(
    model_id: Optional[str] = None,
    region_name: Optional[str] = None
) -> BedrockLLMJudge:
    """
    Factory function to create and return a BedrockLLMJudge instance.
    
    Args:
        model_id: Optional model ID (defaults to environment variable or Claude)
        region_name: Optional AWS region (defaults to environment variable or us-east-1)
        
    Returns:
        BedrockLLMJudge instance
    """
    model = model_id or os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
    region = region_name or os.getenv("AWS_REGION", "us-east-1")
    
    return BedrockLLMJudge(model_id=model, region_name=region)


if __name__ == "__main__":
    # Example usage
    judge = get_bedrock_judge()
    
    # Example evaluation criteria
    criteria = [
        {
            "criteria_id": "C001",
            "name": "Factual Accuracy",
            "description": "The response contains factually correct information.",
            "weight": 0.4,
            "min_score": 0,
            "max_score": 5
        },
        {
            "criteria_id": "C002",
            "name": "Completeness",
            "description": "The response addresses all aspects of the question.",
            "weight": 0.3,
            "min_score": 0,
            "max_score": 5
        },
        {
            "criteria_id": "C003",
            "name": "Clarity",
            "description": "The response is clear and easy to understand.",
            "weight": 0.3,
            "min_score": 0,
            "max_score": 5
        }
    ]
    
    # Example evaluation
    result = judge.evaluate_response(
        question="What is the capital of France?",
        expected_answer="The capital of France is Paris.",
        actual_response="Paris is the capital city of France.",
        criteria=criteria
    )
    
    print(json.dumps(result, indent=2))
