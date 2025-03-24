"""
LLM critique module for the RAG evaluator.

This module provides classes and functions for using LLMs as judges
to evaluate RAG system outputs.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import os
import json
import logging
from dataclasses import dataclass, field
import openai

from rag_evaluator.core.data import EvaluationSample, RAGEvaluationSample, EvaluationResult


class LLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize LLM provider.
        
        Args:
            model_name: Name of the LLM model to use.
            **kwargs: Additional provider-specific parameters.
        """
        self.model_name = model_name
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
    
    def generate(self, prompt: str) -> str:
        """Generate text from the LLM.
        
        Args:
            prompt: Prompt to send to the LLM.
            
        Returns:
            Generated text from the LLM.
        """
        raise NotImplementedError("Subclasses must implement generate method")


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: str = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ):
        """Initialize OpenAI provider.
        
        Args:
            model_name: Name of the OpenAI model to use.
            api_key: OpenAI API key.
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional parameters for the OpenAI API.
        """
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            self.logger.warning("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.api_key
    
    def generate(self, prompt: str) -> str:
        """Generate text from OpenAI.
        
        Args:
            prompt: Prompt to send to OpenAI.
            
        Returns:
            Generated text from OpenAI.
        """
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self.kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating text from OpenAI: {str(e)}")
            return ""


class LLMCritic:
    """Base class for LLM critics."""
    
    def __init__(self, llm_provider: LLMProvider):
        """Initialize LLM critic.
        
        Args:
            llm_provider: LLM provider to use for generation.
        """
        self.llm_provider = llm_provider
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate a sample using the LLM.
        
        Args:
            sample: Sample to evaluate.
            
        Returns:
            Evaluation result.
        """
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def _parse_score(self, text: str) -> float:
        """Parse score from LLM output.
        
        Args:
            text: Text to parse.
            
        Returns:
            Parsed score as a float between 0 and 1.
        """
        try:
            # Look for patterns like "Score: 0.8" or "Rating: 4/5"
            lines = text.strip().split('\n')
            for line in lines:
                if "score:" in line.lower():
                    parts = line.split(":")
                    if len(parts) > 1:
                        score_text = parts[1].strip()
                        # Handle different formats
                        if "/" in score_text:
                            # Format like "4/5"
                            num, denom = score_text.split("/")
                            return float(num) / float(denom)
                        else:
                            # Format like "0.8"
                            return float(score_text)
                
                # Look for just a number at the end
                if line.strip().replace(".", "").isdigit():
                    score = float(line.strip())
                    # Normalize to 0-1 if needed
                    if score > 1:
                        return score / 10 if score <= 10 else score / 100
                    return score
            
            # If no score found, look for the first number in the text
            import re
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
            if numbers:
                score = float(numbers[0])
                # Normalize to 0-1 if needed
                if score > 1:
                    return score / 10 if score <= 10 else score / 100
                return score
            
            return 0.5  # Default score if parsing fails
        except Exception as e:
            self.logger.error(f"Error parsing score: {str(e)}")
            return 0.5  # Default score if parsing fails


class PairwiseComparisonCritic(LLMCritic):
    """Critic for pairwise comparison of responses."""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        criteria: List[str] = None,
        prompt_template: str = None
    ):
        """Initialize pairwise comparison critic.
        
        Args:
            llm_provider: LLM provider to use for generation.
            criteria: List of criteria to use for comparison.
            prompt_template: Custom prompt template for comparison.
        """
        super().__init__(llm_provider)
        self.criteria = criteria or ["relevance", "accuracy", "completeness"]
        self.prompt_template = prompt_template or self._default_prompt_template()
    
    def _default_prompt_template(self) -> str:
        """Get default prompt template for pairwise comparison."""
        return """
        You are an expert evaluator of language model responses. Your task is to compare two responses to the same query and determine which one is better.

        Query: {query}

        Response A:
        {response_a}

        Response B:
        {response_b}

        Please evaluate the responses based on the following criteria:
        {criteria_list}

        First, analyze each response according to these criteria. Then, determine which response is better overall.
        Finally, provide your verdict in the format:
        BETTER: A or B
        SCORE: [0-1] (where 1 means the chosen response is much better, and 0.5 means they are roughly equal)
        REASONING: Your explanation
        """
    
    def evaluate(
        self,
        query: str,
        response_a: str,
        response_b: str,
        sample_id: str = None
    ) -> EvaluationResult:
        """Evaluate two responses to the same query.
        
        Args:
            query: Query that the responses are answering.
            response_a: First response to compare.
            response_b: Second response to compare.
            sample_id: Optional sample ID for the evaluation result.
            
        Returns:
            Evaluation result with comparison details.
        """
        criteria_list = "\n".join([f"- {criterion}" for criterion in self.criteria])
        
        prompt = self.prompt_template.format(
            query=query,
            response_a=response_a,
            response_b=response_b,
            criteria_list=criteria_list
        )
        
        llm_output = self.llm_provider.generate(prompt)
        
        # Parse the output to determine which response is better
        better_response = "A"
        if "BETTER: B" in llm_output or "better: b" in llm_output.lower():
            better_response = "B"
        
        # Parse the score
        score = self._parse_score(llm_output)
        
        # Extract reasoning
        reasoning = ""
        if "REASONING:" in llm_output:
            reasoning = llm_output.split("REASONING:")[1].strip()
        elif "reasoning:" in llm_output.lower():
            reasoning = llm_output.split("reasoning:", 1)[1].strip()
        
        return EvaluationResult(
            sample_id=sample_id or "",
            metric_name="pairwise_comparison",
            score=score,
            details={
                "better_response": better_response,
                "reasoning": reasoning,
                "llm_output": llm_output
            }
        )


class CriteriaBasedCritic(LLMCritic):
    """Critic for evaluating responses based on specific criteria."""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        criterion: str,
        prompt_template: str = None,
        reference_free: bool = True
    ):
        """Initialize criteria-based critic.
        
        Args:
            llm_provider: LLM provider to use for generation.
            criterion: Criterion to evaluate (e.g., "relevance", "accuracy").
            prompt_template: Custom prompt template for evaluation.
            reference_free: Whether the evaluation is reference-free.
        """
        super().__init__(llm_provider)
        self.criterion = criterion
        self.reference_free = reference_free
        self.prompt_template = prompt_template or (
            self._default_reference_free_template() if reference_free
            else self._default_reference_based_template()
        )
    
    def _default_reference_free_template(self) -> str:
        """Get default prompt template for reference-free evaluation."""
        return """
        You are an expert evaluator of language model responses. Your task is to evaluate a response to a query based on {criterion}.

        Query: {query}

        Response:
        {response}

        Please evaluate the response based on {criterion}. Consider the following:
        - How well does the response address the query in terms of {criterion}?
        - What are the strengths and weaknesses of the response regarding {criterion}?

        First, provide a detailed analysis of the response's {criterion}.
        Then, rate the response on a scale from 0 to 1, where:
        - 0 means extremely poor {criterion}
        - 0.5 means acceptable {criterion}
        - 1 means excellent {criterion}

        Finally, provide your verdict in the format:
        SCORE: [0-1]
        REASONING: Your explanation
        """
    
    def _default_reference_based_template(self) -> str:
        """Get default prompt template for reference-based evaluation."""
        return """
        You are an expert evaluator of language model responses. Your task is to evaluate a response to a query based on {criterion}, comparing it to a reference answer.

        Query: {query}

        Response to evaluate:
        {response}

        Reference answer:
        {reference}

        Please evaluate the response based on {criterion} compared to the reference. Consider the following:
        - How well does the response match the reference in terms of {criterion}?
        - What information is missing or incorrect compared to the reference?

        First, provide a detailed analysis of the response's {criterion} relative to the reference.
        Then, rate the response on a scale from 0 to 1, where:
        - 0 means completely different from reference in terms of {criterion}
        - 0.5 means partially matching the reference in terms of {criterion}
        - 1 means perfectly matching the reference in terms of {criterion}

        Finally, provide your verdict in the format:
        SCORE: [0-1]
        REASONING: Your explanation
        """
    
    def evaluate(self, sample: RAGEvaluationSample) -> EvaluationResult:
        """Evaluate a sample based on the specified criterion.
        
        Args:
            sample: Sample to evaluate.
            
        Returns:
            Evaluation result with criterion-based score.
        """
        if not self.reference_free and sample.reference_answer is None:
            self.logger.warning(
                f"Reference-based evaluation requested but no reference answer provided for sample {sample.id}. "
                "Falling back to reference-free evaluation."
            )
            prompt_template = self._default_reference_free_template()
            prompt = prompt_template.format(
                criterion=self.criterion,
                query=sample.query,
                response=sample.response
            )
        else:
            prompt = self.prompt_template.format(
                criterion=self.criterion,
                query=sample.query,
                response=sample.response,
                reference=sample.reference_answer if not self.reference_free else ""
            )
        
        llm_output = self.llm_provider.generate(prompt)
        
        # Parse the score
        score = self._parse_score(llm_output)
        
        # Extract reasoning
        reasoning = ""
        if "REASONING:" in llm_output:
            reasoning = llm_output.split("REASONING:")[1].strip()
        elif "reasoning:" in llm_output.lower():
            reasoning = llm_output.split("reasoning:", 1)[1].strip()
        
        return EvaluationResult(
            sample_id=sample.id,
            metric_name=f"llm_{self.criterion}",
            score=score,
            details={
                "criterion": self.criterion,
                "reasoning": reasoning,
                "llm_output": llm_output,
                "reference_free": self.reference_free
            }
        )


class ContextAwarenessCritic(LLMCritic):
    """Critic for evaluating how well responses use the provided context."""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_template: str = None
    ):
        """Initialize context awareness critic.
        
        Args:
            llm_provider: LLM provider to use for generation.
            prompt_template: Custom prompt template for evaluation.
        """
        super().__init__(llm_provider)
        self.prompt_template = prompt_template or self._default_prompt_template()
    
    def _default_prompt_template(self) -> str:
        """Get default prompt template for context awareness evaluation."""
        return """
        You are an expert evaluator of language model responses. Your task is to evaluate how well a response uses the provided context to answer a query.

        Query: {query}

        Context provided to the model:
        {context}

        Response generated by the model:
        {response}

        Please evaluate how well the response uses the information in the context. Consider the following:
        - Does the response use the relevant information from the context?
        - Does the response include information not present in the context (hallucination)?
        - Is the response faithful to the context?

        First, provide a detailed analysis of how the response uses the context.
        Then, rate the response on a scale from 0 to 1, where:
        - 0 means the response completely ignores the context or contradicts it
        - 0.5 means the response partially uses the context but misses key information or adds hallucinations
        - 1 means the response perfectly uses the relevant information from the context

        Finally, provide your verdict in the format:
        SCORE: [0-1]
        REASONING: Your explanation
        """
    
    def evaluate(self, sample: RAGEvaluationSample) -> EvaluationResult:
        """Evaluate a sample <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>