"""
Non-LLM evaluation methods for the RAG evaluator.

This module provides algorithmic evaluation methods that don't rely on LLMs
for evaluating RAG system outputs.
"""

from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import os
import json
import logging
import re
import string
import numpy as np
from dataclasses import dataclass, field
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

from rag_evaluator.core.data import EvaluationSample, RAGEvaluationSample, EvaluationResult


# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextSimilarityEvaluator:
    """Base class for text similarity evaluators."""
    
    def __init__(self):
        """Initialize text similarity evaluator."""
        self.logger = logging.getLogger(__name__)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Similarity score between 0 and 1.
        """
        raise NotImplementedError("Subclasses must implement compute_similarity method")


class TFIDFSimilarityEvaluator(TextSimilarityEvaluator):
    """Text similarity evaluator using TF-IDF and cosine similarity."""
    
    def __init__(self, stop_words: str = 'english'):
        """Initialize TF-IDF similarity evaluator.
        
        Args:
            stop_words: Language for stop words removal.
        """
        super().__init__()
        self.stop_words = stop_words
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts using TF-IDF and cosine similarity.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Similarity score between 0 and 1.
        """
        if not text1 or not text2:
            return 0.0
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error computing TF-IDF similarity: {str(e)}")
            return 0.0


class JaccardSimilarityEvaluator(TextSimilarityEvaluator):
    """Text similarity evaluator using Jaccard similarity."""
    
    def __init__(self, use_stopwords: bool = True, language: str = 'english'):
        """Initialize Jaccard similarity evaluator.
        
        Args:
            use_stopwords: Whether to remove stop words.
            language: Language for stop words removal.
        """
        super().__init__()
        self.use_stopwords = use_stopwords
        self.language = language
        if use_stopwords:
            self.stop_words = set(stopwords.words(language))
        else:
            self.stop_words = set()
    
    def _tokenize(self, text: str) -> set:
        """Tokenize text into words.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            Set of tokens.
        """
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum()]
        if self.use_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        return set(tokens)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts using Jaccard similarity.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Similarity score between 0 and 1.
        """
        if not text1 or not text2:
            return 0.0
        
        try:
            tokens1 = self._tokenize(text1)
            tokens2 = self._tokenize(text2)
            
            if not tokens1 or not tokens2:
                return 0.0
            
            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            
            return len(intersection) / len(union)
        except Exception as e:
            self.logger.error(f"Error computing Jaccard similarity: {str(e)}")
            return 0.0


class EntityExtractionEvaluator:
    """Evaluator for entity extraction and comparison."""
    
    def __init__(self, model: str = "en_core_web_sm"):
        """Initialize entity extraction evaluator.
        
        Args:
            model: spaCy model to use for entity extraction.
        """
        self.logger = logging.getLogger(__name__)
        try:
            self.nlp = spacy.load(model)
        except OSError:
            self.logger.info(f"Downloading spaCy model {model}")
            spacy.cli.download(model)
            self.nlp = spacy.load(model)
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text.
        
        Args:
            text: Text to extract entities from.
            
        Returns:
            Dictionary mapping entity types to lists of entity texts.
        """
        if not text:
            return {}
        
        try:
            doc = self.nlp(text)
            entities = {}
            
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                entities[ent.label_].append(ent.text)
            
            return entities
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            return {}
    
    def compute_entity_recall(self, reference_text: str, generated_text: str) -> float:
        """Compute entity recall between reference and generated texts.
        
        Args:
            reference_text: Reference text containing expected entities.
            generated_text: Generated text to evaluate.
            
        Returns:
            Entity recall score between 0 and 1.
        """
        if not reference_text or not generated_text:
            return 0.0
        
        try:
            reference_entities = self.extract_entities(reference_text)
            generated_entities = self.extract_entities(generated_text)
            
            if not reference_entities:
                return 1.0  # No entities to recall
            
            total_entities = 0
            recalled_entities = 0
            
            for entity_type, entities in reference_entities.items():
                total_entities += len(entities)
                
                if entity_type in generated_entities:
                    gen_entities = set(generated_entities[entity_type])
                    for entity in entities:
                        if entity in gen_entities:
                            recalled_entities += 1
            
            if total_entities == 0:
                return 1.0
            
            return recalled_entities / total_entities
        except Exception as e:
            self.logger.error(f"Error computing entity recall: {str(e)}")
            return 0.0
    
    def compute_entity_precision(self, reference_text: str, generated_text: str) -> float:
        """Compute entity precision between reference and generated texts.
        
        Args:
            reference_text: Reference text containing expected entities.
            generated_text: Generated text to evaluate.
            
        Returns:
            Entity precision score between 0 and 1.
        """
        if not reference_text or not generated_text:
            return 0.0
        
        try:
            reference_entities = self.extract_entities(reference_text)
            generated_entities = self.extract_entities(generated_text)
            
            if not generated_entities:
                return 0.0  # No entities generated
            
            total_generated = 0
            correct_entities = 0
            
            for entity_type, entities in generated_entities.items():
                total_generated += len(entities)
                
                if entity_type in reference_entities:
                    ref_entities = set(reference_entities[entity_type])
                    for entity in entities:
                        if entity in ref_entities:
                            correct_entities += 1
            
            if total_generated == 0:
                return 0.0
            
            return correct_entities / total_generated
        except Exception as e:
            self.logger.error(f"Error computing entity precision: {str(e)}")
            return 0.0
    
    def compute_entity_f1(self, reference_text: str, generated_text: str) -> float:
        """Compute entity F1 score between reference and generated texts.
        
        Args:
            reference_text: Reference text containing expected entities.
            generated_text: Generated text to evaluate.
            
        Returns:
            Entity F1 score between 0 and 1.
        """
        precision = self.compute_entity_precision(reference_text, generated_text)
        recall = self.compute_entity_recall(reference_text, generated_text)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)


class KeywordMatchingEvaluator:
    """Evaluator for keyword matching between texts."""
    
    def __init__(self, use_stopwords: bool = True, language: str = 'english'):
        """Initialize keyword matching evaluator.
        
        Args:
            use_stopwords: Whether to remove stop words.
            language: Language for stop words removal.
        """
        self.logger = logging.getLogger(__name__)
        self.use_stopwords = use_stopwords
        self.language = language
        if use_stopwords:
            self.stop_words = set(stopwords.words(language))
        else:
            self.stop_words = set()
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text.
        
        Args:
            text: Text to extract keywords from.
            
        Returns:
            List of keywords.
        """
        if not text:
            return []
        
        try:
            # Tokenize and clean
            tokens = word_tokenize(text.lower())
            tokens = [token for token in tokens if token.isalnum()]
            
            # Remove stopwords if needed
            if self.use_stopwords:
                tokens = [token for token in tokens if token not in self.stop_words]
            
            return tokens
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def compute_keyword_overlap(self, text1: str, text2: str) -> float:
        """Compute keyword overlap between two texts.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Keyword overlap score between 0 and 1.
        """
        if not text1 or not text2:
            return 0.0
        
        try:
            keywords1 = set(self.extract_keywords(text1))
            keywords2 = set(self.extract_keywords(text2))
            
            if not keywords1 or not keywords2:
                return 0.0
            
            intersection = keywords1.intersection(keywords2)
            
            # Normalize by the smaller set size
            return len(intersection) / min(len(keywords1), len(keywords2))
        except Exception as e:
            self.logger.error(f"Error computing keyword overlap: {str(e)}")
            return 0.0


class ContextRelevanceEvaluator:
    """Evaluator for context relevance to a query."""
    
    def __init__(self, similarity_evaluator: TextSimilarityEvaluator = None):
        """Initialize context relevance evaluator.
        
        Args:
            similarity_evaluator: Text similarity evaluator to use.
        """
        self.logger = logging.getLogger(__name__)
        self.similarity_evaluator = similarity_evaluator or TFIDFSimilarityEvaluator()
    
    def evaluate_context_relevance(
        self,
        query: str,
        contexts: List[str]
    ) -> Tuple[float, List[float]]:
        """Evaluate relevance of contexts to a query.
        
        Args:
            query: Query to evaluate contexts against.
            contexts: List of contexts to evaluate.
            
        Returns:
            Tuple of (average relevance score, list of individual scores).
        """
        if not query or not contexts:
            return 0.0, []
        
        try:
            scores = [
                self.similarity_evaluator.compute_similarity(query, context)
                for context in contexts
            ]
            
            if not scores:
                return 0.0, []
            
            return sum(scores) / len(scores), scores
        except Exception as e:
            self.logger.error(f"Error evaluating context relevance: {str(e)}")
            return 0.0, []


class SQLEvaluator:
    """Evaluator for SQL query equivalence."""
    
    def __init__(self):
        """Initialize SQL evaluator."""
        self.logger = logging.getLogger(__name__)
    
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL query for comparison.
        
        Args:
            sql: SQL query to normalize.
            
        Returns:
            Normalized SQL query.
        """
        if not sql:
            return ""
        
        # Convert to lowercase
        sql = sql.lower()
        
        # Remove comments
        sql = re.sub(r'--.*?(\n|$)', ' ', sql)
        sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)
        
        # Normalize whitespace
        sql = re.sub(r'\s+', ' ', sql).strip()
        
        # Normalize quotes
        sql = sql.replace('"', "'")
        
        # Remove trailing semicolon
        sql = sql.rstrip(';')
        
        return sql
    
    def compute_string_similarity(self, sql1: str, sql2: str) -> float:
        """Compute string similarity between two SQL queries.
        
        Args:
            sql1: First SQL query.
            sql2: Second SQL query.
            
        Returns:
            Similarity score between 0 and 1.
        """
        if not sql1 or not sql2:
            return 0.0
        
        try:
            normalized_sql1 = self._normalize_sql(sql1)
            normalized_sql2 = self._normalize_sql(sql2)
            
            if normalized_sql1 == normalized_sql2:
                return 1.0
            
            # Use Jaccard similarity for approximate matching
            similarity_evaluator = JaccardSimilarityEvaluator()
            return similarity_evaluator.compute_similarity(normalized_sql1, normalized_sql2)
        except Exception as e:
            self.logger.error(f"Error computing SQL string similarity: {str(e)}")
            return 0.0


class ExactMatchEvaluator:
    """Evaluator for exact match between texts."""
    
    def __init__(self, case_sensitive: bool = False, normalize_whitespace: bool = True):
        """Initialize exact match evaluator.
        
        Args:
            case_sensitive: Whether to perform case-sensitive matching.
            normalize_whitespace: Whether to normalize whitespace before matching.
        """
        self.logger = logging.getLogger(__name__)
        self<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>