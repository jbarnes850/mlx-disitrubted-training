import numpy as np
from typing import Dict, List, Optional, Union
import re
from dataclasses import dataclass
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from collections import Counter

@dataclass
class ValidationConfig:
    """Configuration for data validation"""
    perplexity_threshold: float = 100.0
    repetition_threshold: float = 0.3
    toxicity_threshold: float = 0.1
    min_length: int = 50
    max_length: int = 2048
    min_unique_words: int = 20
    max_line_length: int = 100
    language_threshold: float = 0.8  # For language detection
    enable_toxicity: bool = True
    enable_perplexity: bool = True

class DataValidator:
    """Validates and filters training data"""
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize perplexity model if enabled
        if self.config.enable_perplexity:
            self.perplexity_model = AutoModelForCausalLM.from_pretrained(
                "gpt2-small",
                torch_dtype=torch.float16
            )
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2-small")
            
        # Initialize toxicity model if enabled
        if self.config.enable_toxicity:
            self._setup_toxicity_detection()
            
    def validate_sample(self, text: str) -> Dict[str, Union[bool, float]]:
        """Validate a single text sample"""
        if not isinstance(text, str) or not text.strip():
            return {"valid": False, "reason": "empty_or_invalid"}
            
        # Basic length checks
        if len(text) < self.config.min_length:
            return {"valid": False, "reason": "too_short"}
        if len(text) > self.config.max_length:
            return {"valid": False, "reason": "too_long"}
            
        # Compute quality scores
        scores = {
            "repetition": self._compute_repetition(text),
            "unique_words": self._compute_unique_words(text),
            "line_lengths": self._check_line_lengths(text)
        }
        
        # Add perplexity if enabled
        if self.config.enable_perplexity:
            scores["perplexity"] = self._compute_perplexity(text)
            
        # Add toxicity if enabled
        if self.config.enable_toxicity:
            scores["toxicity"] = self._compute_toxicity(text)
            
        # Validate against thresholds
        valid = all([
            scores.get("perplexity", 0) <= self.config.perplexity_threshold,
            scores["repetition"] <= self.config.repetition_threshold,
            scores.get("toxicity", 0) <= self.config.toxicity_threshold,
            scores["unique_words"] >= self.config.min_unique_words,
            scores["line_lengths"]
        ])
        
        return {
            "valid": valid,
            "scores": scores,
            "reason": self._get_rejection_reason(scores) if not valid else None
        }
        
    def _compute_perplexity(self, text: str) -> float:
        """Compute perplexity score using GPT-2"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.perplexity_model(**inputs)
                
            loss = outputs.loss
            return torch.exp(loss).item()
        except Exception as e:
            self.logger.warning(f"Perplexity computation failed: {str(e)}")
            return float('inf')
            
    def _compute_repetition(self, text: str) -> float:
        """Compute repetition score"""
        words = text.lower().split()
        if not words:
            return 1.0
            
        # Count word frequencies
        word_counts = Counter(words)
        
        # Compute repetition score
        total_words = len(words)
        repeated_words = sum(count for count in word_counts.values() if count > 1)
        
        return repeated_words / total_words if total_words > 0 else 1.0
        
    def _compute_unique_words(self, text: str) -> int:
        """Count unique words"""
        words = set(text.lower().split())
        return len(words)
        
    def _check_line_lengths(self, text: str) -> bool:
        """Check if line lengths are reasonable"""
        lines = text.split('\n')
        return all(len(line) <= self.config.max_line_length for line in lines)
        
    def _compute_toxicity(self, text: str) -> float:
        """Compute toxicity score"""
        if not self.config.enable_toxicity:
            return 0.0
            
        try:
            # Placeholder for actual toxicity detection
            # In production, you'd want to use a proper toxicity model
            toxic_words = {"hate", "violent", "offensive"}  # Example
            words = set(text.lower().split())
            return len(words.intersection(toxic_words)) / len(words) if words else 0.0
        except Exception as e:
            self.logger.warning(f"Toxicity computation failed: {str(e)}")
            return 0.0
            
    def _setup_toxicity_detection(self):
        """Setup toxicity detection model"""
        # TODO: Initialize proper toxicity detection model
        pass
        
    def _get_rejection_reason(self, scores: Dict[str, float]) -> str:
        """Get detailed rejection reason"""
        if scores.get("perplexity", 0) > self.config.perplexity_threshold:
            return "high_perplexity"
        elif scores["repetition"] > self.config.repetition_threshold:
            return "high_repetition"
        elif scores.get("toxicity", 0) > self.config.toxicity_threshold:
            return "toxic_content"
        elif scores["unique_words"] < self.config.min_unique_words:
            return "low_vocabulary"
        elif not scores["line_lengths"]:
            return "line_length_violation"
        return "unknown"
        
    def validate_batch(
        self,
        texts: List[str],
        return_scores: bool = False
    ) -> Union[List[bool], List[Dict]]:
        """Validate a batch of texts"""
        results = [self.validate_sample(text) for text in texts]
        
        if return_scores:
            return results
        return [r["valid"] for r in results] 