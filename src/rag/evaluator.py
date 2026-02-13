"""
Research-OS Evaluator
=====================
Lightweight, encoder-based evaluation for RAG.
Models:
1. Faithfulness: cross-encoder/nli-deberta-v3-xsmall
2. Relevancy: cross-encoder/ms-marco-MiniLM-L-6-v2
"""

import logging
from typing import List, Dict, Tuple

from .config import FAITHFULNESS_MODEL_ID, RELEVANCY_MODEL_ID

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """
    Evaluator using Cross-Encoders for high-speed metrics.
    """
    
    FAITHFULNESS_MODEL = FAITHFULNESS_MODEL_ID
    RELEVANCY_MODEL = RELEVANCY_MODEL_ID
    
    def __init__(self):
        """Initialize models lazily."""
        self.device = None
        self.faithfulness_model = None
        self.relevancy_model = None
        self._available = True
        self.torch = None
        self.CrossEncoder = None
        
        try:
            import torch
            from sentence_transformers import CrossEncoder
            self.torch = torch
            self.CrossEncoder = CrossEncoder
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            logger.warning("torch or sentence_transformers not installed. Evaluator disabled.")
            self._available = False

    def _ensure_models(self):
        """Ensure models are loaded."""
        if not self._available:
            return False
            
        if self.faithfulness_model is None:
            try:
                logger.info(f"Loading Evaluator models on {self.device}...")
                self.faithfulness_model = self.CrossEncoder(self.FAITHFULNESS_MODEL, device=self.device)
                self.relevancy_model = self.CrossEncoder(self.RELEVANCY_MODEL, device=self.device)
                logger.info("✓ Evaluator models loaded.")
            except Exception as e:
                logger.error(f"Failed to load evaluator models: {e}")
                self._available = False
                return False
        return True

    def evaluate_faithfulness(self, context: str, answer: str) -> float:
        """
        Check if answer is supported by context (NLI Entailment).
        Returns score 0.0 to 1.0
        """
        if not self._ensure_models():
            return 0.0

        if not context or not answer:
            return 0.0
            
        # Break answer into sentences
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except (LookupError, OSError):
                nltk.download('punkt', quiet=True)
            sentences = nltk.sent_tokenize(answer)
        except Exception:
            # Fallback to simple splitting if nltk fails
            sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 10]
            if not sentences:
                sentences = [answer]
            
        if not sentences:
            return 0.0
            
        # Create pairs: (context, sentence)
        pairs = [(context, sent) for sent in sentences]
        
        # Predict
        try:
            scores = self.faithfulness_model.predict(pairs)
            
            # cross-encoder/nli-deberta-v3-xsmall labels:
            # 0: contradiction, 1: neutral, 2: entailment
            # We want to check if entailment (index 2) is the highest score
            entailment_idx = 2
            entailed_count = 0
            
            for score_dist in scores:
                predicted_label_idx = score_dist.argmax()
                if predicted_label_idx == entailment_idx:
                    entailed_count += 1
                    
            return entailed_count / len(sentences)
        except Exception as e:
            logger.error(f"Faithfulness evaluation error: {e}")
            return 0.0

    def evaluate_relevancy(self, query: str, answer: str) -> float:
        """
        Check if answer is relevant to the query.
        Returns score 0.0 to 1.0 (Sigmoid)
        """
        if not self._ensure_models():
            return 0.0

        if not query or not answer:
            return 0.0
            
        try:
            # Predict returns a single logit for ms-marco models
            score = self.relevancy_model.predict([(query, answer)])
            
            # Apply sigmoid to get 0-1 probability
            sigmoid_score = 1 / (1 + self.torch.exp(self.torch.tensor(-score[0])))
            
            return float(sigmoid_score)
        except Exception as e:
            logger.error(f"Relevancy evaluation error: {e}")
            return 0.0
