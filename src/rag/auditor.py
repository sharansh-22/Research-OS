"""
Research-OS Conceptual Auditor
==============================
LLM-based evaluation for RAG with Chain-of-Thought reasoning.
Uses Groq (llama-3.3-70b-versatile) for high-fidelity auditing.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

from .config import AUDITOR_MODEL, GROQ_API_KEY

logger = logging.getLogger(__name__)

AUDITOR_PROMPT = """You are a Senior Research Auditor. Your task is to verify the conceptual integrity of an AI-generated answer based on a specific Research Context.

Research Context:
[CONTEXT]

AI Answer:
[ANSWER]

Instructions:
1. Identify the core mathematical theory, formula, or fundamental principle mentioned in the Context.
2. Determine if the AI Answer correctly applies this theory, even if it uses external coding libraries or simplified explanations.
3. Be strict about logical consistency. If the answer contradicts the context's mathematical foundation, mark it as unfaithful.
4. Output your analysis in JSON format:
{
  "faithfulness": float (0.0 to 1.0),
  "relevancy": float (0.0 to 1.0),
  "reasoning": "A concise (1-2 sentence) explanation of your logic."
}
"""

class ResearchAuditor:
    """
    Conceptual Auditor using LLM-as-a-Judge with Chain-of-Thought.
    """
    
    def __init__(self, model: str = AUDITOR_MODEL):
        self.model = model
        self.api_key = GROQ_API_KEY
        self.client = None
        
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                logger.info(f"Auditor initialized with {model}")
            except ImportError:
                logger.warning("Groq not installed. Auditor using dummy evaluation.")

    def audit(self, query: str, context: str, answer: str) -> Dict[str, Any]:
        """
        Perform a conceptual audit of the answer.
        """
        if not self.client:
            return {
                "faithfulness": 0.5,
                "relevancy": 0.5,
                "reasoning": "Auditor unavailable (Groq not installed or API key missing)."
            }

        prompt = AUDITOR_PROMPT.replace("[CONTEXT]", context).replace("[ANSWER]", answer)
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise technical auditor. Output only JSON."},
                    {"role": "user", "content": f"Query: {query}\n\n{prompt}"}
                ],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            
            result = json.loads(chat_completion.choices[0].message.content)
            return {
                "faithfulness": float(result.get("faithfulness", 0.0)),
                "relevancy": float(result.get("relevancy", 0.0)),
                "reasoning": result.get("reasoning", "No reasoning provided.")
            }
        except Exception as e:
            logger.error(f"Auditor failed: {e}")
            return {
                "faithfulness": 0.0,
                "relevancy": 0.0,
                "reasoning": f"Audit failed: {str(e)}"
            }
