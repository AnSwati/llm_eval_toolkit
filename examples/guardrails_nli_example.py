#!/usr/bin/env python
"""
Example script demonstrating the use of GuardrailsAI NLI model for hallucination detection.
"""

import sys
import os
from pprint import pprint

# Add parent directory to path to import llm_eval_toolkit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm_eval_toolkit import LLMEvaluator

def main():
    """Run hallucination detection examples with GuardrailsAI NLI model."""
    print("Initializing LLM Evaluator with GuardrailsAI NLI model...")
    evaluator = LLMEvaluator(nli_model_name='GuardrailsAI/finetuned_nli_provenance')
    
    # Example 1: Entailed statement
    print("\n=== Example 1: Entailed Statement ===")
    reference1 = "The sun rises in the east and sets in the west."
    response1 = "The sun rises in the east."
    
    results1 = evaluator.evaluate_hallucination_nli(response1, reference1)
    pprint(results1)
    
    # Example 2: Contradictory statement
    print("\n=== Example 2: Contradictory Statement ===")
    reference2 = "The sun rises in the east and sets in the west."
    response2 = "The sun rises in the west."
    
    results2 = evaluator.evaluate_hallucination_nli(response2, reference2)
    pprint(results2)
    
    # Example 3: Neutral statement
    print("\n=== Example 3: Neutral Statement ===")
    reference3 = "The sun rises in the east and sets in the west."
    response3 = "The moon orbits the Earth."
    
    results3 = evaluator.evaluate_hallucination_nli(response3, reference3)
    pprint(results3)

if __name__ == "__main__":
    main()