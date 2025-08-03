#!/usr/bin/env python
"""
Example script demonstrating the hallucination detection capabilities of LLMEvaluator.
"""

import sys
import os
import json
from pprint import pprint

# Add parent directory to path to import llm_eval_toolkit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm_eval_toolkit import LLMEvaluator

def main():
    """Run hallucination detection examples."""
    print("Initializing LLM Evaluator...")
    evaluator = LLMEvaluator()
    
    # Example 1: Factual consistency (no hallucination)
    print("\n=== Example 1: Factual Consistency (No Hallucination) ===")
    reference1 = """
    The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.
    It was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair.
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
    The tower is 330 meters (1,083 ft) tall, and was the tallest man-made structure in the world for 41 years.
    """
    
    response1 = """
    The Eiffel Tower, located in Paris, France, was built between 1887 and 1889.
    It was designed by Gustave Eiffel for the 1889 World's Fair. Standing at 330 meters tall,
    it was the world's tallest structure for 41 years after its completion.
    """
    
    results1 = evaluator.evaluate_all("Tell me about the Eiffel Tower.", response1, reference1)
    print_hallucination_metrics(results1)
    
    # Example 2: Entity hallucination
    print("\n=== Example 2: Entity Hallucination ===")
    reference2 = """
    The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.
    It was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair.
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
    The tower is 330 meters (1,083 ft) tall, and was the tallest man-made structure in the world for 41 years.
    """
    
    response2 = """
    The Eiffel Tower was designed by the famous architect Antoine Lefebvre for the 1889 World's Fair in Paris.
    It was built by the Lefebvre Construction Company between 1887 and 1889. The tower stands at 330 meters tall
    and is located on the Champ de Mars. It was named after the mayor of Paris at that time, François Eiffel.
    """
    
    results2 = evaluator.evaluate_all("Tell me about the Eiffel Tower.", response2, reference2)
    print_hallucination_metrics(results2)
    
    # Example 3: Numerical hallucination
    print("\n=== Example 3: Numerical Hallucination ===")
    reference3 = """
    The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.
    It was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair.
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
    The tower is 330 meters (1,083 ft) tall, and was the tallest man-made structure in the world for 41 years.
    """
    
    response3 = """
    The Eiffel Tower in Paris was built by Gustave Eiffel between 1885 and 1888. 
    It stands at 375 meters tall and was constructed for the 1890 World's Fair. 
    It remained the tallest structure in the world for 30 years after its completion.
    """
    
    results3 = evaluator.evaluate_all("Tell me about the Eiffel Tower.", response3, reference3)
    print_hallucination_metrics(results3)
    
    # Example 4: Logical contradiction
    print("\n=== Example 4: Logical Contradiction ===")
    reference4 = """
    The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.
    It was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair.
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
    The tower is 330 meters (1,083 ft) tall, and was the tallest man-made structure in the world for 41 years.
    """
    
    response4 = """
    The Eiffel Tower was both designed and not designed by Gustave Eiffel. While his company was responsible for the construction,
    he personally opposed the project. The tower was built in 1889 but was actually completed before construction began in 1887.
    Despite being the tallest structure in the world at the time, it was shorter than several other buildings in Paris.
    """
    
    results4 = evaluator.evaluate_all("Tell me about the Eiffel Tower.", response4, reference4)
    print_hallucination_metrics(results4)
    
    # Example 5: Source attribution hallucination
    print("\n=== Example 5: Source Attribution Hallucination ===")
    reference5 = """
    The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.
    It was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair.
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
    The tower is 330 meters (1,083 ft) tall, and was the tallest man-made structure in the world for 41 years.
    """
    
    response5 = """
    According to the French Historical Society's 2020 report, the Eiffel Tower was initially meant to be a temporary structure.
    The Paris Architectural Review of 1887 states that Gustave Eiffel's design was selected from over 100 competing proposals.
    As documented in Eiffel's personal diaries, he was inspired by the pyramids of Egypt when designing the tower's distinctive shape.
    """
    
    results5 = evaluator.evaluate_all("Tell me about the Eiffel Tower.", response5, reference5)
    print_hallucination_metrics(results5)

def print_hallucination_metrics(results):
    """Print only the hallucination-related metrics from the results."""
    hallucination_metrics = {
        "NLI_Contradiction": results["NLI_Contradiction"],
        "NLI_Entailment": results["NLI_Entailment"],
        "NLI_Neutral": results["NLI_Neutral"],
        "NLI_Hallucination": results["NLI_Hallucination"],
        "Entity_Hallucination": results["Entity_Hallucination"],
        "Numerical_Hallucination": results["Numerical_Hallucination"],
        "Semantic_Similarity": results["Semantic_Similarity"],
        "Hallucination_Score": results["Hallucination_Score"]
    }
    pprint(hallucination_metrics)

if __name__ == "__main__":
    main()