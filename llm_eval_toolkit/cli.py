"""
Command-line interface for LLM Evaluator
"""

import argparse
import json
import sys
from typing import List, Dict
from .evaluator import LLMEvaluator

def load_data_from_file(filepath: str) -> List[Dict[str, str]]:
    """Load evaluation data from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        sys.exit(1)

def save_results_to_file(results: List[Dict], filepath: str):
    """Save evaluation results to JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Error saving results to {filepath}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="LLM Evaluator CLI")
    parser.add_argument(
        "input_file", 
        help="JSON file containing evaluation data with 'question', 'response', 'reference' fields"
    )
    parser.add_argument(
        "-o", "--output", 
        help="Output file for results (default: results.json)",
        default="results.json"
    )
    parser.add_argument(
        "--device",
        help="Device to use for computation (cpu/cuda)",
        default=None
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Include summary statistics in output"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_data_from_file(args.input_file)
    
    # Validate data format
    required_fields = ['question', 'response', 'reference']
    for i, item in enumerate(data):
        for field in required_fields:
            if field not in item:
                print(f"Error: Missing '{field}' in item {i}")
                sys.exit(1)
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = LLMEvaluator(device=args.device)
    
    # Extract data
    questions = [item['question'] for item in data]
    responses = [item['response'] for item in data]
    references = [item['reference'] for item in data]
    
    # Evaluate
    print(f"Evaluating {len(data)} samples...")
    results = evaluator.evaluate_batch(questions, responses, references)
    
    # Prepare output
    output_data = {
        "evaluation_results": [
            {**data[i], **results[i]} 
            for i in range(len(data))
        ]
    }
    
    if args.summary:
        summary = evaluator.get_summary_stats(results)
        output_data["summary_statistics"] = summary
    
    # Save results
    save_results_to_file(output_data, args.output)
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()