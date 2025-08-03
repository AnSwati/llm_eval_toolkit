"""
Basic usage example for LLM Evaluator
"""

from llm_eval_toolkit import LLMEvaluator

def main():
    # Initialize the evaluator
    print("Initializing LLM Evaluator...")
    evaluator = LLMEvaluator()
    
    # Example 1: Single evaluation
    print("\n=== Single Evaluation Example ===")
    question = "What is the capital of France?"
    response = "The capital of France is Paris, which is also the largest city in the country."
    reference = "Paris is the capital of France."
    
    results = evaluator.evaluate_all(question, response, reference)
    
    print(f"Question: {question}")
    print(f"Response: {response}")
    print(f"Reference: {reference}")
    print("\nEvaluation Results:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.4f}")
    
    # Example 2: Batch evaluation
    print("\n=== Batch Evaluation Example ===")
    questions = [
        "What is artificial intelligence?",
        "Explain machine learning",
        "What is deep learning?"
    ]
    
    responses = [
        "Artificial intelligence is the simulation of human intelligence in machines.",
        "Machine learning is a subset of AI that enables computers to learn from data.",
        "Deep learning uses neural networks with multiple layers to learn patterns."
    ]
    
    references = [
        "AI is the simulation of human intelligence processes by machines.",
        "ML is a method of data analysis that automates analytical model building.",
        "Deep learning is a subset of machine learning with multi-layered neural networks."
    ]
    
    batch_results = evaluator.evaluate_batch(questions, responses, references)
    summary = evaluator.get_summary_stats(batch_results)
    
    print("Summary Statistics:")
    for metric, stats in summary.items():
        print(f"  {metric}:")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Std:  {stats['std']:.4f}")
        print(f"    Min:  {stats['min']:.4f}")
        print(f"    Max:  {stats['max']:.4f}")

if __name__ == "__main__":
    main()