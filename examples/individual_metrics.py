"""
Example showing how to use individual evaluation metrics
"""

from llm_eval_toolkit import LLMEvaluator

def main():
    evaluator = LLMEvaluator()
    
    # Sample data
    candidates = ["The weather is nice today.", "It's a beautiful sunny day."]
    references = ["Today has beautiful weather.", "The day is sunny and pleasant."]
    
    print("=== Individual Metric Examples ===\n")
    
    # BLEU and ROUGE
    bleu, rouge1 = evaluator.evaluate_bleu_rouge(candidates, references)
    print(f"BLEU Score: {bleu:.4f}")
    print(f"ROUGE-1 Score: {rouge1:.4f}")
    
    # BERT Score
    bert_p, bert_r, bert_f1 = evaluator.evaluate_bert_score(candidates, references)
    print(f"BERT Precision: {bert_p:.4f}")
    print(f"BERT Recall: {bert_r:.4f}")
    print(f"BERT F1: {bert_f1:.4f}")
    
    # Perplexity
    text = "The quick brown fox jumps over the lazy dog."
    perplexity = evaluator.evaluate_perplexity(text)
    print(f"Perplexity: {perplexity:.4f}")
    
    # Diversity
    texts = ["Hello world", "Hello there", "Hi world", "Hey there"]
    diversity = evaluator.evaluate_diversity(texts)
    print(f"Diversity Score: {diversity:.4f}")
    
    # METEOR
    meteor = evaluator.evaluate_meteor(candidates, references)
    print(f"METEOR Score: {meteor:.4f}")
    
    # CHRF
    chrf = evaluator.evaluate_chrf(candidates, references)
    print(f"CHRF Score: {chrf:.4f}")
    
    # Readability
    text = "This is a simple sentence that should be easy to read and understand."
    flesch_ease, flesch_grade = evaluator.evaluate_readability(text)
    print(f"Flesch Reading Ease: {flesch_ease:.4f}")
    print(f"Flesch-Kincaid Grade: {flesch_grade:.4f}")
    
    # MAUVE
    ref_texts = ["The sky is blue.", "Water is wet."]
    gen_texts = ["The sky appears blue.", "Water feels wet."]
    mauve_score = evaluator.evaluate_mauve(ref_texts, gen_texts)
    print(f"MAUVE Score: {mauve_score:.4f}")
    
    # Bias detection
    neutral_text = "The weather is nice today."
    bias_score = evaluator.evaluate_bias(neutral_text)
    print(f"Bias Score (neutral text): {bias_score:.4f}")

if __name__ == "__main__":
    main()