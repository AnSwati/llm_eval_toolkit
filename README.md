# LLM Eval Toolkit

A comprehensive evaluation toolkit for Large Language Models (LLMs) that provides various metrics to assess the quality, coherence, and safety of generated text.


## Installation

Install the package using pip:

```bash
pip install llm-eval-toolkit
```

## Quick Start

```python
from llm_eval_toolkit import LLMEvaluator

# Initialize the evaluator (models download automatically on first use)
evaluator = LLMEvaluator()

# Evaluate a single response
question = "What is the capital of France?"
response = "The capital of France is Paris."
reference = "Paris is the capital of France."

results = evaluator.evaluate_all(question, response, reference)
print(results)

# Check key metrics
print(f"Hallucination score: {results['Hallucination_Score']:.3f}")
print(f"BERT F1 score: {results['BERT_F1']:.3f}")
print(f"NLI contradiction: {results['NLI_Contradiction']:.3f}")
print(f"Entity hallucination: {results['Entity_Hallucination']:.3f}")

# Evaluate multiple responses
questions = ["What is AI?", "Explain machine learning"]
responses = ["AI is artificial intelligence", "ML is a subset of AI"]
references = ["Artificial intelligence", "Machine learning uses algorithms"]

batch_results = evaluator.evaluate_batch(questions, responses, references)
summary = evaluator.get_summary_stats(batch_results)
print(summary)
```

## Comprehensive Examples

### All Available Methods

```python
from llm_eval_toolkit import LLMEvaluator

# Initialize evaluator
evaluator = LLMEvaluator()

# Sample data
candidates = ["The capital of France is Paris."]
references = ["Paris is the capital of France."]
generated_text = "The capital of France is Paris, a beautiful city."
reference_text = "Paris is the capital of France."
question = "What is the capital of France?"

# 1. COMPREHENSIVE EVALUATION
print("=== COMPREHENSIVE EVALUATION ===")
results = evaluator.evaluate_all(question, generated_text, reference_text)
for metric, value in results.items():
    print(f"{metric}: {value}")

# 2. TEXT QUALITY METRICS
print("\n=== TEXT QUALITY METRICS ===")

# BLEU and ROUGE scores
bleu_score, rouge_score = evaluator.evaluate_bleu_rouge(candidates, references)
print(f"BLEU Score: {bleu_score:.3f}")
print(f"ROUGE-1 Score: {rouge_score:.3f}")

# BERT Score (Precision, Recall, F1)
bert_p, bert_r, bert_f1 = evaluator.evaluate_bert_score(candidates, references)
print(f"BERT Precision: {bert_p:.3f}")
print(f"BERT Recall: {bert_r:.3f}")
print(f"BERT F1: {bert_f1:.3f}")

# METEOR Score
meteor_score = evaluator.evaluate_meteor(candidates, references)
print(f"METEOR Score: {meteor_score:.3f}")

# CHRF Score
chrf_score = evaluator.evaluate_chrf(candidates, references)
print(f"CHRF Score: {chrf_score:.3f}")

# 3. LANGUAGE MODEL METRICS
print("\n=== LANGUAGE MODEL METRICS ===")

# Perplexity
perplexity = evaluator.evaluate_perplexity(generated_text)
print(f"Perplexity: {perplexity:.3f}")

# 4. DIVERSITY METRICS
print("\n=== DIVERSITY METRICS ===")

# Text diversity
diversity_score = evaluator.evaluate_diversity(candidates)
print(f"Diversity Score: {diversity_score:.3f}")

# 5. SAFETY METRICS
print("\n=== SAFETY METRICS ===")

# Bias detection
bias_score = evaluator.evaluate_bias(generated_text)
print(f"Bias Score: {bias_score:.3f}")

# 6. SEMANTIC METRICS
print("\n=== SEMANTIC METRICS ===")

# MAUVE score
mauve_score = evaluator.evaluate_mauve(references, candidates)
print(f"MAUVE Score: {mauve_score:.3f}")

# Semantic similarity
semantic_sim = evaluator.evaluate_semantic_similarity(generated_text, reference_text)
print(f"Semantic Similarity: {semantic_sim:.3f}")

# 7. READABILITY METRICS
print("\n=== READABILITY METRICS ===")

# Readability scores
flesch_ease, flesch_grade = evaluator.evaluate_readability(generated_text)
print(f"Flesch Reading Ease: {flesch_ease:.3f}")
print(f"Flesch-Kincaid Grade: {flesch_grade:.3f}")

# 8. HALLUCINATION DETECTION
print("\n=== HALLUCINATION DETECTION ===")

# NLI-based hallucination detection
nli_results = evaluator.evaluate_hallucination_nli(generated_text, reference_text)
print(f"NLI Contradiction: {nli_results['contradiction_score']:.3f}")
print(f"NLI Entailment: {nli_results['entailment_score']:.3f}")
print(f"NLI Neutral: {nli_results['neutral_score']:.3f}")
print(f"NLI Hallucination: {nli_results['hallucination_score']:.3f}")

# Entity hallucination detection
entity_hallucination = evaluator.evaluate_entity_hallucination(generated_text, reference_text)
print(f"Entity Hallucination: {entity_hallucination:.3f}")

# Numerical hallucination detection
numerical_hallucination = evaluator.evaluate_numerical_hallucination(generated_text, reference_text)
print(f"Numerical Hallucination: {numerical_hallucination:.3f}")

# 9. BATCH EVALUATION
print("\n=== BATCH EVALUATION ===")

# Multiple questions and responses
questions = [
    "What is AI?",
    "Explain machine learning",
    "What is deep learning?"
]
responses = [
    "AI is artificial intelligence used in computers",
    "Machine learning is a subset of AI that learns from data",
    "Deep learning uses neural networks with multiple layers"
]
references = [
    "Artificial intelligence",
    "Machine learning uses algorithms to learn from data",
    "Deep learning is a subset of machine learning using neural networks"
]

# Batch evaluation
batch_results = evaluator.evaluate_batch(questions, responses, references)
print(f"Evaluated {len(batch_results)} samples")

# Summary statistics
summary_stats = evaluator.get_summary_stats(batch_results)
print("\nSummary Statistics:")
for metric, stats in summary_stats.items():
    print(f"{metric}:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std: {stats['std']:.3f}")
    print(f"  Min: {stats['min']:.3f}")
    print(f"  Max: {stats['max']:.3f}")

# 10. CACHE STATUS CHECK
print("\n=== CACHE STATUS ===")
evaluator.check_cache_status()
```

### Practical Use Cases

```python
# Use Case 1: Chatbot Response Evaluation
def evaluate_chatbot_response(question, response, expected_response):
    evaluator = LLMEvaluator()
    
    results = evaluator.evaluate_all(question, response, expected_response)
    
    # Key metrics for chatbot evaluation
    quality_score = results['BERT_F1']
    hallucination_risk = results['Hallucination_Score']
    bias_risk = results['Bias_Score']
    
    print(f"Quality Score: {quality_score:.3f}")
    print(f"Hallucination Risk: {hallucination_risk:.3f}")
    print(f"Bias Risk: {bias_risk:.3f}")
    
    return results

# Use Case 2: Content Safety Check
def check_content_safety(text):
    evaluator = LLMEvaluator()
    
    bias_score = evaluator.evaluate_bias(text)
    
    if bias_score > 0.7:
        print("High bias/hate speech risk detected!")
    elif bias_score > 0.4:
        print("Moderate bias risk - review recommended")
    else:
        print("Content appears safe")
    
    return bias_score

# Use Case 3: Translation Quality Assessment
def assess_translation_quality(source, translation, reference):
    evaluator = LLMEvaluator()
    
    # Multiple quality metrics
    bleu, rouge = evaluator.evaluate_bleu_rouge([translation], [reference])
    bert_p, bert_r, bert_f1 = evaluator.evaluate_bert_score([translation], [reference])
    meteor = evaluator.evaluate_meteor([translation], [reference])
    
    print(f"BLEU: {bleu:.3f}")
    print(f"ROUGE-1: {rouge:.3f}")
    print(f"BERT F1: {bert_f1:.3f}")
    print(f"METEOR: {meteor:.3f}")
    
    return {
        'bleu': bleu,
        'rouge': rouge,
        'bert_f1': bert_f1,
        'meteor': meteor
    }

# Example usage
question = "What is the weather like today?"
response = "Today is sunny with a temperature of 75°F."
reference = "The weather today is sunny and warm."

results = evaluate_chatbot_response(question, response, reference)
```

## Available Metrics

### Text Quality Metrics
- **BLEU**: Measures n-gram overlap between generated and reference text
- **ROUGE-1**: Measures unigram overlap (recall-oriented)
- **BERT Score**: Semantic similarity using BERT embeddings
- **METEOR**: Considers synonyms and paraphrases
- **CHRF**: Character-level F-score

### Language Model Metrics
- **Perplexity**: Measures how well a language model predicts the text

### Diversity Metrics
- **Diversity**: Ratio of unique bigrams to total tokens

### Safety Metrics
- **Bias Score**: Detects potential hate speech or bias

### Semantic Metrics
- **MAUVE**: Measures similarity between text distributions

### Readability Metrics
- **Flesch Reading Ease**: Text readability score
- **Flesch-Kincaid Grade**: Grade level required to understand the text

### Hallucination Detection Metrics
- **NLI Hallucination**: Uses Natural Language Inference to detect contradictions
- **Entity Hallucination**: Detects non-existent entities in generated text
- **Numerical Hallucination**: Identifies incorrect numbers and statistics
- **Semantic Similarity**: Measures overall semantic alignment
- **Combined Hallucination Score**: Weighted combination of hallucination metrics

## Optimized Models Used

This toolkit uses small, efficient models for faster setup:

- `cross-encoder/nli-deberta-v3-small` NLI contradiction detection
- `martin-ha/toxic-comment-model` Hate speech detection  
- `distilbert-base-multilingual-cased`  Multilingual BERT scoring
- `sentence-transformers/all-MiniLM-L6-v2` Semantic embeddings



## API Reference

### LLMEvaluator

The main class for evaluating LLM outputs.

#### Core Methods

- `evaluate_all(question, response, reference)`: Evaluate all metrics for a single triplet
- `evaluate_batch(questions, responses, references)`: Evaluate multiple triplets
- `get_summary_stats(results)`: Calculate summary statistics for batch results

#### Individual Metric Methods

- `evaluate_bleu_rouge(candidates, references)`: Calculate BLEU and ROUGE scores
- `evaluate_bert_score(candidates, references)`: Calculate BERT Score
- `evaluate_perplexity(text)`: Calculate perplexity
- `evaluate_diversity(texts)`: Calculate diversity score
- `evaluate_bias(text)`: Evaluate bias/hate speech
- `evaluate_meteor(candidates, references)`: Calculate METEOR score
- `evaluate_chrf(candidates, references)`: Calculate CHRF score
- `evaluate_readability(text)`: Calculate readability metrics
- `evaluate_mauve(reference_texts, generated_texts)`: Calculate MAUVE score

#### Hallucination Detection Methods

- `evaluate_hallucination_nli(generated_text, reference_text)`: Detect hallucinations using NLI
- `evaluate_entity_hallucination(generated_text, reference_text)`: Detect entity hallucinations
- `evaluate_numerical_hallucination(generated_text, reference_text)`: Detect numerical hallucinations
- `evaluate_semantic_similarity(generated_text, reference_text)`: Calculate semantic similarity

## 🔧 Requirements

- Python 3.8+
- PyTorch
- Transformers
- NLTK
- SentenceTransformers
- Various NLP libraries (automatically installed)

## 📖 Examples

For more detailed examples, see:
- [Basic Usage Examples](https://github.com/AnSwati/llm_eval_toolkit/tree/main/examples)
- [Hallucination Detection Guide](https://github.com/AnSwati/llm_eval_toolkit/blob/main/examples/HALLUCINATION_DETECTION.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/AnSwati/llm_eval_toolkit/blob/main/LICENSE) file for details.

## 📝 Citation

If you use this library in your research, please cite:

```bibtex
@software{llm_eval_toolkit,
  title={LLM Eval Toolkit: A Comprehensive Evaluation Toolkit for Large Language Models},
  author={Swati Tyagi},
  year={2025},
  url={https://github.com/AnSwati/llm_eval_toolkit},
  note={Available on PyPI: https://pypi.org/project/llm-eval-toolkit/}
}
```

---

⭐ **Star this repository** if you find it helpful!

🐛 **Report issues** on [GitHub Issues](https://github.com/AnSwati/llm_eval_toolkit/issues)

📦 **Install from PyPI**: `pip install llm-eval-toolkit`