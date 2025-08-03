# Hallucination Detection in LLM Evaluator

This document explains the hallucination detection capabilities added to the LLM Evaluator library.

## What is Hallucination?

In the context of Large Language Models (LLMs), hallucination refers to the generation of content that is factually incorrect, contradictory, or not supported by the provided context or reference information. Hallucinations can take various forms:

1. **Factual inconsistency**: Generating facts that contradict established knowledge
2. **Entity hallucination**: Inventing non-existent entities (people, organizations, etc.)
3. **Numerical hallucination**: Presenting incorrect numbers, dates, or statistics
4. **Logical inconsistency**: Producing text with internal contradictions
5. **Source attribution**: Attributing information to sources that don't exist or weren't mentioned

## Hallucination Detection Methods

The LLM Evaluator implements several complementary approaches to detect hallucinations:

### 1. NLI-based Hallucination Detection

Uses Natural Language Inference (NLI) models to detect contradictions between the generated text and the reference text. The implementation supports two NLI models:

#### Default: `facebook/bart-large-mnli`

```python
# Default model
evaluator = LLMEvaluator()
nli_results = evaluator.evaluate_hallucination_nli(generated_text, reference_text)
```

This model classifies each sentence in the generated text as:

- **Contradiction**: The generated sentence contradicts the reference (likely hallucination)
- **Entailment**: The reference supports the generated sentence (likely factual)
- **Neutral**: The reference neither supports nor contradicts the generated sentence

#### Alternative: `GuardrailsAI/finetuned_nli_provenance`

```python
# Using GuardrailsAI model
evaluator = LLMEvaluator(nli_model_name='GuardrailsAI/finetuned_nli_provenance')
nli_results = evaluator.evaluate_hallucination_nli(generated_text, reference_text)
```

This model is specifically fine-tuned for hallucination detection and provides similar classification but with potentially better performance for certain types of hallucinations.

See `examples/guardrails_nli_example.py` for a demonstration of this model.

### 2. Entity Hallucination Detection

Identifies entities (people, organizations, dates, etc.) in both the generated and reference texts, then calculates the proportion of entities in the generated text that don't appear in the reference.

```python
entity_hallucination = evaluator.evaluate_entity_hallucination(generated_text, reference_text)
```

### 3. Numerical Hallucination Detection

Specifically targets numerical information (dates, percentages, monetary values, etc.) to detect when an LLM fabricates numbers not present in the reference.

```python
numerical_hallucination = evaluator.evaluate_numerical_hallucination(generated_text, reference_text)
```

### 4. Semantic Similarity

Uses sentence embeddings to measure the overall semantic alignment between the generated and reference texts. Lower similarity may indicate hallucination.

```python
semantic_similarity = evaluator.evaluate_semantic_similarity(generated_text, reference_text)
```

### 5. Combined Hallucination Score

Provides a weighted combination of the individual hallucination metrics for an overall assessment:

```
Hallucination_Score = (
    0.4 * NLI_Hallucination + 
    0.3 * Entity_Hallucination + 
    0.2 * Numerical_Hallucination + 
    0.1 * (1.0 - Semantic_Similarity)
)
```

## Usage Examples

See the `hallucination_test.py` script for complete examples of detecting different types of hallucinations.

Basic usage:

```python
from llm_evaluator import LLMEvaluator

evaluator = LLMEvaluator()
results = evaluator.evaluate_all(question, response, reference)

# Access hallucination metrics
hallucination_score = results["Hallucination_Score"]
nli_contradiction = results["NLI_Contradiction"]
entity_hallucination = results["Entity_Hallucination"]
```

## Interpreting Results

- **Hallucination_Score**: Overall hallucination measure (0-1, higher means more hallucination)
- **NLI_Contradiction**: Degree to which the generated text contradicts the reference (0-1)
- **NLI_Entailment**: Degree to which the reference supports the generated text (0-1)
- **Entity_Hallucination**: Proportion of entities in the generated text not found in the reference (0-1)
- **Numerical_Hallucination**: Proportion of numbers in the generated text not found in the reference (0-1)
- **Semantic_Similarity**: Overall semantic alignment between texts (0-1, higher means more similar)

## Limitations

1. The entity extraction uses regex patterns rather than a full NER model, which may miss some entities or incorrectly identify others.
2. The NLI model may struggle with complex or nuanced statements.
3. Numerical comparison is based on exact matches, so different formats of the same number (e.g., "one million" vs "1,000,000") will be considered different.
4. The method works best when the reference text is comprehensive and covers all the information that should be in the response.

## Future Improvements

Potential enhancements for future versions:

1. Integration with specialized NER models for better entity extraction
2. Support for domain-specific hallucination detection (medical, legal, etc.)
3. Improved numerical comparison with normalization of different formats
4. Fact verification against external knowledge bases
5. Temporal consistency checking for event sequences