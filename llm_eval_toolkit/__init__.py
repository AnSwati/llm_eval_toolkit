"""
LLM Evaluator - A comprehensive evaluation toolkit for Large Language Models

This package provides various metrics for evaluating LLM outputs including:
- BLEU, ROUGE scores
- BERT Score
- Perplexity
- Diversity metrics
- Bias detection
- MAUVE score
- METEOR, CHRF scores
- Readability metrics
"""

# llm_eval_toolkit/__init__.py
"""
LLM Eval Toolkit Package
Handles automatic model downloading with proper caching
"""

import os
import warnings
from typing import Dict, Any

def _is_model_cached(model_name: str) -> bool:
    """Check if a model is available in cache."""
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        return True
    except:
        return False

def download_required_models():
    """Download all required models if not already present - with smart caching"""
    
    required_models = [
        "cross-encoder/nli-deberta-v3-small",        # ~150MB - NLI/contradiction detection
        "martin-ha/toxic-comment-model",             # ~250MB - Hate speech detection  
        "distilbert-base-multilingual-cased",       # ~500MB - Multilingual BERT
        "sentence-transformers/all-MiniLM-L6-v2",   # ~90MB - Semantic embeddings
        "gpt2",                                      # ~500MB - Perplexity calculation
    ]
    
    try:
        from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
        import torch
        
        print("🔍 Checking model cache status...")
        
        cached_count = 0
        download_count = 0
        
        for model_name in required_models:
            if _is_model_cached(model_name):
                print(f"✅ {model_name} - already cached")
                cached_count += 1
            else:
                print(f"⬇️ {model_name} - downloading...")
                try:
                    # Download tokenizer (this will cache the model)
                    AutoTokenizer.from_pretrained(model_name)
                    
                    # Try to download model weights
                    try:
                        AutoModelForSequenceClassification.from_pretrained(model_name)
                    except:
                        try:
                            AutoModel.from_pretrained(model_name)
                        except:
                            print(f"⚠️ Could not download model weights for {model_name}")
                            continue
                    
                    print(f"✅ {model_name} - downloaded and cached")
                    download_count += 1
                    
                except Exception as e:
                    print(f"⚠️ {model_name} - download failed: {e}")
        
        print(f"📊 Cache summary: {cached_count} cached, {download_count} downloaded")
        
        # Check SentenceTransformers separately (it has its own caching)
        try:
            from sentence_transformers import SentenceTransformer
            # This will use cache if available
            SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception as e:
            print(f"⚠️ SentenceTransformers caching issue: {e}")
                    
    except ImportError as e:
        warnings.warn(f"Could not import transformers: {e}")

def download_nltk_data():
    """Download required NLTK data with caching awareness"""
    try:
        import nltk
        
        required_nltk = [
            'punkt',                        # Sentence tokenization
            'stopwords',                    # Stop word filtering
            'vader_lexicon',               # Sentiment analysis
            'averaged_perceptron_tagger',  # POS tagging
            'wordnet',                     # WordNet corpus
            'omw-1.4'                      # Open Multilingual Wordnet
        ]
        
        print("🔍 Checking NLTK data cache...")
        
        for data in required_nltk:
            try:
                # Check different NLTK data locations
                nltk.data.find(f'tokenizers/{data}')
                print(f"✅ NLTK {data} - cached")
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{data}')
                    print(f"✅ NLTK {data} - cached")
                except LookupError:
                    try:
                        nltk.data.find(f'taggers/{data}')
                        print(f"✅ NLTK {data} - cached")
                    except LookupError:
                        try:
                            nltk.data.find(f'sentiment/{data}')
                            print(f"✅ NLTK {data} - cached")
                        except LookupError:
                            print(f"⬇️ NLTK {data} - downloading...")
                            nltk.download(data, quiet=True)
                        
    except ImportError:
        warnings.warn("NLTK not installed")

# Auto-download on import (only on first import)
_models_downloaded = False

def ensure_models():
    """Ensure all models are downloaded with smart caching"""
    global _models_downloaded
    if not _models_downloaded:
        print("🚀 LLM Evaluator: Checking models and cache...")
        download_required_models()
        download_nltk_data()
        _models_downloaded = True
        print("🎉 LLM Evaluator setup complete!")

# Import main classes
from .evaluator import LLMEvaluator

# Ensure models are available when the package is imported
ensure_models()



__version__ = "1.0"
__author__ = "Swati Tyagi"
__email__ = "swatyagi@udel.edu"
__all__ = ["LLMEvaluator"]
# llm_eval_toolkit/__init__.py

"""This module initializes the LLM Eval Toolkit package, ensuring all required models and NLTK data are downloaded and cached properly."""