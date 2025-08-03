# llm_evaluator/cache_utils.py
"""
Caching utilities for LLM Evaluator
"""

import os
import shutil
from typing import List, Dict, Tuple

def get_cache_info() -> Dict[str, any]:
    """Get comprehensive cache information."""
    cache_info = {
        "transformers_cache": None,
        "sentence_transformers_cache": None,
        "nltk_cache": None,
        "total_size_gb": 0.0,
        "available_models": []
    }
    
    try:
        # Transformers cache
        transformers_cache = os.path.expanduser("~/.cache/huggingface")
        if os.path.exists(transformers_cache):
            size = get_directory_size(transformers_cache)
            cache_info["transformers_cache"] = {
                "path": transformers_cache,
                "size_gb": size,
                "exists": True
            }
            cache_info["total_size_gb"] += size
        
        # SentenceTransformers cache  
        st_cache = os.path.expanduser("~/.cache/torch/sentence_transformers")
        if os.path.exists(st_cache):
            size = get_directory_size(st_cache)
            cache_info["sentence_transformers_cache"] = {
                "path": st_cache,
                "size_gb": size,
                "exists": True
            }
            cache_info["total_size_gb"] += size
        
        # NLTK cache
        try:
            import nltk
            nltk_data_path = nltk.data.find(".")
            if os.path.exists(nltk_data_path):
                size = get_directory_size(nltk_data_path)
                cache_info["nltk_cache"] = {
                    "path": nltk_data_path,
                    "size_gb": size,
                    "exists": True
                }
                cache_info["total_size_gb"] += size
        except:
            pass
            
    except Exception as e:
        print(f"Error getting cache info: {e}")
    
    return cache_info

def get_directory_size(path: str) -> float:
    """Get directory size in GB."""
    try:
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(path)
            for filename in filenames
        )
        return total_size / (1024**3)  # Convert to GB
    except:
        return 0.0

def clear_model_cache(confirm: bool = False) -> bool:
    """Clear all model caches."""
    if not confirm:
        print("⚠️  This will delete all cached models!")
        print("To confirm, call: clear_model_cache(confirm=True)")
        return False
    
    cache_paths = [
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.cache/torch/sentence_transformers"),
    ]
    
    cleared = []
    for cache_path in cache_paths:
        if os.path.exists(cache_path):
            try:
                shutil.rmtree(cache_path)
                cleared.append(cache_path)
                print(f"✅ Cleared: {cache_path}")
            except Exception as e:
                print(f"❌ Failed to clear {cache_path}: {e}")
    
    if cleared:
        print("🧹 Cache cleared! Models will be re-downloaded on next use.")
        return True
    else:
        print("🤷 No cache found to clear.")
        return False

def check_available_models() -> List[str]:
    """Check which models are available in cache."""
    available_models = []
    
    required_models = [
        "cross-encoder/nli-deberta-v3-small",
        "martin-ha/toxic-comment-model",
        "distilbert-base-multilingual-cased",
        "sentence-transformers/all-MiniLM-L6-v2",
        "gpt2"
    ]
    
    try:
        from transformers import AutoTokenizer
        
        for model_name in required_models:
            try:
                AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                available_models.append(model_name)
            except:
                pass
                
    except ImportError:
        pass
    
    return available_models

def print_cache_summary():
    """Print a comprehensive cache summary."""
    print("📋 LLM Evaluator Cache Summary")
    print("=" * 50)
    
    cache_info = get_cache_info()
    
    print(f"💾 Total cache size: {cache_info['total_size_gb']:.1f} GB")
    
    if cache_info["transformers_cache"]:
        tc = cache_info["transformers_cache"]
        print(f"🤗 Transformers cache: {tc['size_gb']:.1f} GB at {tc['path']}")
    
    if cache_info["sentence_transformers_cache"]:
        stc = cache_info["sentence_transformers_cache"]
        print(f"📝 SentenceTransformers cache: {stc['size_gb']:.1f} GB at {stc['path']}")
    
    if cache_info["nltk_cache"]:
        nc = cache_info["nltk_cache"]
        print(f"📚 NLTK cache: {nc['size_gb']:.1f} GB at {nc['path']}")
    
    available_models = check_available_models()
    print(f"🎯 Available models: {len(available_models)}/5")
    
    for model in available_models:
        print(f"  ✅ {model}")
    
    missing_models = [m for m in [
        "cross-encoder/nli-deberta-v3-small",
        "martin-ha/toxic-comment-model", 
        "distilbert-base-multilingual-cased",
        "sentence-transformers/all-MiniLM-L6-v2",
        "gpt2"
    ] if m not in available_models]
    
    if missing_models:
        print("❌ Missing models:")
        for model in missing_models:
            print(f"  ❌ {model}")