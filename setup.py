# setup.py
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

class PostInstallCommand(install):
    """Post-installation with caching awareness."""
    def run(self):
        install.run(self)
        self.setup_models_with_caching()
    
    def _is_model_cached(self, model_name: str) -> bool:
        """Check if a model is available in cache."""
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            return True
        except:
            return False
    
    def setup_models_with_caching(self):
        """Download models with smart caching."""
        print("🚀 Setting up LLM Evaluator models with caching...")
        
        # Small, fast models - total ~890MB vs 2.1GB+
        models_to_download = [
            "cross-encoder/nli-deberta-v3-small",        # ~150MB - NLI tasks
            "martin-ha/toxic-comment-model",             # ~250MB - Hate detection  
            "distilbert-base-multilingual-cased",       # ~500MB - Multilingual BERT
            "sentence-transformers/all-MiniLM-L6-v2",   # ~90MB - Embeddings
            "gpt2",                                      # ~500MB - Perplexity calculation
        ]
        
        try:
            from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
            
            print("🔍 Checking cache status...")
            cached_count = 0
            download_count = 0
            
            for model_name in models_to_download:
                if self._is_model_cached(model_name):
                    print(f"✅ {model_name} - already cached")
                    cached_count += 1
                else:
                    print(f"⬇️ {model_name} - downloading...")
                    try:
                        # Download tokenizer (this caches the model)
                        AutoTokenizer.from_pretrained(model_name)
                        
                        # Download model weights
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
                        print(f"⚠️ Warning: Could not download {model_name}: {e}")
                        print(f"   Model will be downloaded on first use.")
            
            print(f"📊 Setup summary: {cached_count} cached, {download_count} downloaded")
            
            # Handle SentenceTransformers separately (has its own caching)
            try:
                from sentence_transformers import SentenceTransformer
                print("🔍 Checking SentenceTransformers cache...")
                # This will use cache if available
                SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                print("✅ SentenceTransformers model ready")
            except Exception as e:
                print(f"⚠️ SentenceTransformers setup issue: {e}")
                    
        except ImportError:
            print("⚠️ Transformers not installed. Models will be downloaded on first use.")
        
        # Download NLTK data with cache checking
        self.setup_nltk_data()
        
        print("🎉 LLM Evaluator setup complete!")
    
    def setup_nltk_data(self):
        """Download NLTK data with cache awareness."""
        try:
            import nltk
            
            nltk_data = [
                'punkt', 'stopwords', 'vader_lexicon', 
                'averaged_perceptron_tagger', 'wordnet', 'omw-1.4'
            ]
            
            print("🔍 Checking NLTK data cache...")
            
            for data in nltk_data:
                try:
                    # Check if already downloaded
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
                                try:
                                    print(f"⬇️ NLTK {data} - downloading...")
                                    nltk.download(data, quiet=True)
                                    print(f"✅ NLTK {data} - downloaded")
                                except:
                                    print(f"⚠️ Could not download NLTK {data}")
                                    
        except ImportError:
            print("⚠️ NLTK not installed.")

setup(
    name="llm_eval_toolkit",
    version="1.0.3",
    author="Swati Tyagi",
    author_email="swatyagi@udel.edu",
    description="A comprehensive evaluation toolkit for Large Language Models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AnSwati/llm_eval_toolkit",
    packages=["llm_eval_toolkit"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.8",
    install_requires=[
        "transformers>=4.21.0",
        "torch>=1.12.0",
        "nltk>=3.8",
        "sentence-transformers>=2.2.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "datasets>=2.0.0",
        "evaluate>=0.4.0",
        "rouge-score>=0.1.2",
        "bert-score>=0.3.11",
        "mauve-text>=0.3.0",
        "textstat>=0.7.0",
        "spacy>=3.4.0",
        "sacrebleu>=2.0.0",  # Added missing dependency
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    cmdclass={
        'install': PostInstallCommand,
    },
    include_package_data=True,
    zip_safe=False,
)