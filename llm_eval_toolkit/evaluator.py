import torch
import mauve
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.meteor_score import meteor_score
from nltk.translate.chrf_score import sentence_chrf
from textstat import flesch_reading_ease, flesch_kincaid_grade
from sklearn.metrics.pairwise import cosine_similarity
from mauve import compute_mauve
import logging
import re
import numpy as np
import os
from typing import List, Dict, Union, Tuple, Optional, Set

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMEvaluator:
    """
    A comprehensive evaluator for Large Language Model outputs using optimized small models with proper caching.
    
    This class provides various evaluation metrics including:
    
    Standard metrics:
    - BLEU, ROUGE, BERT Score: Text similarity metrics
    - Perplexity: Fluency measure
    - Diversity: Lexical diversity
    - Bias detection: Detect potential bias/hate speech
    - MAUVE: Distribution similarity between human and machine text
    - METEOR, CHRF: Machine translation metrics
    - Readability metrics: Flesch Reading Ease and Flesch-Kincaid Grade
    
    Hallucination detection:
    - NLI-based hallucination: Uses Natural Language Inference to detect contradictions
    - Entity hallucination: Detects non-existent entities
    - Numerical hallucination: Identifies incorrect numbers and statistics
    - Semantic similarity: Measures overall semantic alignment
    - Combined hallucination score: Weighted combination of hallucination metrics
    
    OPTIMIZED FOR SMALL MODELS WITH PROPER CACHING:
    - Uses cross-encoder/nli-deberta-v3-small for NLI (150MB vs 1.6GB)
    - Uses martin-ha/toxic-comment-model for bias detection (250MB vs 440MB)
    - Uses distilbert-base-multilingual-cased for BERT Score (500MB vs 700MB)
    - Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings (90MB)
    - Implements proper caching to avoid re-downloading models
    """
    
    def __init__(self, device: Optional[str] = None, nli_model_name: Optional[str] = None, 
                 bias_model_name: Optional[str] = None, bert_model_name: Optional[str] = None):
        """
        Initialize the LLM Evaluator with optimized small models and proper caching.
        
        Args:
            device (str, optional): Device to use for computations ('cpu', 'cuda', etc.).
                                  If None, automatically detects available device.
            nli_model_name (str, optional): Name of the NLI model to use for hallucination detection.
                                         Default: 'cross-encoder/nli-deberta-v3-small' (150MB, fast)
            bias_model_name (str, optional): Name of the bias detection model.
                                            Default: 'martin-ha/toxic-comment-model' (250MB, fast)
            bert_model_name (str, optional): Name of the BERT model for scoring.
                                           Default: 'distilbert-base-multilingual-cased' (500MB)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing LLMEvaluator on device: {self.device}")
        
        # Set optimized small model names
        self.nli_model_name = nli_model_name or 'cross-encoder/nli-deberta-v3-small'
        self.bias_model_name = bias_model_name or 'martin-ha/toxic-comment-model'
        self.bert_model_name = bert_model_name or 'distilbert-base-multilingual-cased'
        
        logger.info(f"🚀 Using optimized small models with caching:")
        logger.info(f"  NLI model: {self.nli_model_name} (~150MB)")
        logger.info(f"  Bias model: {self.bias_model_name} (~250MB)")
        logger.info(f"  BERT model: {self.bert_model_name} (~500MB)")
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize models
        self.gpt2_model = None
        self.gpt2_tokenizer = None
        self.bias_pipeline = None
        self.nli_model = None
        self.nli_tokenizer = None
        self.nli_pipeline = None
        self.sentence_model = None
        
        # Lazy loading flags
        self._gpt2_loaded = False
        self._bias_pipeline_loaded = False
        self._nli_model_loaded = False
        self._nli_pipeline_loaded = False
        self._sentence_model_loaded = False

    def check_cache_status(self):
        """Check which models are available in cache."""
        try:
            from transformers import AutoTokenizer
            
            models_to_check = [
                ("NLI Model", self.nli_model_name),
                ("Bias Model", self.bias_model_name), 
                ("BERT Model", self.bert_model_name),
                ("GPT-2", "gpt2"),
                ("Sentence Transformer", "sentence-transformers/all-MiniLM-L6-v2")
            ]
            
            print("🔍 Cache Status Check:")
            cached_count = 0
            for name, model_name in models_to_check:
                try:
                    AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                    print(f"✅ {name}: Cached")
                    cached_count += 1
                except:
                    print(f"❌ {name}: Not cached")
            
            print(f"📊 {cached_count}/{len(models_to_check)} models cached")
            
            # Check Hugging Face cache directory size
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            if os.path.exists(cache_dir):
                try:
                    cache_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                   for dirpath, dirnames, filenames in os.walk(cache_dir)
                                   for filename in filenames) / (1024**3)  # GB
                    print(f"📁 Cache directory size: {cache_size:.1f} GB")
                except:
                    print("📁 Cache directory exists but size calculation failed")
            else:
                print("📁 Cache directory not found")
                
        except ImportError:
            print("❌ Transformers not available for cache check")

    def _is_model_cached(self, model_name: str) -> bool:
        """Check if a model is available in cache."""
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            return True
        except:
            return False

    def _download_nltk_data(self):
        """Download required NLTK data packages."""
        required_data = ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4']
        
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}' if 'punkt' in data_name else f'corpora/{data_name}')
            except LookupError:
                try:
                    logger.info(f"Downloading NLTK {data_name}...")
                    nltk.download(data_name, quiet=True)
                except Exception as e:
                    logger.warning(f"Could not download NLTK {data_name}: {e}")

    def _load_gpt2_model(self):
        """Lazy load GPT-2 model with proper caching."""
        if not self._gpt2_loaded:
            try:
                # Check if model is cached
                if self._is_model_cached('gpt2'):
                    logger.info("✅ Loading GPT-2 from cache...")
                    self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
                    self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
                else:
                    logger.info("⬇️ Downloading GPT-2 (first time only)...")
                    self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                    self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
                
                self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
                self.gpt2_model.to(self.device)
                self._gpt2_loaded = True
                logger.info("✅ GPT-2 model ready")
                
            except Exception as e:
                logger.error(f"Error loading GPT-2: {e}")

    def _load_bias_pipeline(self):
        """Lazy load bias detection pipeline with proper caching."""
        if not self._bias_pipeline_loaded:
            try:
                # Check if model is cached
                if self._is_model_cached(self.bias_model_name):
                    logger.info(f"✅ Loading bias model from cache: {self.bias_model_name}")
                else:
                    logger.info(f"⬇️ Downloading bias model (first time only): {self.bias_model_name}")
                
                self.bias_pipeline = pipeline(
                    "text-classification", 
                    model=self.bias_model_name,
                    device=0 if self.device == 'cuda' else -1
                )
                self._bias_pipeline_loaded = True
                logger.info("✅ Bias detection pipeline ready")
                
            except Exception as e:
                logger.warning(f"Could not load bias detection model: {e}")
                # Try fallback model
                try:
                    logger.info("🔄 Trying fallback bias detection model...")
                    self.bias_pipeline = pipeline(
                        "text-classification", 
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=0 if self.device == 'cuda' else -1
                    )
                    self._bias_pipeline_loaded = True
                    logger.info("✅ Fallback bias detection ready")
                except Exception as e2:
                    logger.warning(f"Fallback bias detection also failed: {e2}")
                    self.bias_pipeline = None
                
    def _load_nli_model(self):
        """Lazy load NLI model with proper caching."""
        if not self._nli_model_loaded and not self._nli_pipeline_loaded:
            try:
                # Check if model is cached
                if self._is_model_cached(self.nli_model_name):
                    logger.info(f"✅ Loading NLI model from cache: {self.nli_model_name}")
                else:
                    logger.info(f"⬇️ Downloading NLI model (first time only): {self.nli_model_name}")
                
                # Load tokenizer and model
                self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
                self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)
                self.nli_model.to(self.device)
                self._nli_model_loaded = True
                logger.info("✅ NLI model ready")
                
            except Exception as e:
                logger.warning(f"Could not load NLI model: {e}")
                # Try fallback approach
                try:
                    logger.info("🔄 Trying NLI pipeline approach...")
                    self.nli_pipeline = pipeline(
                        "zero-shot-classification",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=0 if self.device == 'cuda' else -1
                    )
                    self._nli_pipeline_loaded = True
                    logger.info("✅ Fallback NLI pipeline ready")
                except Exception as e2:
                    logger.warning(f"Fallback NLI also failed: {e2}")
                    self.nli_model = None
                    self.nli_tokenizer = None
                    self.nli_pipeline = None
                
    def _load_sentence_model(self):
        """Lazy load sentence transformer model with proper caching."""
        if not self._sentence_model_loaded:
            try:
                model_name = 'sentence-transformers/all-MiniLM-L6-v2'
                
                # Check if model is cached (SentenceTransformers handles caching automatically)
                logger.info("Loading sentence transformer model: all-MiniLM-L6-v2")
                self.sentence_model = SentenceTransformer(model_name)
                self.sentence_model.to(self.device)
                self._sentence_model_loaded = True
                logger.info("✅ Sentence transformer model ready")
                
            except Exception as e:
                logger.warning(f"Could not load sentence transformer model: {e}")
                self.sentence_model = None

    def evaluate_bleu_rouge(self, candidates: List[str], references: List[str]) -> Tuple[float, float]:
        """
        Evaluate BLEU and ROUGE-1 scores.
        
        Args:
            candidates (List[str]): Generated texts
            references (List[str]): Reference texts
            
        Returns:
            Tuple[float, float]: BLEU score and ROUGE-1 F1 score
        """
        try:
            bleu_score = corpus_bleu(candidates, [references]).score
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
            rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
            return bleu_score, rouge1
        except Exception as e:
            logger.error(f"Error calculating BLEU/ROUGE scores: {e}")
            return 0.0, 0.0

    def evaluate_bert_score(self, candidates: List[str], references: List[str]) -> Tuple[float, float, float]:
        """
        Evaluate BERT Score using optimized multilingual DistilBERT with proper caching.
        
        Args:
            candidates (List[str]): Generated texts
            references (List[str]): Reference texts
            
        Returns:
            Tuple[float, float, float]: Precision, Recall, and F1 scores
        """
        try:
            # Use the exact model name for proper caching
            model_type = "distilbert-base-multilingual-cased"
            
            # BERT Score will handle caching automatically
            P, R, F1 = score(candidates, references, lang="en", model_type=model_type)
            return P.mean().item(), R.mean().item(), F1.mean().item()
        except Exception as e:
            logger.error(f"Error calculating BERT Score: {e}")
            return 0.0, 0.0, 0.0

    def evaluate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity using GPT-2.
        
        Args:
            text (str): Text to evaluate
            
        Returns:
            float: Perplexity score
        """
        try:
            self._load_gpt2_model()
            
            encodings = self.gpt2_tokenizer(text, return_tensors='pt')
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
            
            max_length = self.gpt2_model.config.n_positions
            stride = 512
            lls = []
            
            for i in range(0, encodings['input_ids'].size(1), stride):
                begin_loc = max(i + stride - max_length, 0)
                end_loc = min(i + stride, encodings['input_ids'].size(1))
                trg_len = end_loc - i
                input_ids = encodings['input_ids'][:, begin_loc:end_loc]
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100
                
                with torch.no_grad():
                    outputs = self.gpt2_model(input_ids, labels=target_ids)
                    log_likelihood = outputs.loss * trg_len
                lls.append(log_likelihood)
            
            ppl = torch.exp(torch.stack(lls).sum() / end_loc)
            return ppl.item()
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return float('inf')

    def evaluate_diversity(self, texts: List[str]) -> float:
        """
        Calculate diversity score based on unique bigrams.
        
        Args:
            texts (List[str]): List of texts to evaluate
            
        Returns:
            float: Diversity score (ratio of unique bigrams to total tokens)
        """
        try:
            all_tokens = [tok for text in texts for tok in text.split()]
            if not all_tokens:
                return 0.0
            unique_bigrams = set(ngrams(all_tokens, 2))
            diversity_score = len(unique_bigrams) / len(all_tokens)
            return diversity_score
        except Exception as e:
            logger.error(f"Error calculating diversity: {e}")
            return 0.0

    def evaluate_bias(self, text: str) -> float:
        """
        Evaluate potential bias/hate speech in text using small model.
        
        Args:
            text (str): Text to evaluate for bias
            
        Returns:
            float: Bias score (0-1, higher means more likely to contain hate speech)
        """
        try:
            self._load_bias_pipeline()
            if self.bias_pipeline is None:
                logger.warning("Bias detection pipeline not available")
                return 0.0
            
            # Handle different pipeline types
            if hasattr(self.bias_pipeline, 'task') and self.bias_pipeline.task == 'zero-shot-classification':
                # Zero-shot classification approach
                results = self.bias_pipeline([text], candidate_labels=["hate speech", "not hate speech"])
                bias_score = results[0]['scores'][results[0]['labels'].index('hate speech')]
            else:
                # Direct text classification
                results = self.bias_pipeline(text)
                if isinstance(results, list):
                    results = results[0]
                
                # Handle different label formats
                label = results.get('label', '').upper()
                if 'TOXIC' in label or 'HATE' in label or 'NEGATIVE' in label:
                    bias_score = results['score']
                else:
                    # Conservative approach for unclear labels
                    bias_score = 0.1
            
            return bias_score
        except Exception as e:
            logger.error(f"Error calculating bias score: {e}")
            return 0.0

    def evaluate_meteor(self, candidates: List[str], references: List[str]) -> float:
        """
        Calculate METEOR score.
        
        Args:
            candidates (List[str]): Generated texts
            references (List[str]): Reference texts
            
        Returns:
            float: Average METEOR score
        """
        try:
            meteor_scores = [
                meteor_score([word_tokenize(ref)], word_tokenize(cand))
                for ref, cand in zip(references, candidates)
            ]
            return sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
        except Exception as e:
            logger.error(f"Error calculating METEOR score: {e}")
            return 0.0
    
    def evaluate_chrf(self, candidates: List[str], references: List[str]) -> float:
        """
        Calculate CHRF score.
        
        Args:
            candidates (List[str]): Generated texts
            references (List[str]): Reference texts
            
        Returns:
            float: Average CHRF score
        """
        try:
            chrf_scores = [sentence_chrf(ref, cand) for ref, cand in zip(references, candidates)]
            return sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0.0
        except Exception as e:
            logger.error(f"Error calculating CHRF score: {e}")
            return 0.0
    
    def evaluate_readability(self, text: str) -> Tuple[float, float]:
        """
        Calculate readability metrics.
        
        Args:
            text (str): Text to evaluate
            
        Returns:
            Tuple[float, float]: Flesch Reading Ease and Flesch-Kincaid Grade
        """
        try:
            flesch_ease = flesch_reading_ease(text)
            flesch_grade = flesch_kincaid_grade(text)
            return flesch_ease, flesch_grade
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return 0.0, 0.0
        
    def evaluate_mauve(self, reference_texts: List[str], generated_texts: List[str]) -> float:
        """
        Calculate MAUVE score.
        
        Args:
            reference_texts (List[str]): Reference texts
            generated_texts (List[str]): Generated texts
            
        Returns:
            float: MAUVE score
        """
        try:
            device_id = 0 if self.device == 'cuda' else -1
            out = mauve.compute_mauve(
                p_text=reference_texts,
                q_text=generated_texts,
                device_id=device_id,
                max_text_length=1024,
                verbose=False
            )
            return out.mauve
        except Exception as e:
            logger.error(f"Error calculating MAUVE score: {e}")
            return 0.0
            
    def _extract_entities(self, text: str) -> Set[str]:
        """
        Extract named entities from text using simple regex patterns.
        
        Args:
            text (str): Text to extract entities from
            
        Returns:
            Set[str]: Set of extracted entities
        """
        patterns = [
            r'(?:[A-Z][a-z]+ ){1,4}(?:Corporation|Corp|Inc|LLC|Ltd|Company|Co|Group|Foundation|Association|Organization)',
            r'(?:[A-Z][a-z]+ ){1,3}(?:University|College|School|Institute|Academy)',
            r'(?:[A-Z][a-z]+ ){1,3}(?:Hospital|Medical Center|Clinic|Healthcare)',
            r'(?:[A-Z][a-z]+ ){1,2}(?:[A-Z][a-z]+)',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\$\d+(?:\.\d+)?(?:\s?(?:million|billion|trillion))?',
            r'\b\d+(?:\.\d+)?%\b',
            r'\b\d+(?:\.\d+)?(?:\s?(?:million|billion|trillion))\b',
        ]
        
        entities = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.update(matches)
        
        return entities
    
    def evaluate_hallucination_nli(self, generated_text: str, reference_text: str) -> Dict[str, float]:
        """
        Evaluate hallucination using small NLI model with proper caching.
        
        Args:
            generated_text (str): Generated text to evaluate
            reference_text (str): Reference text to compare against
            
        Returns:
            Dict[str, float]: Dictionary with hallucination scores
        """
        try:
            self._load_nli_model()
            
            # Check if any NLI model is available
            if (not self._nli_model_loaded and not self._nli_pipeline_loaded):
                logger.warning("NLI model not available for hallucination detection")
                return {
                    "contradiction_score": 0.0,
                    "entailment_score": 0.0,
                    "neutral_score": 0.0,
                    "hallucination_score": 0.0
                }
            
            # Split texts into sentences
            generated_sentences = sent_tokenize(generated_text)
            
            if not generated_sentences:
                return {
                    "contradiction_score": 0.0,
                    "entailment_score": 0.0,
                    "neutral_score": 0.0,
                    "hallucination_score": 0.0
                }
            
            contradiction_scores = []
            entailment_scores = []
            neutral_scores = []
            
            for sentence in generated_sentences:
                if len(sentence.split()) < 3:
                    continue
                
                if self._nli_pipeline_loaded:
                    # Using pipeline approach
                    result = self.nli_pipeline(sentence, candidate_labels=['entailment', 'contradiction', 'neutral'])
                    labels = result['labels']
                    scores = result['scores']
                    
                    entailment = scores[labels.index('entailment')] if 'entailment' in labels else 0.0
                    contradiction = scores[labels.index('contradiction')] if 'contradiction' in labels else 0.0
                    neutral = scores[labels.index('neutral')] if 'neutral' in labels else 0.0
                    
                else:
                    # Using cross-encoder model
                    inputs = self.nli_tokenizer(
                        reference_text, 
                        sentence, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=512,
                        padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.nli_model(**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
                    
                    # Handle different label arrangements
                    if predictions.shape[1] == 3:
                        contradiction = predictions[0, 0].item()
                        neutral = predictions[0, 1].item()
                        entailment = predictions[0, 2].item()
                    else:
                        entailment = predictions[0, -1].item()
                        contradiction = predictions[0, 0].item()
                        neutral = 1.0 - entailment - contradiction
                
                contradiction_scores.append(contradiction)
                neutral_scores.append(neutral)
                entailment_scores.append(entailment)
            
            # Calculate average scores
            avg_contradiction = sum(contradiction_scores) / len(contradiction_scores) if contradiction_scores else 0.0
            avg_entailment = sum(entailment_scores) / len(entailment_scores) if entailment_scores else 0.0
            avg_neutral = sum(neutral_scores) / len(neutral_scores) if neutral_scores else 0.0
            
            return {
                "contradiction_score": avg_contradiction,
                "entailment_score": avg_entailment,
                "neutral_score": avg_neutral,
                "hallucination_score": avg_contradiction
            }
        except Exception as e:
            logger.error(f"Error calculating hallucination with NLI: {e}")
            return {
                "contradiction_score": 0.0,
                "entailment_score": 0.0,
                "neutral_score": 0.0,
                "hallucination_score": 0.0
            }
    
    def evaluate_entity_hallucination(self, generated_text: str, reference_text: str) -> float:
        """
        Evaluate entity hallucination by comparing entities in generated and reference texts.
        
        Args:
            generated_text (str): Generated text to evaluate
            reference_text (str): Reference text to compare against
            
        Returns:
            float: Entity hallucination score (0-1, higher means more hallucinated entities)
        """
        try:
            generated_entities = self._extract_entities(generated_text)
            reference_entities = self._extract_entities(reference_text)
            
            if not generated_entities:
                return 0.0
            
            hallucinated_entities = generated_entities - reference_entities
            hallucination_score = len(hallucinated_entities) / len(generated_entities) if generated_entities else 0.0
            
            return hallucination_score
        except Exception as e:
            logger.error(f"Error calculating entity hallucination: {e}")
            return 0.0
    
    def evaluate_numerical_hallucination(self, generated_text: str, reference_text: str) -> float:
        """
        Evaluate numerical hallucination by comparing numbers in generated and reference texts.
        
        Args:
            generated_text (str): Generated text to evaluate
            reference_text (str): Reference text to compare against
            
        Returns:
            float: Numerical hallucination score (0-1, higher means more hallucinated numbers)
        """
        try:
            number_pattern = r'\b\d+(?:,\d+)*(?:\.\d+)?(?:\s?(?:million|billion|trillion|thousand|percent|%))?\b'
            
            generated_numbers = set(re.findall(number_pattern, generated_text))
            reference_numbers = set(re.findall(number_pattern, reference_text))
            
            if not generated_numbers:
                return 0.0
            
            hallucinated_numbers = generated_numbers - reference_numbers
            hallucination_score = len(hallucinated_numbers) / len(generated_numbers) if generated_numbers else 0.0
            
            return hallucination_score
        except Exception as e:
            logger.error(f"Error calculating numerical hallucination: {e}")
            return 0.0
    
    
    def evaluate_chrf(self, candidates: List[str], references: List[str]) -> float:
        """
        Calculate CHRF score.
        
        Args:
            candidates (List[str]): Generated texts
            references (List[str]): Reference texts
            
        Returns:
            float: Average CHRF score
        """
        try:
            chrf_scores = [sentence_chrf(ref, cand) for ref, cand in zip(references, candidates)]
            return sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0.0
        except Exception as e:
            logger.error(f"Error calculating CHRF score: {e}")
            return 0.0
    
    def evaluate_readability(self, text: str) -> Tuple[float, float]:
        """
        Calculate readability metrics.
        
        Args:
            text (str): Text to evaluate
            
        Returns:
            Tuple[float, float]: Flesch Reading Ease and Flesch-Kincaid Grade
        """
        try:
            flesch_ease = flesch_reading_ease(text)
            flesch_grade = flesch_kincaid_grade(text)
            return flesch_ease, flesch_grade
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return 0.0, 0.0
        
    def evaluate_mauve(self, reference_texts: List[str], generated_texts: List[str]) -> float:
        """
        Calculate MAUVE score.
        
        Args:
            reference_texts (List[str]): Reference texts
            generated_texts (List[str]): Generated texts
            
        Returns:
            float: MAUVE score
        """
        try:
            device_id = 0 if self.device == 'cuda' else -1
            out = mauve.compute_mauve(
                p_text=reference_texts,
                q_text=generated_texts,
                device_id=device_id,
                max_text_length=1024,
                verbose=False
            )
            return out.mauve
        except Exception as e:
            logger.error(f"Error calculating MAUVE score: {e}")
            return 0.0
            
    def _extract_entities(self, text: str) -> Set[str]:
        """
        Extract named entities from text using simple regex patterns.
        
        Args:
            text (str): Text to extract entities from
            
        Returns:
            Set[str]: Set of extracted entities
        """
        # Simple regex patterns for entity extraction
        # This is a basic implementation - could be improved with NER models
        patterns = [
            r'(?:[A-Z][a-z]+ ){1,4}(?:Corporation|Corp|Inc|LLC|Ltd|Company|Co|Group|Foundation|Association|Organization)',  # Organizations
            r'(?:[A-Z][a-z]+ ){1,3}(?:University|College|School|Institute|Academy)',  # Educational institutions
            r'(?:[A-Z][a-z]+ ){1,3}(?:Hospital|Medical Center|Clinic|Healthcare)',  # Medical institutions
            r'(?:[A-Z][a-z]+ ){1,2}(?:[A-Z][a-z]+)',  # Potential person names
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b',  # Dates
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # Date format MM/DD/YYYY
            r'\b\d{4}-\d{2}-\d{2}\b',  # Date format YYYY-MM-DD
            r'\$\d+(?:\.\d+)?(?:\s?(?:million|billion|trillion))?',  # Money amounts
            r'\b\d+(?:\.\d+)?%\b',  # Percentages
            r'\b\d+(?:\.\d+)?(?:\s?(?:million|billion|trillion))\b',  # Large numbers
        ]
        
        entities = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.update(matches)
        
        return entities
    
    def evaluate_hallucination_nli(self, generated_text: str, reference_text: str) -> Dict[str, float]:
        """
        Evaluate hallucination using small NLI model.
        
        Args:
            generated_text (str): Generated text to evaluate
            reference_text (str): Reference text to compare against
            
        Returns:
            Dict[str, float]: Dictionary with hallucination scores
        """
        try:
            self._load_nli_model()
            
            # Check if any NLI model is available
            if (not self._nli_model_loaded and not self._nli_pipeline_loaded):
                logger.warning("NLI model not available for hallucination detection")
                return {
                    "contradiction_score": 0.0,
                    "entailment_score": 0.0,
                    "neutral_score": 0.0,
                    "hallucination_score": 0.0
                }
            
            # Split texts into sentences
            generated_sentences = sent_tokenize(generated_text)
            
            # Skip if no sentences
            if not generated_sentences:
                return {
                    "contradiction_score": 0.0,
                    "entailment_score": 0.0,
                    "neutral_score": 0.0,
                    "hallucination_score": 0.0
                }
            
            # For each generated sentence, check if it's entailed by the reference
            contradiction_scores = []
            entailment_scores = []
            neutral_scores = []
            
            for sentence in generated_sentences:
                # Skip very short sentences
                if len(sentence.split()) < 3:
                    continue
                
                # Different processing based on model type
                if self._nli_pipeline_loaded:
                    # Using pipeline approach
                    if hasattr(self.nli_pipeline, 'task') and self.nli_pipeline.task == 'zero-shot-classification':
                        # Zero-shot classification
                        result = self.nli_pipeline(sentence, candidate_labels=['entailment', 'contradiction', 'neutral'])
                        labels = result['labels']
                        scores = result['scores']
                        
                        # Map scores to our format
                        entailment = scores[labels.index('entailment')] if 'entailment' in labels else 0.0
                        contradiction = scores[labels.index('contradiction')] if 'contradiction' in labels else 0.0
                        neutral = scores[labels.index('neutral')] if 'neutral' in labels else 0.0
                    else:
                        # Regular text classification
                        result = self.nli_pipeline({'text': reference_text, 'text_pair': sentence})
                        if isinstance(result, list):
                            result = result[0]
                        
                        label = result['label'].lower()
                        score = result['score']
                        
                        # Map the label to scores
                        if 'entail' in label:
                            entailment = score
                            contradiction = 0.0
                            neutral = 1.0 - score
                        elif 'contradict' in label:
                            entailment = 0.0
                            contradiction = score
                            neutral = 1.0 - score
                        else:  # neutral
                            entailment = 0.0
                            contradiction = 0.0
                            neutral = score
                else:
                    # Using cross-encoder model with traditional approach
                    # Prepare input for NLI model
                    inputs = self.nli_tokenizer(
                        reference_text, 
                        sentence, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=512,
                        padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get NLI predictions
                    with torch.no_grad():
                        outputs = self.nli_model(**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
                    
                    # Extract scores (assuming 3 labels: contradiction, neutral, entailment)
                    if predictions.shape[1] == 3:
                        contradiction = predictions[0, 0].item()
                        neutral = predictions[0, 1].item()
                        entailment = predictions[0, 2].item()
                    else:
                        # Fallback for different label arrangements
                        entailment = predictions[0, -1].item()  # Last is usually entailment
                        contradiction = predictions[0, 0].item()  # First is usually contradiction
                        neutral = 1.0 - entailment - contradiction
                
                contradiction_scores.append(contradiction)
                neutral_scores.append(neutral)
                entailment_scores.append(entailment)
            
            # Calculate average scores
            avg_contradiction = sum(contradiction_scores) / len(contradiction_scores) if contradiction_scores else 0.0
            avg_entailment = sum(entailment_scores) / len(entailment_scores) if entailment_scores else 0.0
            avg_neutral = sum(neutral_scores) / len(neutral_scores) if neutral_scores else 0.0
            
            # Hallucination score is based on contradiction score
            hallucination_score = avg_contradiction
            
            return {
                "contradiction_score": avg_contradiction,
                "entailment_score": avg_entailment,
                "neutral_score": avg_neutral,
                "hallucination_score": hallucination_score
            }
        except Exception as e:
            logger.error(f"Error calculating hallucination with NLI: {e}")
            return {
                "contradiction_score": 0.0,
                "entailment_score": 0.0,
                "neutral_score": 0.0,
                "hallucination_score": 0.0
            }
    
    def evaluate_entity_hallucination(self, generated_text: str, reference_text: str) -> float:
        """
        Evaluate entity hallucination by comparing entities in generated and reference texts.
        
        Args:
            generated_text (str): Generated text to evaluate
            reference_text (str): Reference text to compare against
            
        Returns:
            float: Entity hallucination score (0-1, higher means more hallucinated entities)
        """
        try:
            # Extract entities from both texts
            generated_entities = self._extract_entities(generated_text)
            reference_entities = self._extract_entities(reference_text)
            
            # Skip if no entities found
            if not generated_entities:
                return 0.0
            
            # Calculate hallucinated entities (those in generated but not in reference)
            hallucinated_entities = generated_entities - reference_entities
            
            # Calculate hallucination score
            hallucination_score = len(hallucinated_entities) / len(generated_entities) if generated_entities else 0.0
            
            return hallucination_score
        except Exception as e:
            logger.error(f"Error calculating entity hallucination: {e}")
            return 0.0
    
    def evaluate_numerical_hallucination(self, generated_text: str, reference_text: str) -> float:
        """
        Evaluate numerical hallucination by comparing numbers in generated and reference texts.
        
        Args:
            generated_text (str): Generated text to evaluate
            reference_text (str): Reference text to compare against
            
        Returns:
            float: Numerical hallucination score (0-1, higher means more hallucinated numbers)
        """
        try:
            # Extract numbers using regex
            number_pattern = r'\b\d+(?:,\d+)*(?:\.\d+)?(?:\s?(?:million|billion|trillion|thousand|percent|%))?\b'
            
            generated_numbers = set(re.findall(number_pattern, generated_text))
            reference_numbers = set(re.findall(number_pattern, reference_text))
            
            # Skip if no numbers found
            if not generated_numbers:
                return 0.0
            
            # Calculate hallucinated numbers (those in generated but not in reference)
            hallucinated_numbers = generated_numbers - reference_numbers
            
            # Calculate hallucination score
            hallucination_score = len(hallucinated_numbers) / len(generated_numbers) if generated_numbers else 0.0
            
            return hallucination_score
        except Exception as e:
            logger.error(f"Error calculating numerical hallucination: {e}")
            return 0.0
    
    def evaluate_semantic_similarity(self, generated_text: str, reference_text: str) -> float:
        """
        Evaluate semantic similarity between generated and reference texts using sentence embeddings.
        
        Args:
            generated_text (str): Generated text to evaluate
            reference_text (str): Reference text to compare against
            
        Returns:
            float: Semantic similarity score (0-1, higher means more similar)
        """
        try:
            self._load_sentence_model()
            
            if self.sentence_model is None:
                logger.warning("Sentence transformer model not available for semantic similarity")
                return 0.0
            
            # Generate embeddings
            generated_embedding = self.sentence_model.encode(generated_text, convert_to_tensor=True)
            reference_embedding = self.sentence_model.encode(reference_text, convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(generated_embedding.unsqueeze(0), reference_embedding.unsqueeze(0))
            
            return similarity.item()
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
        
    def evaluate_all(self, question: str, response: str, reference: str) -> Dict[str, float]:
        """
        Evaluate all metrics for a single question-response-reference triplet.
        
        Args:
            question (str): The input question/prompt
            response (str): Generated response
            reference (str): Reference/ground truth response
            
        Returns:
            Dict[str, float]: Dictionary containing all evaluation metrics
        """
        candidates = [response]
        references = [reference]
        
        logger.info("Calculating evaluation metrics...")
        
        # Calculate standard metrics
        bleu, rouge1 = self.evaluate_bleu_rouge(candidates, references)
        bert_p, bert_r, bert_f1 = self.evaluate_bert_score(candidates, references)
        perplexity = self.evaluate_perplexity(response)
        diversity = self.evaluate_diversity(candidates)
        bias_score = self.evaluate_bias(response)
        mauve_score = self.evaluate_mauve([reference], [response])
        meteor = self.evaluate_meteor(candidates, references)
        chrf = self.evaluate_chrf(candidates, references)
        flesch_ease, flesch_grade = self.evaluate_readability(response)
        
        # Calculate hallucination metrics
        logger.info("Calculating hallucination metrics...")
        nli_results = self.evaluate_hallucination_nli(response, reference)
        entity_hallucination = self.evaluate_entity_hallucination(response, reference)
        numerical_hallucination = self.evaluate_numerical_hallucination(response, reference)
        semantic_similarity = self.evaluate_semantic_similarity(response, reference)
        
        # Combine all metrics
        results = {
            "BLEU": bleu,
            "ROUGE-1": rouge1,
            "BERT_Precision": bert_p,
            "BERT_Recall": bert_r,
            "BERT_F1": bert_f1,
            "Perplexity": perplexity,
            "Diversity": diversity,
            "Bias_Score": bias_score,
            "MAUVE": mauve_score,
            "METEOR": meteor,
            "CHRF": chrf,
            "Flesch_Reading_Ease": flesch_ease,
            "Flesch_Kincaid_Grade": flesch_grade,
            
            # Hallucination metrics
            "NLI_Contradiction": nli_results["contradiction_score"],
            "NLI_Entailment": nli_results["entailment_score"],
            "NLI_Neutral": nli_results["neutral_score"],
            "NLI_Hallucination": nli_results["hallucination_score"],
            "Entity_Hallucination": entity_hallucination,
            "Numerical_Hallucination": numerical_hallucination,
            "Semantic_Similarity": semantic_similarity,
            
            # Combined hallucination score (weighted average of different hallucination metrics)
            "Hallucination_Score": (
                0.4 * nli_results["hallucination_score"] + 
                0.3 * entity_hallucination + 
                0.2 * numerical_hallucination + 
                0.1 * (1.0 - semantic_similarity)  # Invert similarity for hallucination
            )
        }
        
        return results

    def evaluate_batch(self, questions: List[str], responses: List[str], 
                      references: List[str]) -> List[Dict[str, float]]:
        """
        Evaluate multiple question-response-reference triplets.
        
        Args:
            questions (List[str]): List of input questions/prompts
            responses (List[str]): List of generated responses
            references (List[str]): List of reference/ground truth responses
            
        Returns:
            List[Dict[str, float]]: List of evaluation results for each triplet
        """
        if not (len(questions) == len(responses) == len(references)):
            raise ValueError("All input lists must have the same length")
        
        results = []
        for i, (q, r, ref) in enumerate(zip(questions, responses, references)):
            logger.info(f"Evaluating sample {i+1}/{len(questions)}")
            result = self.evaluate_all(q, r, ref)
            results.append(result)
        
        return results

    def get_summary_stats(self, results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate summary statistics for batch evaluation results.
        
        Args:
            results (List[Dict[str, float]]): List of evaluation results
            
        Returns:
            Dict[str, Dict[str, float]]: Summary statistics (mean, std, min, max) for each metric
        """
        import numpy as np
        
        if not results:
            return {}
        
        metrics = results[0].keys()
        summary = {}
        
        for metric in metrics:
            values = [result[metric] for result in results if not np.isnan(result[metric]) and not np.isinf(result[metric])]
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
            else:
                summary[metric] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0
                }
        
        return summary


