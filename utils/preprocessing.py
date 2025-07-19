"""
Text preprocessing utilities for customer review sentiment analysis.
"""

import re
import string
import spacy
from typing import List, Tuple, Dict, Optional
import torch
from collections import Counter
import numpy as np


class TextPreprocessor:
    """
    Text preprocessing utilities for customer reviews.
    """
    
    def __init__(self, language: str = "en_core_web_sm"):
        """
        Initialize the text preprocessor.
        
        Args:
            language: spaCy language model
        """
        try:
            self.nlp = spacy.load(language)
        except OSError:
            print(f"Warning: {language} model not found. Using basic tokenization.")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.nlp:
            # Use spaCy for tokenization
            doc = self.nlp(text)
            tokens = [token.text for token in doc if not token.is_space]
        else:
            # Basic tokenization
            tokens = text.split()
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        if self.nlp:
            # Use spaCy stopwords
            stopwords = self.nlp.Defaults.stop_words
            return [token for token in tokens if token.lower() not in stopwords]
        else:
            # Basic English stopwords
            basic_stopwords = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
                'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
                'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
                'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
                'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'in',
                'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
            }
            return [token for token in tokens if token.lower() not in basic_stopwords]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        if self.nlp:
            # Use spaCy for lemmatization
            doc = self.nlp(' '.join(tokens))
            return [token.lemma_ for token in doc]
        else:
            # Return tokens as is if spaCy not available
            return tokens
    
    def remove_punctuation(self, tokens: List[str]) -> List[str]:
        """
        Remove punctuation from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        # Remove tokens that are pure punctuation
        return [token for token in tokens if token not in string.punctuation]
    
    def filter_tokens(self, tokens: List[str], min_length: int = 2) -> List[str]:
        """
        Filter tokens by length and other criteria.
        
        Args:
            tokens: List of tokens
            min_length: Minimum token length
            
        Returns:
            Filtered tokens
        """
        filtered = []
        for token in tokens:
            # Skip if too short
            if len(token) < min_length:
                continue
            
            # Skip if all numbers
            if token.isdigit():
                continue
            
            # Skip if contains only punctuation
            if all(c in string.punctuation for c in token):
                continue
            
            filtered.append(token)
        
        return filtered
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True, 
                       lemmatize: bool = True, remove_punct: bool = True) -> List[str]:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text: Raw text
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize
            remove_punct: Whether to remove punctuation
            
        Returns:
            Processed tokens
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove punctuation
        if remove_punct:
            tokens = self.remove_punctuation(tokens)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        # Filter tokens
        tokens = self.filter_tokens(tokens)
        
        return tokens


class VocabularyBuilder:
    """
    Build vocabulary from text data.
    """
    
    def __init__(self, min_freq: int = 2, max_vocab_size: Optional[int] = None):
        """
        Initialize vocabulary builder.
        
        Args:
            min_freq: Minimum frequency for word inclusion
            max_vocab_size: Maximum vocabulary size
        """
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.vocab = None
        self.word_to_idx = {}
        self.idx_to_word = {}
    
    def build_from_texts(self, texts: List[List[str]], special_tokens: List[str] = None) -> Dict[str, int]:
        """
        Build vocabulary from tokenized texts.
        
        Args:
            texts: List of tokenized texts
            special_tokens: Special tokens to add
            
        Returns:
            Word to index mapping
        """
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>"]
        
        # Count word frequencies
        word_freq = Counter()
        for tokens in texts:
            word_freq.update(tokens)
        
        # Filter by frequency
        filtered_words = [word for word, freq in word_freq.items() if freq >= self.min_freq]
        
        # Sort by frequency (descending)
        filtered_words.sort(key=lambda x: word_freq[x], reverse=True)
        
        # Limit vocabulary size
        if self.max_vocab_size:
            filtered_words = filtered_words[:self.max_vocab_size - len(special_tokens)]
        
        # Create vocabulary
        vocab_words = special_tokens + filtered_words
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        return self.word_to_idx
    
    def text_to_indices(self, tokens: List[str], max_length: Optional[int] = None) -> List[int]:
        """
        Convert tokens to indices.
        
        Args:
            tokens: List of tokens
            max_length: Maximum sequence length
            
        Returns:
            List of indices
        """
        unk_idx = self.word_to_idx.get("<unk>", 1)
        indices = [self.word_to_idx.get(token, unk_idx) for token in tokens]
        
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                pad_idx = self.word_to_idx.get("<pad>", 0)
                indices.extend([pad_idx] * (max_length - len(indices)))
        
        return indices
    
    def indices_to_text(self, indices: List[int]) -> List[str]:
        """
        Convert indices back to tokens.
        
        Args:
            indices: List of indices
            
        Returns:
            List of tokens
        """
        return [self.idx_to_word.get(idx, "<unk>") for idx in indices]
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.word_to_idx)
    
    def get_pad_index(self) -> int:
        """Get padding token index."""
        return self.word_to_idx.get("<pad>", 0)
    
    def get_unk_index(self) -> int:
        """Get unknown token index."""
        return self.word_to_idx.get("<unk>", 1)


def load_glove_embeddings(vocab: Dict[str, int], embedding_dim: int = 100) -> torch.Tensor:
    """
    Load GloVe embeddings for vocabulary (simplified version without torchtext).
    
    Args:
        vocab: Word to index mapping
        embedding_dim: Embedding dimension
        
    Returns:
        Embedding tensor
    """
    # Create embedding matrix with random initialization
    # In a production environment, you would load actual GloVe vectors
    print(f"Creating random embeddings (GloVe not available without torchtext)")
    embeddings = torch.randn(len(vocab), embedding_dim) * 0.1
    
    # Set padding token to zero
    if "<pad>" in vocab:
        embeddings[vocab["<pad>"]] = torch.zeros(embedding_dim)
    
    return embeddings


if __name__ == "__main__":
    # Test text preprocessing
    print("Testing TextPreprocessor...")
    
    preprocessor = TextPreprocessor()
    
    # Test text
    sample_text = "This product is AMAZING!!! I love it so much. Best purchase ever! 😍"
    
    print(f"Original text: {sample_text}")
    print(f"Cleaned text: {preprocessor.clean_text(sample_text)}")
    print(f"Tokenized: {preprocessor.tokenize(sample_text)}")
    print(f"Preprocessed: {preprocessor.preprocess_text(sample_text)}")
    
    # Test vocabulary builder
    print("\nTesting VocabularyBuilder...")
    
    texts = [
        ["great", "product", "love", "it"],
        ["terrible", "product", "hate", "it"],
        ["good", "value", "money"],
        ["bad", "quality", "product"]
    ]
    
    vocab_builder = VocabularyBuilder(min_freq=1)
    vocab = vocab_builder.build_from_texts(texts)
    
    print(f"Vocabulary size: {vocab_builder.get_vocab_size()}")
    print(f"Sample vocabulary: {list(vocab.items())[:10]}")
    
    # Test text to indices
    test_tokens = ["great", "product", "unknown_word"]
    indices = vocab_builder.text_to_indices(test_tokens)
    print(f"Tokens to indices: {test_tokens} -> {indices}")
    
    # Test indices to text
    back_to_tokens = vocab_builder.indices_to_text(indices)
    print(f"Indices to tokens: {indices} -> {back_to_tokens}")
