"""
Neural Bag of Words model for sentiment analysis.
Based on bentrevett/pytorch-sentiment-analysis tutorial.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class NBoW(nn.Module):
    """
    Neural Bag of Words model for sentiment classification.
    
    This model represents text as a bag of words, ignoring word order,
    and uses a simple neural network for classification.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        output_dim: int = 2,
        dropout_rate: float = 0.3,
        pad_index: int = 0
    ):
        """
        Initialize the Neural Bag of Words model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden layer dimension
            output_dim: Number of output classes (2 for binary classification)
            dropout_rate: Dropout rate for regularization
            pad_index: Index of padding token
        """
        super(NBoW, self).__init__()
        
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_index
        )
        
        # Hidden layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embedding weights
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        
        # Initialize linear layer weights
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            logits: Output logits [batch_size, output_dim]
        """
        # Embedding layer
        embedded = self.embedding(ids)  # [batch_size, seq_len, embedding_dim]
        
        # Average embeddings (bag of words)
        # This removes the sequential information
        pooled = embedded.mean(dim=1)  # [batch_size, embedding_dim]
        
        # Apply dropout
        pooled = self.dropout(pooled)
        
        # First hidden layer
        hidden1 = F.relu(self.fc1(pooled))
        hidden1 = self.dropout(hidden1)
        
        # Second hidden layer
        hidden2 = F.relu(self.fc2(hidden1))
        hidden2 = self.dropout(hidden2)
        
        # Output layer
        logits = self.fc3(hidden2)
        
        return logits
    
    def predict(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence scores.
        
        Args:
            ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            predictions: Predicted class indices [batch_size]
            confidence: Confidence scores [batch_size]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(ids)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            confidence = torch.max(probs, dim=-1)[0]
        
        return predictions, confidence


def create_nbow_model(vocab_size: int, **kwargs) -> NBoW:
    """
    Factory function to create a Neural Bag of Words model.
    
    Args:
        vocab_size: Size of the vocabulary
        **kwargs: Additional model parameters
        
    Returns:
        NBoW model instance
    """
    return NBoW(vocab_size, **kwargs)


if __name__ == "__main__":
    # Test the model
    vocab_size = 10000
    batch_size = 32
    seq_len = 50
    
    model = create_nbow_model(vocab_size)
    
    # Create dummy input
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test prediction
    predictions, confidence = model.predict(dummy_input)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Confidence shape: {confidence.shape}")
