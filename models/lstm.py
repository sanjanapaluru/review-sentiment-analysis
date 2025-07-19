"""
LSTM model for sentiment analysis.
Based on bentrevett/pytorch-sentiment-analysis tutorial.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LSTMModel(nn.Module):
    """
    LSTM-based model for sentiment classification.
    
    This model uses bidirectional LSTM layers to capture sequential
    dependencies in text for sentiment analysis.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        output_dim: int = 2,
        n_layers: int = 2,
        bidirectional: bool = True,
        dropout_rate: float = 0.3,
        pad_index: int = 0
    ):
        """
        Initialize the LSTM model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden layer dimension
            output_dim: Number of output classes
            n_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout_rate: Dropout rate for regularization
            pad_index: Index of padding token
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_index
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout_rate if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate final hidden dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embedding weights
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize linear layer weights
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            ids: Input token IDs [batch_size, seq_len]
            lengths: Actual sequence lengths [batch_size]
            
        Returns:
            logits: Output logits [batch_size, output_dim]
        """
        # Embedding layer
        embedded = self.embedding(ids)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # Pack sequences if lengths provided
        if lengths is not None:
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        if self.bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # First fully connected layer
        fc1_output = F.relu(self.fc1(hidden))
        fc1_output = self.dropout(fc1_output)
        
        # Output layer
        logits = self.fc2(fc1_output)
        
        return logits
    
    def predict(self, ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence scores.
        
        Args:
            ids: Input token IDs [batch_size, seq_len]
            lengths: Actual sequence lengths [batch_size]
            
        Returns:
            predictions: Predicted class indices [batch_size]
            confidence: Confidence scores [batch_size]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(ids, lengths)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            confidence = torch.max(probs, dim=-1)[0]
        
        return predictions, confidence


def create_lstm_model(vocab_size: int, **kwargs) -> LSTMModel:
    """
    Factory function to create an LSTM model.
    
    Args:
        vocab_size: Size of the vocabulary
        **kwargs: Additional model parameters
        
    Returns:
        LSTMModel instance
    """
    return LSTMModel(vocab_size, **kwargs)


if __name__ == "__main__":
    # Test the model
    vocab_size = 10000
    batch_size = 32
    seq_len = 50
    
    model = create_lstm_model(vocab_size)
    
    # Create dummy input
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_lengths = torch.randint(10, seq_len, (batch_size,))
    
    # Forward pass
    output = model(dummy_input, dummy_lengths)
    print(f"Model output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test prediction
    predictions, confidence = model.predict(dummy_input, dummy_lengths)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Confidence shape: {confidence.shape}")
