"""
CNN model for sentiment analysis.
Based on bentrevett/pytorch-sentiment-analysis tutorial.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class CNNModel(nn.Module):
    """
    CNN-based model for sentiment classification.
    
    This model uses convolutional layers with multiple filter sizes
    to capture local patterns in text for sentiment analysis.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        num_filters: int = 100,
        filter_sizes: List[int] = [2, 3, 4, 5],
        output_dim: int = 2,
        dropout_rate: float = 0.3,
        pad_index: int = 0
    ):
        """
        Initialize the CNN model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            num_filters: Number of filters per filter size
            filter_sizes: List of filter sizes (n-grams)
            output_dim: Number of output classes
            dropout_rate: Dropout rate for regularization
            pad_index: Index of padding token
        """
        super(CNNModel, self).__init__()
        
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_index
        )
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=filter_size
            )
            for filter_size in filter_sizes
        ])
        
        # Fully connected layers
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, num_filters)
        self.fc2 = nn.Linear(num_filters, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embedding weights
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        
        # Initialize convolutional layer weights
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        
        # Initialize linear layer weights
        for layer in [self.fc1, self.fc2]:
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
        
        # Transpose for conv1d (expects [batch_size, embedding_dim, seq_len])
        embedded = embedded.transpose(1, 2)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution
            conv_out = F.relu(conv(embedded))  # [batch_size, num_filters, conv_seq_len]
            
            # Apply max pooling over the sequence dimension
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            pooled = pooled.squeeze(2)  # [batch_size, num_filters]
            
            conv_outputs.append(pooled)
        
        # Concatenate all conv outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, len(filter_sizes) * num_filters]
        
        # Apply dropout
        concatenated = self.dropout(concatenated)
        
        # First fully connected layer
        fc1_output = F.relu(self.fc1(concatenated))
        fc1_output = self.dropout(fc1_output)
        
        # Output layer
        logits = self.fc2(fc1_output)
        
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


def create_cnn_model(vocab_size: int, **kwargs) -> CNNModel:
    """
    Factory function to create a CNN model.
    
    Args:
        vocab_size: Size of the vocabulary
        **kwargs: Additional model parameters
        
    Returns:
        CNNModel instance
    """
    return CNNModel(vocab_size, **kwargs)


if __name__ == "__main__":
    # Test the model
    vocab_size = 10000
    batch_size = 32
    seq_len = 50
    
    model = create_cnn_model(vocab_size)
    
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
