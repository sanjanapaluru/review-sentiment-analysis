"""
Transformer model for sentiment analysis.
Based on bentrevett/pytorch-sentiment-analysis tutorial.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Tuple, Optional


class TransformerModel(nn.Module):
    """
    Transformer-based model for sentiment classification.
    
    This model uses a pre-trained transformer (BERT) as the base
    and adds a classification head for sentiment analysis.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dim: int = 2,
        dropout_rate: float = 0.3,
        freeze_transformer: bool = False
    ):
        """
        Initialize the Transformer model.
        
        Args:
            model_name: Name of the pre-trained transformer model
            output_dim: Number of output classes
            dropout_rate: Dropout rate for regularization
            freeze_transformer: Whether to freeze transformer weights
        """
        super(TransformerModel, self).__init__()
        
        self.model_name = model_name
        self.freeze_transformer = freeze_transformer
        
        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze transformer weights if specified
        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Get transformer hidden size
        hidden_size = self.transformer.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_dim)
        )
        
        # Initialize classification head weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights."""
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            logits: Output logits [batch_size, output_dim]
        """
        # Pass through transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Pass through classification head
        logits = self.classifier(cls_output)
        
        return logits
    
    def predict(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence scores.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            predictions: Predicted class indices [batch_size]
            confidence: Confidence scores [batch_size]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            confidence = torch.max(probs, dim=-1)[0]
        
        return predictions, confidence


class TransformerTokenizer:
    """
    Wrapper for transformer tokenizer with preprocessing utilities.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        """
        Initialize the tokenizer.
        
        Args:
            model_name: Name of the pre-trained model
            max_length: Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def tokenize(self, texts: list, return_tensors: str = "pt") -> dict:
        """
        Tokenize a list of texts.
        
        Args:
            texts: List of text strings
            return_tensors: Return format ("pt" for PyTorch tensors)
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors
        )
        
        return encoded
    
    def decode(self, token_ids: torch.Tensor) -> list:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs tensor
            
        Returns:
            List of decoded text strings
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def create_transformer_model(model_name: str = "bert-base-uncased", **kwargs) -> TransformerModel:
    """
    Factory function to create a Transformer model.
    
    Args:
        model_name: Name of the pre-trained transformer model
        **kwargs: Additional model parameters
        
    Returns:
        TransformerModel instance
    """
    return TransformerModel(model_name, **kwargs)


def create_transformer_tokenizer(model_name: str = "bert-base-uncased", **kwargs) -> TransformerTokenizer:
    """
    Factory function to create a Transformer tokenizer.
    
    Args:
        model_name: Name of the pre-trained transformer model
        **kwargs: Additional tokenizer parameters
        
    Returns:
        TransformerTokenizer instance
    """
    return TransformerTokenizer(model_name, **kwargs)


if __name__ == "__main__":
    # Test the model
    model_name = "bert-base-uncased"
    batch_size = 4
    
    # Create model and tokenizer
    model = create_transformer_model(model_name)
    tokenizer = create_transformer_tokenizer(model_name)
    
    # Test data
    texts = [
        "This product is amazing!",
        "I hate this terrible product.",
        "It's okay, nothing special.",
        "Best purchase ever made!"
    ]
    
    # Tokenize
    encoded = tokenizer.tokenize(texts)
    
    # Forward pass
    output = model(encoded['input_ids'], encoded['attention_mask'])
    print(f"Model output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test prediction
    predictions, confidence = model.predict(encoded['input_ids'], encoded['attention_mask'])
    print(f"Predictions: {predictions}")
    print(f"Confidence: {confidence}")
