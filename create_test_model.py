"""
Quick test script to train and save a minimal NBoW model for testing inference.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

# Import our modules
from models.nbow import NBoW
from utils.preprocessing import TextPreprocessor, VocabularyBuilder
from utils.training import Trainer

# Create simple dataset
class SimpleDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': torch.tensor(self.texts[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def create_test_model():
    """Create a test model for inference testing."""
    print("Creating test model...")
    
    # Sample data
    texts = [
        "this product is amazing",
        "terrible quality bad",
        "great value recommend",
        "poor service disappointed",
        "excellent exceeded expectations"
    ]
    labels = [1, 0, 1, 0, 1]  # 1 = positive, 0 = negative
    
    # Preprocess
    preprocessor = TextPreprocessor()
    processed_texts = [preprocessor.preprocess_text(text) for text in texts]
    
    # Build vocabulary
    vocab_builder = VocabularyBuilder(min_freq=1)
    vocab = vocab_builder.build_from_texts(processed_texts)
    
    # Convert to indices
    max_length = 20
    text_indices = [vocab_builder.text_to_indices(tokens, max_length) for tokens in processed_texts]
    
    # Create model
    model = NBoW(
        vocab_size=len(vocab),
        embedding_dim=50,
        hidden_dim=32,
        output_dim=2,
        dropout_rate=0.1
    )
    
    # Create dataset and dataloader
    dataset = SimpleDataset(text_indices, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Train for a few epochs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(model, device=device)
    
    print("Training model...")
    history = trainer.fit(
        train_loader=dataloader,
        val_loader=dataloader,  # Using same data for validation in this test
        epochs=3,
        lr=1e-2,
        save_best_model=True
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_config = {
        'vocab_size': len(vocab),
        'embed_dim': 50,
        'hidden_dim': 32,
        'num_classes': 2,
        'dropout': 0.1,
        'max_length': max_length
    }
    
    trainer.save_model('models/nbow_model.pth', vocab=vocab, model_config=model_config)
    print("Test model saved to models/nbow_model.pth")
    
    return vocab, model_config

if __name__ == "__main__":
    create_test_model()
