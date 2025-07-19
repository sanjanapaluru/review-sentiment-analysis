"""
Train Neural Bag of Words (NBoW) model for customer review sentiment analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse
from pathlib import Path

# Import our modules
from models.nbow import NBoW
from utils.data_loader import ReviewDataset, CustomDataLoader
from utils.preprocessing import TextPreprocessor, VocabularyBuilder
from utils.training import Trainer, calculate_class_weights
from utils.visualization import plot_training_history, plot_confusion_matrix, plot_data_distribution


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train NBoW model for sentiment analysis')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/sample_reviews.csv',
                       help='Path to the dataset')
    parser.add_argument('--text_column', type=str, default='review_text',
                       help='Name of the text column')
    parser.add_argument('--label_column', type=str, default='sentiment',
                       help='Name of the label column')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=100,
                       help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden layer dimension')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Preprocessing arguments
    parser.add_argument('--max_length', type=int, default=100,
                       help='Maximum sequence length')
    parser.add_argument('--min_freq', type=int, default=2,
                       help='Minimum word frequency')
    parser.add_argument('--max_vocab_size', type=int, default=10000,
                       help='Maximum vocabulary size')
    
    # Other arguments
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='Validation set size')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--save_model', action='store_true',
                       help='Save the trained model')
    parser.add_argument('--model_path', type=str, default='models/nbow_model.pth',
                       help='Path to save the model')
    
    return parser.parse_args()


def load_and_preprocess_data(args):
    """Load and preprocess the data."""
    print("Loading data...")
    
    # Load data
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} samples")
    
    # Check for missing values
    print(f"Missing values: {df.isnull().sum().sum()}")
    df = df.dropna()
    
    # Extract texts and labels
    texts = df[args.text_column].astype(str).tolist()
    labels = df[args.label_column].tolist()
    
    # Map labels to integers if they're strings
    if isinstance(labels[0], str):
        unique_labels = sorted(list(set(labels)))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        labels = [label_to_idx[label] for label in labels]
        print(f"Label mapping: {label_to_idx}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=args.test_size, random_state=args.random_state, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=args.val_size/(1-args.test_size), 
        random_state=args.random_state, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess texts
    print("Preprocessing texts...")
    X_train_processed = [preprocessor.preprocess_text(text) for text in X_train]
    X_val_processed = [preprocessor.preprocess_text(text) for text in X_val]
    X_test_processed = [preprocessor.preprocess_text(text) for text in X_test]
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab_builder = VocabularyBuilder(min_freq=args.min_freq, max_vocab_size=args.max_vocab_size)
    vocab = vocab_builder.build_from_texts(X_train_processed)
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Convert to indices
    X_train_indices = [vocab_builder.text_to_indices(tokens, args.max_length) 
                      for tokens in X_train_processed]
    X_val_indices = [vocab_builder.text_to_indices(tokens, args.max_length) 
                    for tokens in X_val_processed]
    X_test_indices = [vocab_builder.text_to_indices(tokens, args.max_length) 
                     for tokens in X_test_processed]
    
    return {
        'X_train': X_train_indices,
        'X_val': X_val_indices,
        'X_test': X_test_indices,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'vocab': vocab,
        'vocab_builder': vocab_builder,
        'preprocessor': preprocessor
    }


def create_data_loaders(data, args):
    """Create data loaders."""
    print("Creating data loaders...")
    
    # Create simple dataset class for tokenized data
    class TokenizedDataset(torch.utils.data.Dataset):
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
    
    # Create datasets
    train_dataset = TokenizedDataset(data['X_train'], data['y_train'])
    val_dataset = TokenizedDataset(data['X_val'], data['y_val'])
    test_dataset = TokenizedDataset(data['X_test'], data['y_test'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader


def create_model(vocab_size, args):
    """Create the NBoW model."""
    print("Creating NBoW model...")
    
    model = NBoW(
        vocab_size=vocab_size,
        embedding_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.num_classes,
        dropout_rate=args.dropout
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    return model


def train_model(model, train_loader, val_loader, args):
    """Train the model."""
    print("Starting training...")
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Initialize trainer
    trainer = Trainer(model, device=device)
    
    # Train model
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        save_best_model=True
    )
    
    return trainer, history


def evaluate_model(trainer, test_loader, data, args):
    """Evaluate the model."""
    print("Evaluating model...")
    
    # Evaluate on test set
    results = trainer.evaluate(test_loader)
    
    # Print results
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test Precision: {results['precision']:.4f}")
    print(f"Test Recall: {results['recall']:.4f}")
    print(f"Test F1-Score: {results['f1']:.4f}")
    
    # Classification report
    class_names = [f'Class {i}' for i in range(args.num_classes)]
    print("\nClassification Report:")
    print(classification_report(results['labels'], results['predictions'], 
                              target_names=class_names))
    
    return results


def visualize_results(history, results, data, args):
    """Create visualizations."""
    print("Creating visualizations...")
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix
    class_names = [f'Class {i}' for i in range(args.num_classes)]
    plot_confusion_matrix(results['confusion_matrix'], class_names)
    
    # Plot data distribution
    plot_data_distribution(data['y_train'], class_names)


def main():
    """Main function."""
    args = parse_args()
    
    print("NBoW Model Training")
    print("=" * 50)
    print(f"Data path: {args.data_path}")
    print(f"Model parameters: embed_dim={args.embed_dim}, hidden_dim={args.hidden_dim}")
    print(f"Training parameters: batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}")
    print("=" * 50)
    
    # Load and preprocess data
    data = load_and_preprocess_data(args)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(data, args)
    
    # Create model
    model = create_model(len(data['vocab']), args)
    
    # Train model
    trainer, history = train_model(model, train_loader, val_loader, args)
    
    # Evaluate model
    results = evaluate_model(trainer, test_loader, data, args)
    
    # Visualize results
    visualize_results(history, results, data, args)
    
    # Save model
    if args.save_model:
        print(f"Saving model to {args.model_path}")
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        
        model_config = {
            'vocab_size': len(data['vocab']),
            'embed_dim': args.embed_dim,
            'hidden_dim': args.hidden_dim,
            'num_classes': args.num_classes,
            'dropout': args.dropout,
            'max_length': args.max_length
        }
        
        trainer.save_model(args.model_path, vocab=data['vocab'], model_config=model_config)
        print("Model saved successfully!")
    
    print("Training completed!")


if __name__ == "__main__":
    main()
