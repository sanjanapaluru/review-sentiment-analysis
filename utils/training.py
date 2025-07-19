"""
Training utilities for customer review sentiment analysis models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import pickle
from pathlib import Path
import json


class Trainer:
    """
    Generic trainer for sentiment analysis models.
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Device to use for training
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        
    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module, 
                   optimizer: optim.Optimizer) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Average loss and accuracy
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            if isinstance(batch, dict):
                inputs = batch['text'].to(self.device)
                labels = batch['label'].to(self.device)
            else:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float, Dict[str, float]]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Average loss, accuracy, and detailed metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                if isinstance(batch, dict):
                    inputs = batch['text'].to(self.device)
                    labels = batch['label'].to(self.device)
                else:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Calculate detailed metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        }
        
        return avg_loss, accuracy, metrics
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, 
            epochs: int, lr: float = 1e-3, weight_decay: float = 1e-4,
            scheduler_step_size: int = 7, scheduler_gamma: float = 0.1,
            early_stopping_patience: int = 10, save_best_model: bool = True) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: Weight decay
            scheduler_step_size: Step size for learning rate scheduler
            scheduler_gamma: Gamma for learning rate scheduler
            early_stopping_patience: Early stopping patience
            save_best_model: Whether to save best model
            
        Returns:
            Training history
        """
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        patience_counter = 0
        
        print(f"Starting training on {self.device}...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validation
            val_loss, val_acc, val_metrics = self.validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_acc
                if save_best_model:
                    self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print epoch results
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s)')
            print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
            print(f'Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
            print(f'Val Metrics - Precision: {val_metrics["precision"]:.4f}, Recall: {val_metrics["recall"]:.4f}, F1: {val_metrics["f1"]:.4f}')
            print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            print('-' * 50)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if save_best_model and self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model with validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                if isinstance(batch, dict):
                    inputs = batch['text'].to(self.device)
                    labels = batch['label'].to(self.device)
                else:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def save_model(self, filepath: str, vocab: Dict[str, int] = None, 
                  model_config: Dict[str, Any] = None) -> None:
        """
        Save model and related data.
        
        Args:
            filepath: Path to save model
            vocab: Vocabulary dictionary
            model_config: Model configuration
        """
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'best_val_loss': self.best_val_loss,
                'best_val_accuracy': self.best_val_accuracy
            }
        }
        
        if vocab is not None:
            save_dict['vocab'] = vocab
        
        if model_config is not None:
            save_dict['model_config'] = model_config
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """
        Load model and related data.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Vocabulary and model configuration
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            self.train_losses = history.get('train_losses', [])
            self.val_losses = history.get('val_losses', [])
            self.train_accuracies = history.get('train_accuracies', [])
            self.val_accuracies = history.get('val_accuracies', [])
            self.best_val_loss = history.get('best_val_loss', float('inf'))
            self.best_val_accuracy = history.get('best_val_accuracy', 0.0)
        
        print(f"Model loaded from {filepath}")
        
        return checkpoint.get('vocab'), checkpoint.get('model_config')


class EarlyStopping:
    """
    Early stopping utility.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save
            
        Returns:
            Whether to stop training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


def calculate_class_weights(labels: List[int]) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: List of labels
        
    Returns:
        Class weights tensor
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    
    return torch.FloatTensor(class_weights)


def save_training_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save training configuration.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save configuration
    """
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training configuration saved to {filepath}")


def load_training_config(filepath: str) -> Dict[str, Any]:
    """
    Load training configuration.
    
    Args:
        filepath: Path to load configuration from
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config = json.load(f)
    print(f"Training configuration loaded from {filepath}")
    return config


if __name__ == "__main__":
    # Test training utilities
    print("Testing training utilities...")
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)
        
        def forward(self, x):
            return self.linear(x)
    
    # Test trainer initialization
    model = SimpleModel()
    trainer = Trainer(model)
    
    print(f"Trainer initialized with device: {trainer.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=5)
    print(f"Early stopping initialized with patience: {early_stopping.patience}")
    
    # Test class weights calculation
    labels = [0, 0, 0, 1, 1, 2]
    weights = calculate_class_weights(labels)
    print(f"Class weights: {weights}")
    
    print("Training utilities test completed!")
