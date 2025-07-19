"""
Data loading utilities for customer review sentiment analysis.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.model_selection import train_test_split


class ReviewDataset(Dataset):
    """
    Custom dataset for customer review sentiment analysis.
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer=None, max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            texts: List of review texts
            labels: List of sentiment labels
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.tokenizer:
            # Use transformer tokenizer
            encoded = self.tokenizer.tokenize([text])
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            # For traditional models, return text as is
            return {
                'text': text,
                'label': torch.tensor(label, dtype=torch.long)
            }


class CustomDataLoader:
    """
    Data loader for customer review sentiment analysis.
    """
    
    def __init__(self, data_path: str = None, df: pd.DataFrame = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to CSV file
            df: DataFrame with review data
        """
        if data_path:
            self.df = pd.read_csv(data_path)
        elif df is not None:
            self.df = df
        else:
            raise ValueError("Either data_path or df must be provided")
        
        self.validate_data()
    
    def validate_data(self):
        """Validate the data format."""
        required_columns = ['text', 'label']
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Check for missing values
        if self.df['text'].isnull().any():
            print("Warning: Found missing text values, removing them...")
            self.df = self.df.dropna(subset=['text'])
        
        # Ensure labels are binary (0 or 1)
        unique_labels = self.df['label'].unique()
        if not all(label in [0, 1] for label in unique_labels):
            print("Warning: Labels are not binary (0, 1), converting...")
            self.df['label'] = (self.df['label'] > 0).astype(int)
    
    def get_basic_stats(self) -> Dict:
        """Get basic statistics about the data."""
        stats = {
            'total_reviews': len(self.df),
            'positive_reviews': (self.df['label'] == 1).sum(),
            'negative_reviews': (self.df['label'] == 0).sum(),
            'avg_text_length': self.df['text'].str.len().mean(),
            'max_text_length': self.df['text'].str.len().max(),
            'min_text_length': self.df['text'].str.len().min()
        }
        return stats
    
    def split_data(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            test_size: Proportion of test data
            val_size: Proportion of validation data
            random_state: Random seed
            
        Returns:
            train_df, val_df, test_df
        """
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            self.df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.df['label']
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_df['label']
        )
        
        return train_df, val_df, test_df
    
    def create_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                       tokenizer=None, max_length: int = 512) -> Tuple[ReviewDataset, ReviewDataset, ReviewDataset]:
        """
        Create PyTorch datasets from DataFrames.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        train_dataset = ReviewDataset(
            train_df['text'].tolist(),
            train_df['label'].tolist(),
            tokenizer,
            max_length
        )
        
        val_dataset = ReviewDataset(
            val_df['text'].tolist(),
            val_df['label'].tolist(),
            tokenizer,
            max_length
        )
        
        test_dataset = ReviewDataset(
            test_df['text'].tolist(),
            test_df['label'].tolist(),
            tokenizer,
            max_length
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset: ReviewDataset, val_dataset: ReviewDataset, 
                          test_dataset: ReviewDataset, batch_size: int = 32, 
                          num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            train_loader, val_loader, test_loader
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader


def load_sample_data() -> pd.DataFrame:
    """
    Load sample customer review data.
    
    Returns:
        DataFrame with sample reviews
    """
    import os
    
    # Try to load from data directory
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_reviews.csv')
    
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        # Create sample data if file doesn't exist
        sample_data = {
            'text': [
                "This product is absolutely amazing! Great quality and fast delivery.",
                "Terrible experience. Product arrived damaged and customer service was unhelpful.",
                "Good value for money. Not perfect but does the job well.",
                "Worst purchase I've ever made. Complete waste of money.",
                "Excellent service and high quality product. Very satisfied."
            ],
            'label': [1, 0, 1, 0, 1]
        }
        return pd.DataFrame(sample_data)


if __name__ == "__main__":
    # Test the data loader
    print("Testing CustomDataLoader...")
    
    # Load sample data
    df = load_sample_data()
    loader = CustomDataLoader(df=df)
    
    # Get basic stats
    stats = loader.get_basic_stats()
    print("Data statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Split data
    train_df, val_df, test_df = loader.split_data()
    print(f"\nData splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = loader.create_datasets(train_df, val_df, test_df)
    print(f"\nDataset sizes:")
    print(f"  Train dataset: {len(train_dataset)} samples")
    print(f"  Validation dataset: {len(val_dataset)} samples")
    print(f"  Test dataset: {len(test_dataset)} samples")
    
    # Test dataset item
    sample = train_dataset[0]
    print(f"\nSample item keys: {sample.keys()}")
    print(f"Sample label: {sample['label']}")
    print(f"Sample text: {sample['text'][:50]}...")
