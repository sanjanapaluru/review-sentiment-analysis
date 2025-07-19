"""
Main inference script for customer review sentiment analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import argparse
import json
from pathlib import Path

# Import our modules
from models.nbow import NBoW
from models.lstm import LSTMModel
from models.cnn import CNNModel
from models.transformer import TransformerModel
from utils.preprocessing import TextPreprocessor, VocabularyBuilder
from utils.training import Trainer


class SentimentAnalyzer:
    """
    Sentiment analyzer for customer reviews.
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_path: Path to the trained model
            device: Device to use for inference
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vocab = None
        self.vocab_builder = None
        self.preprocessor = None
        self.model_config = None
        self.class_names = ["Negative", "Positive"]  # Default class names
        
        self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load the trained model.
        
        Args:
            model_path: Path to the model file
        """
        print(f"Loading model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract information
        self.vocab = checkpoint['vocab']
        self.model_config = checkpoint['model_config']
        model_class = checkpoint['model_class']
        
        # Initialize vocabulary builder
        self.vocab_builder = VocabularyBuilder()
        self.vocab_builder.word_to_idx = self.vocab
        self.vocab_builder.idx_to_word = {idx: word for word, idx in self.vocab.items()}
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor()
        
        # Create model based on class
        if model_class == 'NBoW':
            self.model = NBoW(
                vocab_size=self.model_config['vocab_size'],
                embedding_dim=self.model_config['embed_dim'],
                hidden_dim=self.model_config['hidden_dim'],
                output_dim=self.model_config['num_classes'],
                dropout_rate=self.model_config['dropout']
            )
        elif model_class == 'LSTMModel':
            self.model = LSTMModel(
                vocab_size=self.model_config['vocab_size'],
                embed_dim=self.model_config['embed_dim'],
                hidden_dim=self.model_config['hidden_dim'],
                num_classes=self.model_config['num_classes'],
                num_layers=self.model_config.get('num_layers', 2),
                dropout=self.model_config['dropout'],
                bidirectional=self.model_config.get('bidirectional', True)
            )
        elif model_class == 'CNNModel':
            self.model = CNNModel(
                vocab_size=self.model_config['vocab_size'],
                embed_dim=self.model_config['embed_dim'],
                num_classes=self.model_config['num_classes'],
                filter_sizes=self.model_config.get('filter_sizes', [3, 4, 5]),
                num_filters=self.model_config.get('num_filters', 100),
                dropout=self.model_config['dropout']
            )
        elif model_class == 'TransformerModel':
            self.model = TransformerModel(
                num_classes=self.model_config['num_classes'],
                model_name=self.model_config.get('model_name', 'bert-base-uncased'),
                dropout=self.model_config['dropout']
            )
        else:
            raise ValueError(f"Unknown model class: {model_class}")
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully! ({model_class})")
    
    def preprocess_text(self, text: str) -> torch.Tensor:
        """
        Preprocess text for inference.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed tensor
        """
        # Preprocess text
        tokens = self.preprocessor.preprocess_text(text)
        
        # Convert to indices
        max_length = self.model_config.get('max_length', 100)
        indices = self.vocab_builder.text_to_indices(tokens, max_length)
        
        # Convert to tensor
        tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device)
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Prediction results
        """
        # Preprocess text
        input_tensor = self.preprocess_text(text)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get class probabilities
        class_probs = {}
        for i, class_name in enumerate(self.class_names):
            class_probs[class_name] = probabilities[0][i].item()
        
        return {
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': class_probs
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction results
        """
        results = []
        
        for text in texts:
            result = self.predict(text)
            results.append(result)
        
        return results
    
    def analyze_review(self, review: str, detailed: bool = False) -> Dict:
        """
        Analyze a customer review with detailed information.
        
        Args:
            review: Customer review text
            detailed: Whether to include detailed analysis
            
        Returns:
            Analysis results
        """
        # Basic prediction
        prediction = self.predict(review)
        
        result = {
            'review': review,
            'sentiment': prediction['predicted_class'],
            'confidence': prediction['confidence'],
            'probabilities': prediction['probabilities']
        }
        
        if detailed:
            # Add detailed analysis
            tokens = self.preprocessor.preprocess_text(review)
            
            result.update({
                'review_length': len(review),
                'word_count': len(tokens),
                'processed_tokens': tokens[:10],  # First 10 tokens
                'confidence_category': self._get_confidence_category(prediction['confidence'])
            })
        
        return result
    
    def _get_confidence_category(self, confidence: float) -> str:
        """Get confidence category."""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.5:
            return "Medium"
        else:
            return "Low"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sentiment Analysis Inference')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to analyze')
    parser.add_argument('--input_file', type=str, default=None,
                       help='Input file with texts to analyze')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for results')
    parser.add_argument('--text_column', type=str, default='text',
                       help='Column name for text in input file')
    parser.add_argument('--detailed', action='store_true',
                       help='Include detailed analysis')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("Sentiment Analysis Inference")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(args.model_path, device=args.device)
    
    if args.text:
        # Analyze single text
        print(f"Analyzing text: {args.text}")
        result = analyzer.analyze_review(args.text, detailed=args.detailed)
        
        print("\nResults:")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: {result['probabilities']}")
        
        if args.detailed:
            print(f"Review length: {result['review_length']}")
            print(f"Word count: {result['word_count']}")
            print(f"Confidence category: {result['confidence_category']}")
    
    elif args.input_file:
        # Analyze file
        print(f"Analyzing file: {args.input_file}")
        
        # Load data
        if args.input_file.endswith('.csv'):
            df = pd.read_csv(args.input_file)
        elif args.input_file.endswith('.json'):
            df = pd.read_json(args.input_file)
        else:
            raise ValueError("Unsupported file format")
        
        # Analyze texts
        texts = df[args.text_column].astype(str).tolist()
        results = []
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"Processing {i}/{len(texts)}...")
            
            result = analyzer.analyze_review(text, detailed=args.detailed)
            results.append(result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        if args.output_file:
            if args.output_file.endswith('.csv'):
                results_df.to_csv(args.output_file, index=False)
            elif args.output_file.endswith('.json'):
                results_df.to_json(args.output_file, orient='records', indent=2)
            else:
                raise ValueError("Unsupported output format")
            
            print(f"Results saved to {args.output_file}")
        else:
            print("\nSample results:")
            print(results_df.head())
            
            # Summary statistics
            print("\nSummary:")
            print(results_df['sentiment'].value_counts())
            print(f"Average confidence: {results_df['confidence'].mean():.4f}")
    
    else:
        # Interactive mode
        print("Interactive mode - Enter reviews to analyze (type 'quit' to exit)")
        
        while True:
            review = input("\nEnter review: ")
            if review.lower() == 'quit':
                break
            
            result = analyzer.analyze_review(review, detailed=True)
            
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.4f} ({result['confidence_category']})")
            print(f"Probabilities: {result['probabilities']}")
    
    print("Analysis completed!")


if __name__ == "__main__":
    main()
