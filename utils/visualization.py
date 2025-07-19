"""
Visualization utilities for customer review sentiment analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import itertools
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter


# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(history['train_losses']) + 1)
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         normalize: bool = False, save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save plot
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()


def plot_classification_report(y_true: List[int], y_pred: List[int], 
                             class_names: List[str], save_path: Optional[str] = None) -> None:
    """
    Plot classification report as heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
    """
    from sklearn.metrics import classification_report
    
    # Get classification report as dict
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                 output_dict=True, zero_division=0)
    
    # Extract metrics for each class
    metrics = ['precision', 'recall', 'f1-score']
    data = []
    
    for class_name in class_names:
        if class_name in report:
            data.append([report[class_name][metric] for metric in metrics])
    
    # Create DataFrame
    df = pd.DataFrame(data, index=class_names, columns=metrics)
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'Score'})
    plt.title('Classification Report', fontsize=14, fontweight='bold')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Classification report plot saved to {save_path}")
    
    plt.show()


def plot_data_distribution(labels: List[int], class_names: List[str], 
                          save_path: Optional[str] = None) -> None:
    """
    Plot data distribution by class.
    
    Args:
        labels: List of labels
        class_names: List of class names
        save_path: Path to save plot
    """
    # Count labels
    label_counts = Counter(labels)
    
    # Create data
    counts = [label_counts[i] for i in range(len(class_names))]
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, counts, color=sns.color_palette("husl", len(class_names)))
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts),
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Data Distribution by Class', fontsize=14, fontweight='bold')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    total = sum(counts)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        percentage = (count / total) * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{percentage:.1f}%', ha='center', va='center', 
                fontweight='bold', color='white')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Data distribution plot saved to {save_path}")
    
    plt.show()


def plot_word_cloud(texts: List[str], title: str = "Word Cloud", 
                   save_path: Optional[str] = None) -> None:
    """
    Generate and plot word cloud.
    
    Args:
        texts: List of texts
        title: Title for the plot
        save_path: Path to save plot
    """
    # Combine all texts
    combined_text = ' '.join(texts)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap='viridis',
                         max_words=100).generate(combined_text)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Word cloud saved to {save_path}")
    
    plt.show()


def plot_text_length_distribution(texts: List[str], save_path: Optional[str] = None) -> None:
    """
    Plot text length distribution.
    
    Args:
        texts: List of texts
        save_path: Path to save plot
    """
    # Calculate text lengths
    lengths = [len(text.split()) for text in texts]
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(lengths), color='red', linestyle='--', 
                label=f'Mean: {np.mean(lengths):.1f}')
    plt.axvline(np.median(lengths), color='green', linestyle='--', 
                label=f'Median: {np.median(lengths):.1f}')
    
    plt.title('Text Length Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Text length distribution plot saved to {save_path}")
    
    plt.show()


def plot_model_comparison(results: Dict[str, Dict[str, float]], 
                         save_path: Optional[str] = None) -> None:
    """
    Plot model comparison results.
    
    Args:
        results: Dictionary of model results
        save_path: Path to save plot
    """
    # Extract data
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Create data matrix
    data = []
    for model in models:
        data.append([results[model].get(metric, 0) for metric in metrics])
    
    # Create DataFrame
    df = pd.DataFrame(data, index=models, columns=metrics)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind='bar', ax=ax, width=0.8)
    
    plt.title('Model Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.legend(title='Metrics')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, model in enumerate(models):
        for j, metric in enumerate(metrics):
            value = results[model].get(metric, 0)
            plt.text(i + (j-1.5)*0.2, value + 0.01, f'{value:.3f}', 
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    plt.show()


def plot_learning_curves(train_sizes: List[int], train_scores: List[float], 
                        val_scores: List[float], save_path: Optional[str] = None) -> None:
    """
    Plot learning curves.
    
    Args:
        train_sizes: Training set sizes
        train_scores: Training scores
        val_scores: Validation scores
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_sizes, train_scores, 'b-', label='Training Score', linewidth=2)
    plt.plot(train_sizes, val_scores, 'r-', label='Validation Score', linewidth=2)
    
    plt.title('Learning Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves plot saved to {save_path}")
    
    plt.show()


def plot_attention_weights(attention_weights: np.ndarray, tokens: List[str], 
                          save_path: Optional[str] = None) -> None:
    """
    Plot attention weights heatmap.
    
    Args:
        attention_weights: Attention weights matrix
        tokens: List of tokens
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens,
                cmap='Blues', cbar_kws={'label': 'Attention Weight'})
    
    plt.title('Attention Weights', fontsize=14, fontweight='bold')
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention weights plot saved to {save_path}")
    
    plt.show()


def create_interactive_dashboard(results: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Create interactive dashboard with plotly.
    
    Args:
        results: Results dictionary
        save_path: Path to save HTML file
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training History', 'Confusion Matrix', 
                       'Class Distribution', 'Model Comparison'),
        specs=[[{'secondary_y': True}, {'type': 'heatmap'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Training history
    if 'history' in results:
        history = results['history']
        epochs = list(range(1, len(history['train_losses']) + 1))
        
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_losses'], 
                      name='Train Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_losses'], 
                      name='Val Loss', line=dict(color='red')),
            row=1, col=1
        )
    
    # Confusion matrix
    if 'confusion_matrix' in results:
        cm = results['confusion_matrix']
        class_names = results.get('class_names', ['Negative', 'Positive'])
        
        fig.add_trace(
            go.Heatmap(z=cm, x=class_names, y=class_names,
                      colorscale='Blues', showscale=False),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='Sentiment Analysis Dashboard',
        showlegend=True,
        height=800
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")
    
    fig.show()


def save_all_plots(results: Dict[str, Any], output_dir: str = "plots") -> None:
    """
    Save all plots to directory.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Training history
    if 'history' in results:
        plot_training_history(results['history'], 
                            save_path=os.path.join(output_dir, 'training_history.png'))
    
    # Confusion matrix
    if 'confusion_matrix' in results and 'class_names' in results:
        plot_confusion_matrix(results['confusion_matrix'], results['class_names'],
                            save_path=os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Classification report
    if 'y_true' in results and 'y_pred' in results:
        plot_classification_report(results['y_true'], results['y_pred'], 
                                  results['class_names'],
                                  save_path=os.path.join(output_dir, 'classification_report.png'))
    
    # Data distribution
    if 'labels' in results and 'class_names' in results:
        plot_data_distribution(results['labels'], results['class_names'],
                             save_path=os.path.join(output_dir, 'data_distribution.png'))
    
    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")
    
    # Test training history plot
    history = {
        'train_losses': [0.8, 0.6, 0.4, 0.3, 0.2],
        'val_losses': [0.7, 0.5, 0.4, 0.35, 0.3],
        'train_accuracies': [0.6, 0.7, 0.8, 0.85, 0.9],
        'val_accuracies': [0.65, 0.72, 0.78, 0.82, 0.85]
    }
    
    # Test confusion matrix
    cm = np.array([[85, 15], [20, 80]])
    class_names = ['Negative', 'Positive']
    
    print("Plotting training history...")
    plot_training_history(history)
    
    print("Plotting confusion matrix...")
    plot_confusion_matrix(cm, class_names)
    
    print("Plotting data distribution...")
    labels = [0] * 100 + [1] * 80
    plot_data_distribution(labels, class_names)
    
    print("Visualization utilities test completed!")
