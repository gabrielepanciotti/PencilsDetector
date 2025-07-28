"""
Visualization utilities for anomaly detection results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
import cv2
from matplotlib.colors import LinearSegmentedColormap
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_images_with_scores(images: List[np.ndarray], scores: List[float], 
                           labels: Optional[List[int]] = None,
                           threshold: Optional[float] = None,
                           n_cols: int = 5, figsize: Tuple[int, int] = (15, 10),
                           save_path: Optional[str] = None) -> None:
    """
    Plot images with their anomaly scores.
    
    Args:
        images: List of images to plot
        scores: List of anomaly scores
        labels: Optional list of ground truth labels (0 for normal, 1 for anomaly)
        threshold: Optional threshold for anomaly detection
        n_cols: Number of columns in the grid
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    for i in range(n_images):
        # Plot image
        axes[i].imshow(images[i])
        
        # Set title with score
        title = f"Score: {scores[i]:.4f}"
        
        # Add label if provided
        if labels is not None:
            title += f" | Label: {'Anomaly' if labels[i] == 1 else 'Normal'}"
        
        # Highlight if above threshold
        if threshold is not None and scores[i] >= threshold:
            title += " (Detected)"
            axes[i].spines['bottom'].set_color('red')
            axes[i].spines['top'].set_color('red')
            axes[i].spines['left'].set_color('red')
            axes[i].spines['right'].set_color('red')
            axes[i].spines['bottom'].set_linewidth(2)
            axes[i].spines['top'].set_linewidth(2)
            axes[i].spines['left'].set_linewidth(2)
            axes[i].spines['right'].set_linewidth(2)
            
        axes[i].set_title(title)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    plt.show()


def plot_anomaly_heatmap(image: np.ndarray, anomaly_map: np.ndarray, 
                        alpha: float = 0.5, cmap: str = 'jet',
                        figsize: Tuple[int, int] = (12, 6),
                        save_path: Optional[str] = None) -> None:
    """
    Plot an image with an anomaly heatmap overlay.
    
    Args:
        image: Original image
        anomaly_map: Anomaly heatmap (higher values indicate more anomalous regions)
        alpha: Transparency of the heatmap
        cmap: Colormap for the heatmap
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Normalize anomaly map to [0, 1]
    anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-10)
    
    # Resize anomaly map to match image size if needed
    if anomaly_map_norm.shape[:2] != image.shape[:2]:
        anomaly_map_norm = cv2.resize(anomaly_map_norm, (image.shape[1], image.shape[0]))
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Plot anomaly heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(anomaly_map_norm, cmap=cmap)
    plt.title("Anomaly Heatmap")
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Plot overlay
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(anomaly_map_norm, cmap=cmap, alpha=alpha)
    plt.title("Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    plt.show()


def plot_feature_space(features: np.ndarray, labels: np.ndarray, 
                      method: str = 'tsne', perplexity: int = 30,
                      figsize: Tuple[int, int] = (10, 8),
                      save_path: Optional[str] = None) -> None:
    """
    Plot feature space using dimensionality reduction.
    
    Args:
        features: Feature vectors (n_samples, n_features)
        labels: Labels (0 for normal, 1 for anomaly)
        method: Dimensionality reduction method ('tsne' or 'pca')
        perplexity: Perplexity parameter for t-SNE
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Apply dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embedding = reducer.fit_transform(features)
        title = f"t-SNE Visualization (perplexity={perplexity})"
    elif method == 'pca':
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(features)
        explained_var = reducer.explained_variance_ratio_
        title = f"PCA Visualization (explained variance: {explained_var[0]:.2f}, {explained_var[1]:.2f})"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot normal and anomalous samples
    normal_idx = labels == 0
    anomaly_idx = labels == 1
    
    plt.scatter(embedding[normal_idx, 0], embedding[normal_idx, 1], 
               c='blue', marker='o', label='Normal', alpha=0.7)
    plt.scatter(embedding[anomaly_idx, 0], embedding[anomaly_idx, 1], 
               c='red', marker='x', label='Anomaly', alpha=0.7)
    
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    plt.show()


def plot_training_history(history: Dict[str, List[float]], 
                         figsize: Tuple[int, int] = (10, 6),
                         save_path: Optional[str] = None) -> None:
    """
    Plot training history.
    
    Args:
        history: Dictionary of training metrics
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)
    
    for metric_name, values in history.items():
        plt.plot(values, label=metric_name)
    
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    plt.show()


def plot_reconstruction_comparison(original_images: List[np.ndarray], 
                                 reconstructed_images: List[np.ndarray],
                                 n_cols: int = 5, figsize: Tuple[int, int] = (15, 10),
                                 save_path: Optional[str] = None) -> None:
    """
    Plot original images and their reconstructions side by side.
    
    Args:
        original_images: List of original images
        reconstructed_images: List of reconstructed images
        n_cols: Number of columns in the grid
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    n_images = len(original_images)
    n_rows = 2 * ((n_images + n_cols - 1) // n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    for i in range(n_images):
        row = 2 * (i // n_cols)
        col = i % n_cols
        
        # Plot original image
        axes[row, col].imshow(original_images[i])
        axes[row, col].set_title(f"Original {i+1}")
        axes[row, col].axis('off')
        
        # Plot reconstructed image
        axes[row+1, col].imshow(reconstructed_images[i])
        axes[row+1, col].set_title(f"Reconstructed {i+1}")
        axes[row+1, col].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, (n_rows // 2) * n_cols):
        row = 2 * (i // n_cols)
        col = i % n_cols
        axes[row, col].axis('off')
        axes[row+1, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    plt.show()


def plot_reconstruction_error_distribution(errors: np.ndarray, labels: np.ndarray,
                                         threshold: Optional[float] = None,
                                         figsize: Tuple[int, int] = (10, 6),
                                         save_path: Optional[str] = None) -> None:
    """
    Plot distribution of reconstruction errors for normal and anomalous samples.
    
    Args:
        errors: Reconstruction errors
        labels: Labels (0 for normal, 1 for anomaly)
        threshold: Optional threshold for anomaly detection
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Split errors by label
    normal_errors = errors[labels == 0]
    anomaly_errors = errors[labels == 1]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot histograms
    plt.hist(normal_errors, bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(anomaly_errors, bins=50, alpha=0.5, label='Anomaly', density=True)
    
    # Plot threshold if provided
    if threshold is not None:
        plt.axvline(x=threshold, color='r', linestyle='--', 
                   label=f'Threshold: {threshold:.4f}')
    
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    plt.show()


def create_attention_heatmap(image: np.ndarray, attention_map: np.ndarray) -> np.ndarray:
    """
    Create an attention heatmap overlay on an image.
    
    Args:
        image: Original image
        attention_map: Attention map (higher values indicate more attention)
        
    Returns:
        Image with attention heatmap overlay
    """
    # Normalize attention map to [0, 1]
    attention_map_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-10)
    
    # Resize attention map to match image size if needed
    if attention_map_norm.shape[:2] != image.shape[:2]:
        attention_map_norm = cv2.resize(attention_map_norm, (image.shape[1], image.shape[0]))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map_norm), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
    
    return overlay


def plot_model_comparison(models: List[str], metrics: Dict[str, List[float]],
                         figsize: Tuple[int, int] = (12, 8),
                         save_path: Optional[str] = None) -> None:
    """
    Plot comparison of different models based on evaluation metrics.
    
    Args:
        models: List of model names
        metrics: Dictionary of metrics for each model
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Get metric names
    metric_names = list(metrics.keys())
    n_metrics = len(metric_names)
    n_models = len(models)
    
    # Create figure
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    # Bar width
    width = 0.8 / n_models
    
    # Plot each metric
    for i, metric_name in enumerate(metric_names):
        # Get values for this metric
        values = metrics[metric_name]
        
        # Plot bars
        for j, model_name in enumerate(models):
            x = j - 0.4 + (i + 0.5) * width
            axes[i].bar(x, values[j], width=width, label=model_name if i == 0 else "")
        
        # Set title and labels
        axes[i].set_title(metric_name)
        axes[i].set_xticks(range(n_models))
        axes[i].set_xticklabels(models, rotation=45)
        axes[i].set_ylim(0, 1.1 * max(values))
        
    # Add legend to the first subplot
    axes[0].legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    plt.show()


def visualize_batch(batch: Dict[str, torch.Tensor], n_samples: int = 5,
                   figsize: Tuple[int, int] = (12, 5),
                   save_path: Optional[str] = None) -> None:
    """
    Visualize a batch of images from a DataLoader.
    
    Args:
        batch: Batch from DataLoader
        n_samples: Number of samples to visualize
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Get images and labels
    images = batch['image'][:n_samples]
    labels = batch['label'][:n_samples] if 'label' in batch else None
    
    # Convert to numpy
    if isinstance(images, torch.Tensor):
        # Denormalize if needed
        if images.max() <= 1.0:
            images = images * 255.0
        
        # Move channels to the end
        if images.shape[1] in [1, 3]:  # CHW format
            images = images.permute(0, 2, 3, 1)
        
        images = images.cpu().numpy().astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(1, n_samples, figsize=figsize)
    
    # Plot each image
    for i in range(n_samples):
        axes[i].imshow(images[i])
        
        # Add label if available
        if labels is not None:
            label = labels[i].item() if isinstance(labels, torch.Tensor) else labels[i]
            axes[i].set_title(f"Label: {'Anomaly' if label == 1 else 'Normal'}")
            
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    plt.show()
