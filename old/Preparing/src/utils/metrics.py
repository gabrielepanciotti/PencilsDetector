"""
Evaluation metrics for anomaly detection models.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix
)
import matplotlib.pyplot as plt


def calculate_metrics(labels: np.ndarray, scores: np.ndarray, threshold: Optional[float] = None) -> Dict[str, float]:
    """
    Calculate evaluation metrics for anomaly detection.
    
    Args:
        labels: Ground truth labels (0 for normal, 1 for anomaly)
        scores: Anomaly scores (higher means more anomalous)
        threshold: Optional threshold for binary predictions
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # ROC AUC
    if len(np.unique(labels)) > 1:  # Only calculate if both classes are present
        metrics['roc_auc'] = roc_auc_score(labels, scores)
    
    # Average Precision (PR AUC)
    metrics['pr_auc'] = average_precision_score(labels, scores)
    
    # If threshold is not provided, find the optimal one
    if threshold is None:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        # Find threshold that maximizes F1 score
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    
    # Binary predictions
    predictions = (scores >= threshold).astype(int)
    
    # Classification metrics
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['precision'] = precision_score(labels, predictions, zero_division=0)
    metrics['recall'] = recall_score(labels, predictions, zero_division=0)
    metrics['f1_score'] = f1_score(labels, predictions, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics['true_negative'] = int(tn)
    metrics['false_positive'] = int(fp)
    metrics['false_negative'] = int(fn)
    metrics['true_positive'] = int(tp)
    
    # Additional metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    
    # Threshold
    metrics['threshold'] = threshold
    
    return metrics


def plot_roc_curve(labels: np.ndarray, scores: np.ndarray, 
                  save_path: Optional[str] = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Plot ROC curve and calculate AUC.
    
    Args:
        labels: Ground truth labels (0 for normal, 1 for anomaly)
        scores: Anomaly scores (higher means more anomalous)
        save_path: Optional path to save the plot
        
    Returns:
        Tuple of (AUC, FPR, TPR)
    """
    from sklearn.metrics import roc_curve, auc
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
    
    return roc_auc, fpr, tpr


def plot_precision_recall_curve(labels: np.ndarray, scores: np.ndarray, 
                               save_path: Optional[str] = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Plot Precision-Recall curve and calculate AUC.
    
    Args:
        labels: Ground truth labels (0 for normal, 1 for anomaly)
        scores: Anomaly scores (higher means more anomalous)
        save_path: Optional path to save the plot
        
    Returns:
        Tuple of (AUC, precision, recall)
    """
    # Calculate Precision-Recall curve and AUC
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'Precision-Recall curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
    
    return pr_auc, precision, recall


def plot_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, 
                         class_names: List[str] = ['Normal', 'Anomaly'],
                         save_path: Optional[str] = None) -> np.ndarray:
    """
    Plot confusion matrix.
    
    Args:
        labels: Ground truth labels (0 for normal, 1 for anomaly)
        predictions: Predicted labels (0 for normal, 1 for anomaly)
        class_names: Names of the classes
        save_path: Optional path to save the plot
        
    Returns:
        Confusion matrix
    """
    import seaborn as sns
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
    
    return cm


def plot_anomaly_score_distribution(scores: np.ndarray, labels: np.ndarray, 
                                   threshold: Optional[float] = None,
                                   save_path: Optional[str] = None) -> None:
    """
    Plot distribution of anomaly scores for normal and anomalous samples.
    
    Args:
        scores: Anomaly scores (higher means more anomalous)
        labels: Ground truth labels (0 for normal, 1 for anomaly)
        threshold: Optional threshold for binary predictions
        save_path: Optional path to save the plot
    """
    # Split scores by label
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(normal_scores, bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(anomaly_scores, bins=50, alpha=0.5, label='Anomaly', density=True)
    
    # Plot threshold if provided
    if threshold is not None:
        plt.axvline(x=threshold, color='r', linestyle='--', 
                   label=f'Threshold: {threshold:.3f}')
    
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Distribution of Anomaly Scores')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()


def find_optimal_threshold(labels: np.ndarray, scores: np.ndarray, 
                          criterion: str = 'f1') -> float:
    """
    Find the optimal threshold for binary predictions.
    
    Args:
        labels: Ground truth labels (0 for normal, 1 for anomaly)
        scores: Anomaly scores (higher means more anomalous)
        criterion: Criterion for optimization ('f1', 'accuracy', 'precision', 'recall')
        
    Returns:
        Optimal threshold
    """
    # Get precision, recall, thresholds from PR curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    
    # Ensure thresholds has the same length as precision and recall
    if len(thresholds) < len(precision):
        thresholds = np.append(thresholds, thresholds[-1])
    
    if criterion == 'f1':
        # F1 score
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx]
    
    elif criterion == 'accuracy':
        # Try different thresholds and calculate accuracy
        best_threshold = 0
        best_accuracy = 0
        
        for threshold in thresholds:
            predictions = (scores >= threshold).astype(int)
            accuracy = accuracy_score(labels, predictions)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
                
        return best_threshold
    
    elif criterion == 'precision':
        # Find threshold that gives at least 0.9 precision
        target_precision = 0.9
        for i, p in enumerate(precision):
            if p >= target_precision:
                return thresholds[i]
        return thresholds[-1]
    
    elif criterion == 'recall':
        # Find threshold that gives at least 0.9 recall
        target_recall = 0.9
        for i, r in enumerate(recall):
            if r >= target_recall:
                return thresholds[i]
        return thresholds[0]
    
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def evaluate_threshold_sensitivity(labels: np.ndarray, scores: np.ndarray, 
                                 thresholds: Optional[np.ndarray] = None,
                                 save_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Evaluate sensitivity of metrics to threshold value.
    
    Args:
        labels: Ground truth labels (0 for normal, 1 for anomaly)
        scores: Anomaly scores (higher means more anomalous)
        thresholds: Optional array of thresholds to evaluate
        save_path: Optional path to save the plot
        
    Returns:
        Dictionary of metrics for each threshold
    """
    # If thresholds not provided, create a range
    if thresholds is None:
        min_score, max_score = scores.min(), scores.max()
        thresholds = np.linspace(min_score, max_score, 100)
    
    # Initialize metrics
    results = {
        'threshold': thresholds,
        'accuracy': np.zeros_like(thresholds),
        'precision': np.zeros_like(thresholds),
        'recall': np.zeros_like(thresholds),
        'f1_score': np.zeros_like(thresholds),
        'specificity': np.zeros_like(thresholds)
    }
    
    # Calculate metrics for each threshold
    for i, threshold in enumerate(thresholds):
        predictions = (scores >= threshold).astype(int)
        
        # Classification metrics
        results['accuracy'][i] = accuracy_score(labels, predictions)
        results['precision'][i] = precision_score(labels, predictions, zero_division=0)
        results['recall'][i] = recall_score(labels, predictions, zero_division=0)
        results['f1_score'][i] = f1_score(labels, predictions, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        results['specificity'][i] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']:
        plt.plot(thresholds, results[metric], label=metric)
    
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Sensitivity of Metrics to Threshold Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
    
    return results
