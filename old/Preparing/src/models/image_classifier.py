"""
Image classification models for anomaly detection in computer vision.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import time
from tqdm import tqdm


class BaseClassifier:
    """Base class for image classifiers."""
    
    def __init__(self, num_classes: int, device: str = None):
        """
        Initialize the classifier.
        
        Args:
            num_classes: Number of classes
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.num_classes = num_classes
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
    def build_model(self):
        """Build the model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement build_model()")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None, 
           epochs: int = 10, patience: int = 5, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            patience: Early stopping patience
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            if val_loader is not None:
                val_loss, val_acc = self.validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                          f"Time: {time.time() - start_time:.2f}s")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Time: {time.time() - start_time:.2f}s")
            
            # Step scheduler if it exists
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Load best model if validation was used
        if val_loader is not None and hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            
        return self.history
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Predicted class indices
        """
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu()
    
    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Class probabilities
        """
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            probas = F.softmax(outputs, dim=1)
        return probas.cpu()
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                _, predictions = torch.max(outputs, 1)
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
        
        # Calculate metrics
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        
        metrics = {
            'accuracy': accuracy_score(all_targets, all_predictions),
            'precision': precision_score(all_targets, all_predictions, average='weighted'),
            'recall': recall_score(all_targets, all_predictions, average='weighted'),
            'f1_score': f1_score(all_targets, all_predictions, average='weighted')
        }
        
        return metrics
    
    def plot_confusion_matrix(self, test_loader: DataLoader, class_names: List[str] = None,
                             figsize: Tuple[int, int] = (10, 8)) -> np.ndarray:
        """
        Plot confusion matrix.
        
        Args:
            test_loader: Test data loader
            class_names: List of class names
            figsize: Figure size
            
        Returns:
            Confusion matrix
        """
        self.model.eval()
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                _, predictions = torch.max(outputs, 1)
                
                all_targets.extend(targets.numpy())
                all_predictions.extend(predictions.cpu().numpy())
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Plot
        plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        if class_names is None:
            class_names = [str(i) for i in range(self.num_classes)]
            
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
        return cm
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 5)) -> None:
        """
        Plot training history.
        
        Args:
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        if 'val_loss' in self.history and self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        if 'val_acc' in self.history and self.history['val_acc']:
            plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'num_classes': self.num_classes,
            'history': self.history
        }
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the model.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Ensure model is built
        if self.model is None:
            self.num_classes = checkpoint['num_classes']
            self.build_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            
        print(f"Model loaded from {path}")


class ResNetClassifier(BaseClassifier):
    """ResNet-based image classifier."""
    
    def __init__(self, num_classes: int, pretrained: bool = True, 
                learning_rate: float = 0.001, weight_decay: float = 0.0001,
                resnet_version: int = 18, device: str = None):
        """
        Initialize ResNet classifier.
        
        Args:
            num_classes: Number of classes
            pretrained: Whether to use pretrained weights
            learning_rate: Learning rate
            weight_decay: Weight decay for L2 regularization
            resnet_version: ResNet version (18, 34, 50, 101, 152)
            device: Device to use
        """
        super().__init__(num_classes, device)
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.resnet_version = resnet_version
        
        self.build_model()
    
    def build_model(self) -> None:
        """Build ResNet model."""
        # Select ResNet version
        if self.resnet_version == 18:
            base_model = models.resnet18(pretrained=self.pretrained)
        elif self.resnet_version == 34:
            base_model = models.resnet34(pretrained=self.pretrained)
        elif self.resnet_version == 50:
            base_model = models.resnet50(pretrained=self.pretrained)
        elif self.resnet_version == 101:
            base_model = models.resnet101(pretrained=self.pretrained)
        elif self.resnet_version == 152:
            base_model = models.resnet152(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported ResNet version: {self.resnet_version}")
        
        # Replace final fully connected layer
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, self.num_classes)
        
        self.model = base_model.to(self.device)
        
        # Define loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )


class EfficientNetClassifier(BaseClassifier):
    """EfficientNet-based image classifier."""
    
    def __init__(self, num_classes: int, pretrained: bool = True, 
                learning_rate: float = 0.001, weight_decay: float = 0.0001,
                version: str = 'b0', device: str = None):
        """
        Initialize EfficientNet classifier.
        
        Args:
            num_classes: Number of classes
            pretrained: Whether to use pretrained weights
            learning_rate: Learning rate
            weight_decay: Weight decay for L2 regularization
            version: EfficientNet version ('b0' to 'b7')
            device: Device to use
        """
        super().__init__(num_classes, device)
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.version = version
        
        self.build_model()
    
    def build_model(self) -> None:
        """Build EfficientNet model."""
        # Select EfficientNet version
        model_name = f"efficientnet_{self.version}"
        if not hasattr(models, model_name):
            raise ValueError(f"Unsupported EfficientNet version: {self.version}")
            
        base_model = getattr(models, model_name)(pretrained=self.pretrained)
        
        # Replace classifier
        if hasattr(base_model, 'classifier'):
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Linear(in_features, self.num_classes)
        else:
            # For older torchvision versions
            in_features = base_model.fc.in_features
            base_model.fc = nn.Linear(in_features, self.num_classes)
        
        self.model = base_model.to(self.device)
        
        # Define loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )


class CustomCNN(BaseClassifier):
    """Custom CNN classifier for smaller datasets or specific tasks."""
    
    def __init__(self, num_classes: int, input_shape: Tuple[int, int, int] = (3, 224, 224),
                learning_rate: float = 0.001, weight_decay: float = 0.0001,
                dropout_rate: float = 0.5, device: str = None):
        """
        Initialize custom CNN classifier.
        
        Args:
            num_classes: Number of classes
            input_shape: Input shape (channels, height, width)
            learning_rate: Learning rate
            weight_decay: Weight decay for L2 regularization
            dropout_rate: Dropout rate
            device: Device to use
        """
        super().__init__(num_classes, device)
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        
        self.build_model()
    
    def build_model(self) -> None:
        """Build custom CNN model."""
        channels, height, width = self.input_shape
        
        # Simple CNN architecture
        self.model = nn.Sequential(
            # First convolutional block
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Flatten
            nn.Flatten(),
            
            # Fully connected layers
            nn.Linear(256 * (height // 16) * (width // 16), 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, self.num_classes)
        ).to(self.device)
        
        # Define loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )


def create_data_augmentation(imbalanced_classes: bool = False, 
                           class_weights: Optional[Dict[int, float]] = None) -> transforms.Compose:
    """
    Create data augmentation transforms.
    
    Args:
        imbalanced_classes: Whether the dataset has imbalanced classes
        class_weights: Optional weights for each class
        
    Returns:
        Transforms composition
    """
    # Basic augmentation
    basic_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # Additional augmentation for imbalanced classes
    if imbalanced_classes:
        # Add more aggressive augmentation
        advanced_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        return transforms.Compose(advanced_transforms)
    else:
        return transforms.Compose(basic_transforms)


def get_class_weights(dataset: Any) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        dataset: Dataset with targets attribute
        
    Returns:
        Tensor of class weights
    """
    # Get all targets
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'labels'):
        targets = dataset.labels
    else:
        # Try to extract targets from dataset
        targets = []
        for _, target in dataset:
            targets.append(target)
    
    # Count class occurrences
    class_counts = np.bincount(targets)
    
    # Calculate weights (inverse frequency)
    n_samples = len(targets)
    n_classes = len(class_counts)
    weights = n_samples / (n_classes * class_counts)
    
    # Normalize weights
    weights = weights / weights.sum() * n_classes
    
    return torch.FloatTensor(weights)
