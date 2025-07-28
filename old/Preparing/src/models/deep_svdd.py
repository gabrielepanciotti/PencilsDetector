"""
Deep SVDD (Support Vector Data Description) implementation for anomaly detection.

This module implements Deep SVDD, a deep learning-based anomaly detection method
that maps inputs to a hypersphere and minimizes the volume of this hypersphere.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Union, Optional, Any
from tqdm import tqdm

# Import local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.anomaly_detector import BaseAnomalyDetector


class DeepSVDDNetwork(nn.Module):
    """
    Network architecture for Deep SVDD.
    """
    def __init__(self, input_shape: Tuple[int, int, int], output_dim: int = 128):
        """
        Initialize the network.
        
        Args:
            input_shape: Shape of input images (channels, height, width)
            output_dim: Dimension of the output representation
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.output_dim = output_dim
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            # First block
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        
        # Calculate size of flattened features
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.feature_extractor(dummy_input)
            self.feature_size = dummy_output.numel()
        
        # Output layer
        self.fc = nn.Linear(self.feature_size, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output representation
        """
        features = self.feature_extractor(x)
        features = torch.flatten(features, start_dim=1)
        return self.fc(features)


class DeepSVDD(BaseAnomalyDetector):
    """
    Deep Support Vector Data Description (Deep SVDD) for anomaly detection.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        output_dim: int = 128,
        learning_rate: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 32,
        weight_decay: float = 1e-6,
        objective: str = 'one-class',
        nu: float = 0.1,
        name: str = "DeepSVDD"
    ):
        """
        Initialize Deep SVDD.
        
        Args:
            input_shape: Shape of input images (channels, height, width)
            output_dim: Dimension of the output representation
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            batch_size: Batch size for training
            weight_decay: Weight decay for regularization
            objective: 'one-class' or 'soft-boundary'
            nu: Regularization hyperparameter for soft-boundary objective
            name: Name of the model
        """
        super().__init__(name=name)
        
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        # Validate objective
        if objective not in ['one-class', 'soft-boundary']:
            raise ValueError(f"Objective must be 'one-class' or 'soft-boundary', got {objective}")
        self.objective = objective
        self.nu = nu
        
        # Initialize network
        self.net = DeepSVDDNetwork(input_shape, output_dim)
        self.net.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Initialize center and radius
        self.center = None
        self.radius = 0.0
        
    def init_center(self, train_loader: DataLoader, eps: float = 0.1) -> None:
        """
        Initialize the center as the mean of the network outputs.
        
        Args:
            train_loader: DataLoader containing training data
            eps: Small constant to avoid numerical issues
        """
        self.net.eval()
        n_samples = 0
        c = torch.zeros(self.output_dim, device=self.device)
        
        with torch.no_grad():
            for batch in train_loader:
                # Get batch
                x = batch['image'].to(self.device)
                
                # Forward pass
                outputs = self.net(x)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
                
        # Average
        c /= n_samples
        
        # If c is close to zero, add small constant
        if torch.norm(c) < eps:
            c = torch.randn_like(c) * eps
            
        self.center = c
        
    def fit(self, train_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Train the Deep SVDD model.
        
        Args:
            train_loader: DataLoader containing training data
            
        Returns:
            Dictionary of training metrics
        """
        # Initialize center
        self.init_center(train_loader)
        
        self.net.train()
        train_losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                # Get batch
                x = batch['image'].to(self.device)
                
                # Forward pass
                outputs = self.net(x)
                
                # Calculate distance to center
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                
                # Calculate loss based on objective
                if self.objective == 'one-class':
                    loss = torch.mean(dist)
                else:  # soft-boundary
                    loss = self.radius**2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(dist), dist - self.radius**2))
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                
            # Average loss for the epoch
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
            
        # Set radius (for soft-boundary)
        if self.objective == 'soft-boundary':
            self._set_radius(train_loader)
            
        # Calculate threshold based on training data
        anomaly_scores = self._get_anomaly_scores(train_loader)
        self.threshold = np.percentile(anomaly_scores, 95)  # 95th percentile
        
        self.is_fitted = True
        
        return {"loss": train_losses}
    
    def _set_radius(self, train_loader: DataLoader) -> None:
        """
        Set the radius for soft-boundary objective.
        
        Args:
            train_loader: DataLoader containing training data
        """
        self.net.eval()
        distances = []
        
        with torch.no_grad():
            for batch in train_loader:
                # Get batch
                x = batch['image'].to(self.device)
                
                # Forward pass
                outputs = self.net(x)
                
                # Calculate distance to center
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                distances.extend(dist.cpu().numpy())
                
        # Set radius as the (1-nu) quantile of distances
        self.radius = np.quantile(distances, 1 - self.nu)
        
    def _get_anomaly_scores(self, data_loader: DataLoader) -> np.ndarray:
        """
        Get anomaly scores for the given data.
        
        Args:
            data_loader: DataLoader containing data
            
        Returns:
            Array of anomaly scores
        """
        self.net.eval()
        scores = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Get batch
                x = batch['image'].to(self.device)
                
                # Forward pass
                outputs = self.net(x)
                
                # Calculate distance to center
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                scores.extend(dist.cpu().numpy())
                
        return np.array(scores)
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly scores for the given data.
        
        Args:
            data_loader: DataLoader containing data to predict
            
        Returns:
            Tuple of (anomaly_scores, labels)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        self.net.eval()
        anomaly_scores = []
        labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Get batch
                x = batch['image'].to(self.device)
                y = batch['label'].cpu().numpy() if 'label' in batch else None
                
                # Forward pass
                outputs = self.net(x)
                
                # Calculate distance to center
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                anomaly_scores.extend(dist.cpu().numpy())
                
                if y is not None:
                    labels.extend(y)
                
        return np.array(anomaly_scores), np.array(labels) if labels else np.array([])
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the model state dictionary.
        
        Returns:
            State dictionary
        """
        return {
            'net': self.net.state_dict(),
            'center': self.center,
            'radius': self.radius,
            'threshold': self.threshold if hasattr(self, 'threshold') else None
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the model state dictionary.
        
        Args:
            state_dict: State dictionary
        """
        self.net.load_state_dict(state_dict['net'])
        self.center = state_dict['center']
        self.radius = state_dict['radius']
        if 'threshold' in state_dict and state_dict['threshold'] is not None:
            self.threshold = state_dict['threshold']
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'input_shape': self.input_shape,
            'output_dim': self.output_dim,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'objective': self.objective,
            'nu': self.nu
        }
    
    def set_params(self, **params) -> None:
        """
        Set the model parameters.
        
        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            setattr(self, key, value)
