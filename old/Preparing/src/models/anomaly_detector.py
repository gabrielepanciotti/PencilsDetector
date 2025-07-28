"""
Anomaly detection models for computer vision tasks.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from abc import ABC, abstractmethod

# Import local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import calculate_metrics


class BaseAnomalyDetector(ABC):
    """
    Base class for all anomaly detection models.
    """
    def __init__(self, name: str = "BaseAnomalyDetector"):
        """
        Initialize the base anomaly detector.
        
        Args:
            name: Name of the model
        """
        self.name = name
        self.is_fitted = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def fit(self, train_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Train the model on normal data.
        
        Args:
            train_loader: DataLoader containing training data
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly scores for the given data.
        
        Args:
            data_loader: DataLoader containing data to predict
            
        Returns:
            Tuple of (anomaly_scores, labels)
        """
        pass
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader containing test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        
        # Get predictions
        anomaly_scores, labels = self.predict(test_loader)
        
        # Calculate metrics
        metrics = calculate_metrics(labels, anomaly_scores)
        
        return metrics
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.state_dict() if hasattr(self, 'state_dict') else None,
            'name': self.name,
            'is_fitted': self.is_fitted,
            'model_params': self.get_params() if hasattr(self, 'get_params') else {}
        }, path)
        
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        if hasattr(self, 'load_state_dict') and checkpoint['model_state'] is not None:
            self.load_state_dict(checkpoint['model_state'])
        self.name = checkpoint['name']
        self.is_fitted = checkpoint['is_fitted']
        if hasattr(self, 'set_params') and 'model_params' in checkpoint:
            self.set_params(**checkpoint['model_params'])
            
    def visualize_results(self, test_loader: DataLoader, num_samples: int = 10, 
                         save_path: Optional[str] = None) -> None:
        """
        Visualize anomaly detection results.
        
        Args:
            test_loader: DataLoader containing test data
            num_samples: Number of samples to visualize
            save_path: Path to save the visualization
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before visualization")
        
        # Get predictions
        anomaly_scores, labels = self.predict(test_loader)
        
        # Get images
        images = []
        paths = []
        for batch in test_loader:
            batch_images = batch['image']
            if 'path' in batch:
                batch_paths = batch['path']
                paths.extend(batch_paths)
            images.extend(batch_images)
            if len(images) >= num_samples:
                break
        
        # Limit to num_samples
        images = images[:num_samples]
        anomaly_scores = anomaly_scores[:num_samples]
        labels = labels[:num_samples]
        
        # Create figure
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
        
        # Plot images and anomaly scores
        for i in range(num_samples):
            # Convert tensor to numpy
            if isinstance(images[i], torch.Tensor):
                img = images[i].permute(1, 2, 0).cpu().numpy()
                # Denormalize if needed
                if img.max() <= 1.0:
                    img = img * 255.0
                img = img.astype(np.uint8)
            else:
                img = images[i]
            
            # Plot image
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Label: {'Anomaly' if labels[i] == 1 else 'Normal'}")
            axes[0, i].axis('off')
            
            # Plot anomaly score
            axes[1, i].bar(['Anomaly Score'], [anomaly_scores[i]])
            axes[1, i].set_ylim([0, max(1.0, np.max(anomaly_scores) * 1.1)])
            axes[1, i].set_title(f"Score: {anomaly_scores[i]:.4f}")
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            
        plt.show()


class AutoencoderAnomalyDetector(BaseAnomalyDetector):
    """
    Anomaly detection using an autoencoder.
    The model is trained on normal data only and anomalies are detected
    based on reconstruction error.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        latent_dim: int = 128,
        hidden_dims: List[int] = [32, 64, 128, 256],
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        name: str = "AutoencoderAnomalyDetector"
    ):
        """
        Initialize the autoencoder anomaly detector.
        
        Args:
            input_shape: Shape of input images (channels, height, width)
            latent_dim: Dimension of the latent space
            hidden_dims: Dimensions of hidden layers
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            batch_size: Batch size for training
            name: Name of the model
        """
        super().__init__(name=name)
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Build encoder and decoder
        self._build_model()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate
        )
        
        # Track reconstruction errors on normal data
        self.normal_errors = None
        
    def _build_model(self) -> None:
        """
        Build the encoder and decoder networks.
        """
        # Encoder
        encoder_layers = []
        in_channels = self.input_shape[0]
        
        for h_dim in self.hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate size of flattened features
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)
            dummy_output = self.encoder(dummy_input)
            self.feature_size = dummy_output.numel()
        
        # Latent representation
        self.fc_mu = nn.Linear(self.feature_size, self.latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(self.latent_dim, self.feature_size)
        
        # Decoder
        decoder_layers = []
        hidden_dims = self.hidden_dims[::-1]  # Reverse for decoder
        
        for i in range(len(hidden_dims) - 1):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                     kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
            
        # Final layer
        decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], self.input_shape[0],
                                 kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()  # Output in [0, 1]
            )
        )
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Move to device
        self.encoder.to(self.device)
        self.fc_mu.to(self.device)
        self.decoder_input.to(self.device)
        self.decoder.to(self.device)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation
        """
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        z = self.fc_mu(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.
        
        Args:
            z: Latent representation
            
        Returns:
            Reconstructed input
        """
        x = self.decoder_input(z)
        
        # Reshape to match the encoder output shape
        batch_size = x.shape[0]
        feature_shape = (batch_size, self.hidden_dims[-1], 
                        self.input_shape[1] // 2**len(self.hidden_dims),
                        self.input_shape[2] // 2**len(self.hidden_dims))
        x = x.view(feature_shape)
        
        # Decode
        x = self.decoder(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstructed input
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
    
    def fit(self, train_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Train the autoencoder on normal data.
        
        Args:
            train_loader: DataLoader containing normal training data
            
        Returns:
            Dictionary of training metrics
        """
        self.train()
        
        train_losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                # Get batch
                x = batch['image'].to(self.device)
                
                # Forward pass
                x_hat = self.forward(x)
                
                # Compute loss
                loss = F.mse_loss(x_hat, x)
                
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
            
        # Compute reconstruction errors on normal data
        self.normal_errors = self._compute_reconstruction_errors(train_loader)
        
        # Set threshold as mean + 3*std of reconstruction errors on normal data
        self.threshold = np.mean(self.normal_errors) + 3 * np.std(self.normal_errors)
        
        self.is_fitted = True
        
        return {"loss": train_losses}
    
    def _compute_reconstruction_errors(self, data_loader: DataLoader) -> np.ndarray:
        """
        Compute reconstruction errors for the given data.
        
        Args:
            data_loader: DataLoader containing data
            
        Returns:
            Array of reconstruction errors
        """
        self.eval()
        errors = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Get batch
                x = batch['image'].to(self.device)
                
                # Forward pass
                x_hat = self.forward(x)
                
                # Compute error (MSE per sample)
                error = F.mse_loss(x_hat, x, reduction='none')
                error = error.mean(dim=[1, 2, 3]).cpu().numpy()
                
                errors.extend(error)
                
        return np.array(errors)
    
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
        
        self.eval()
        anomaly_scores = []
        labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Get batch
                x = batch['image'].to(self.device)
                y = batch['label'].cpu().numpy() if 'label' in batch else None
                
                # Forward pass
                x_hat = self.forward(x)
                
                # Compute error (MSE per sample)
                error = F.mse_loss(x_hat, x, reduction='none')
                error = error.mean(dim=[1, 2, 3]).cpu().numpy()
                
                # Normalize by normal data statistics
                normalized_error = (error - np.mean(self.normal_errors)) / (np.std(self.normal_errors) + 1e-10)
                
                anomaly_scores.extend(normalized_error)
                if y is not None:
                    labels.extend(y)
                
        return np.array(anomaly_scores), np.array(labels) if labels else np.array([])
    
    def train(self, mode: bool = True) -> 'AutoencoderAnomalyDetector':
        """
        Set the model to training mode.
        
        Args:
            mode: Whether to set training mode
            
        Returns:
            Self
        """
        self.encoder.train(mode)
        self.decoder.train(mode)
        return self
    
    def eval(self) -> 'AutoencoderAnomalyDetector':
        """
        Set the model to evaluation mode.
        
        Returns:
            Self
        """
        self.encoder.eval()
        self.decoder.eval()
        return self
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the model state dictionary.
        
        Returns:
            State dictionary
        """
        return {
            'encoder': self.encoder.state_dict(),
            'fc_mu': self.fc_mu.state_dict(),
            'decoder_input': self.decoder_input.state_dict(),
            'decoder': self.decoder.state_dict(),
            'normal_errors': self.normal_errors,
            'threshold': self.threshold if hasattr(self, 'threshold') else None
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the model state dictionary.
        
        Args:
            state_dict: State dictionary
        """
        self.encoder.load_state_dict(state_dict['encoder'])
        self.fc_mu.load_state_dict(state_dict['fc_mu'])
        self.decoder_input.load_state_dict(state_dict['decoder_input'])
        self.decoder.load_state_dict(state_dict['decoder'])
        self.normal_errors = state_dict['normal_errors']
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
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
    
    def set_params(self, **params) -> None:
        """
        Set the model parameters.
        
        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            setattr(self, key, value)
