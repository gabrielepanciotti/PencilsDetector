"""
Image segmentation models and utilities for anomaly detection in computer vision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union
import os
from tqdm import tqdm
from skimage import segmentation, color, morphology


class UNetDown(nn.Module):
    """U-Net downsampling block."""
    
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True, dropout: float = 0.0):
        """
        Initialize U-Net downsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            normalize: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
            
        layers.append(nn.LeakyReLU(0.2))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class UNetUp(nn.Module):
    """U-Net upsampling block."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        """
        Initialize U-Net upsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            skip: Skip connection tensor
            
        Returns:
            Output tensor
        """
        x = self.model(x)
        return torch.cat([x, skip], dim=1)


class UNet(nn.Module):
    """U-Net architecture for image segmentation."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        """
        Initialize U-Net.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        
        # Downsampling
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        
        # Upsampling
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Downsampling
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # Upsampling
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final(u7)


class SegmentationModel:
    """Base class for segmentation models."""
    
    def __init__(self, device: str = None):
        """
        Initialize segmentation model.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
    
    def build_model(self):
        """Build the model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement build_model()")
    
    def train_epoch(self, data_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            data_loader: Training data loader
            
        Returns:
            Tuple of (average loss, average IoU)
        """
        self.model.train()
        running_loss = 0.0
        running_iou = 0.0
        
        for images, masks in tqdm(data_loader, desc="Training", leave=False):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            # Calculate IoU
            pred_masks = (outputs > 0.5).float()
            iou = self._calculate_iou(pred_masks, masks)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            running_iou += iou * images.size(0)
        
        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_iou = running_iou / len(data_loader.dataset)
        
        return epoch_loss, epoch_iou
    
    def validate_epoch(self, data_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            data_loader: Validation data loader
            
        Returns:
            Tuple of (average loss, average IoU)
        """
        self.model.eval()
        running_loss = 0.0
        running_iou = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(data_loader, desc="Validating", leave=False):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Calculate IoU
                pred_masks = (outputs > 0.5).float()
                iou = self._calculate_iou(pred_masks, masks)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                running_iou += iou * images.size(0)
        
        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_iou = running_iou / len(data_loader.dataset)
        
        return epoch_loss, epoch_iou
    
    def _calculate_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate IoU (Intersection over Union).
        
        Args:
            pred: Predicted masks
            target: Target masks
            
        Returns:
            IoU score
        """
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection
        
        return (intersection / (union + 1e-6)).item()
    
    def fit(self, train_loader: torch.utils.data.DataLoader, 
           val_loader: Optional[torch.utils.data.DataLoader] = None,
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
            # Train
            train_loss, train_iou = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_iou'].append(train_iou)
            
            # Validate
            if val_loader is not None:
                val_loss, val_iou = self.validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_iou'].append(val_iou)
                
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
                          f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        
        # Load best model if validation was used
        if val_loader is not None and hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            
        return self.history
    
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict segmentation mask.
        
        Args:
            image: Input image tensor
            
        Returns:
            Predicted mask
        """
        self.model.eval()
        
        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image)
            mask = (output > 0.5).float()
            
        return mask.cpu()
    
    def predict_numpy(self, image: np.ndarray) -> np.ndarray:
        """
        Predict segmentation mask from numpy array.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Predicted mask as numpy array
        """
        # Convert to tensor
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            
        # Add batch and channel dimensions if needed
        if len(image.shape) == 2:
            # Grayscale image
            image = image[np.newaxis, np.newaxis, :, :]
        elif len(image.shape) == 3:
            # RGB image
            image = image.transpose(2, 0, 1)[np.newaxis, :, :, :]
            
        image_tensor = torch.from_numpy(image).float()
        
        # Predict
        mask_tensor = self.predict(image_tensor)
        
        # Convert back to numpy
        mask = mask_tensor.numpy()[0, 0]
        
        return mask
    
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
            self.build_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            
        print(f"Model loaded from {path}")
    
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
        
        # Plot IoU
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_iou'], label='Train IoU')
        if 'val_iou' in self.history and self.history['val_iou']:
            plt.plot(self.history['val_iou'], label='Validation IoU')
        plt.title('IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
