"""
Dataset loading and preprocessing utilities for computer vision tasks.
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import torch


class ImageDataset(Dataset):
    """
    A PyTorch Dataset for loading and preprocessing images for anomaly detection.
    """
    def __init__(
        self,
        image_paths: List[str],
        labels: Optional[List[int]] = None,
        transform: Optional[A.Compose] = None,
        return_paths: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to images
            labels: Optional list of labels (0 for normal, 1 for anomaly)
            transform: Albumentations transformations to apply
            return_paths: Whether to return image paths along with images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.return_paths = return_paths
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        result = {"image": image}
        
        # Add label if available
        if self.labels is not None:
            result["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        # Add path if requested
        if self.return_paths:
            result["path"] = img_path
            
        return result


class DatasetManager:
    """
    Manager class for handling datasets for anomaly detection.
    """
    def __init__(
        self,
        data_dir: str,
        img_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        val_split: float = 0.2,
        test_split: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize the dataset manager.
        
        Args:
            data_dir: Directory containing the dataset
            img_size: Target image size (height, width)
            batch_size: Batch size for dataloaders
            val_split: Validation split ratio
            test_split: Test split ratio
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        
        # Default transformations
        self.train_transform = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        self.val_transform = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
    def find_images(self, subdirs: Optional[List[str]] = None) -> Tuple[List[str], List[int]]:
        """
        Find all images in the data directory and assign labels based on subdirectory names.
        
        Args:
            subdirs: Optional list of subdirectories to include
            
        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []
        
        # If subdirs not provided, use all subdirectories
        if subdirs is None:
            subdirs = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        
        # Collect images from each subdirectory
        for subdir in subdirs:
            subdir_path = self.data_dir / subdir
            if not subdir_path.exists():
                continue
                
            # Determine label (assume "normal" is class 0, others are anomalies)
            is_anomaly = 0 if subdir.lower() == "normal" else 1
            
            # Find all images
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in subdir_path.glob(ext):
                    image_paths.append(str(img_path))
                    labels.append(is_anomaly)
                    
        return image_paths, labels
    
    def create_data_loaders(
        self, 
        image_paths: Optional[List[str]] = None, 
        labels: Optional[List[int]] = None
    ) -> Dict[str, DataLoader]:
        """
        Create train, validation, and test data loaders.
        
        Args:
            image_paths: Optional list of image paths (if None, will be found automatically)
            labels: Optional list of labels (if None, will be inferred from directory structure)
            
        Returns:
            Dictionary containing train, val, and test dataloaders
        """
        # Find images if not provided
        if image_paths is None or labels is None:
            image_paths, labels = self.find_images()
            
        # Split data
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, test_size=self.val_split + self.test_split, 
            random_state=self.random_state, stratify=labels
        )
        
        # Further split into validation and test
        val_ratio = self.test_split / (self.val_split + self.test_split)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=val_ratio, 
            random_state=self.random_state, stratify=temp_labels
        )
        
        # Create datasets
        train_dataset = ImageDataset(train_paths, train_labels, transform=self.train_transform)
        val_dataset = ImageDataset(val_paths, val_labels, transform=self.val_transform)
        test_dataset = ImageDataset(test_paths, test_labels, transform=self.val_transform, return_paths=True)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        
        return {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }
    
    def get_class_weights(self, labels: List[int]) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.
        
        Args:
            labels: List of labels
            
        Returns:
            Tensor of class weights
        """
        class_counts = np.bincount(labels)
        total = len(labels)
        weights = torch.FloatTensor(total / (len(class_counts) * class_counts))
        return weights
    
    def save_split_info(self, output_dir: str, train_paths: List[str], val_paths: List[str], 
                       test_paths: List[str], train_labels: List[int], val_labels: List[int], 
                       test_labels: List[int]) -> None:
        """
        Save train/val/test split information to CSV files.
        
        Args:
            output_dir: Directory to save split information
            train_paths, val_paths, test_paths: Lists of image paths
            train_labels, val_labels, test_labels: Lists of labels
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrames
        train_df = pd.DataFrame({"path": train_paths, "label": train_labels})
        val_df = pd.DataFrame({"path": val_paths, "label": val_labels})
        test_df = pd.DataFrame({"path": test_paths, "label": test_labels})
        
        # Save to CSV
        train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val_split.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)


def get_transforms(img_size: Tuple[int, int] = (224, 224), augment: bool = True) -> Dict[str, A.Compose]:
    """
    Get standard image transformations for training and validation.
    
    Args:
        img_size: Target image size (height, width)
        augment: Whether to include data augmentation for training
        
    Returns:
        Dictionary containing train and val transforms
    """
    train_transforms = []
    
    # Basic transforms
    train_transforms.extend([
        A.Resize(height=img_size[0], width=img_size[1]),
    ])
    
    # Augmentation transforms
    if augment:
        train_transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=1),
                A.GaussianBlur(p=1),
                A.MedianBlur(blur_limit=3, p=1),
            ], p=0.2),
        ])
    
    # Normalization (using ImageNet stats)
    train_transforms.append(
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    )
    
    # Validation transforms (only resize and normalize)
    val_transforms = [
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
    
    return {
        "train": A.Compose(train_transforms),
        "val": A.Compose(val_transforms)
    }
