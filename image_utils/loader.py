"""
Module for loading images from disk.
"""
import os
import cv2
import numpy as np
from typing import Tuple

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from a file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as a NumPy array in BGR format
        
    Raises:
        FileNotFoundError: If the image file does not exist
        ValueError: If the image could not be loaded
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image
