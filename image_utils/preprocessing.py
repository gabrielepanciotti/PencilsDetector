"""
Module for image preprocessing operations.
"""
import cv2
import numpy as np
from typing import Tuple

def to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to HSV color space.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        Image in HSV format
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def blur_image(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur to an image to reduce noise.
    
    Args:
        image: Input image
        kernel_size: Size of the Gaussian kernel (must be odd)
        
    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image for pencil detection.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        Tuple containing (hsv_image, blurred_image)
    """
    # Convert to HSV color space
    hsv = to_hsv(image)
    
    # Apply slight Gaussian blur to reduce noise
    blurred = blur_image(image)
    
    return hsv, blurred
