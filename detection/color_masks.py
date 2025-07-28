"""
Module for creating color masks using HSV thresholds.
"""
import cv2
import numpy as np
from typing import Tuple, Dict

from config import COLOR_RANGES

def create_color_segmentation(hsv_image: np.ndarray) -> np.ndarray:
    """
    Create a segmentation mask based on color differences.
    This approach uses color differences to separate adjacent pencils.
    
    Args:
        hsv_image: Input image in HSV format
        
    Returns:
        Segmentation mask with different colors for different segments
    """
    # Extract the hue channel which contains color information
    hue = hsv_image[:, :, 0]
    
    # Apply median blur to reduce noise while preserving edges
    hue_blurred = cv2.medianBlur(hue, 7)
    
    # Apply adaptive thresholding to find edges between different colors
    # This works because adjacent pencils have different hue values
    edges = cv2.adaptiveThreshold(hue_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    
    # Dilate the edges to ensure they separate the pencils
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Use watershed algorithm for better segmentation
    # First, find sure background
    sure_bg = cv2.dilate(edges_dilated, kernel, iterations=3)
    
    # Find markers for watershed
    ret, markers = cv2.connectedComponents(np.uint8(255) - edges_dilated)
    
    # Add 1 to all labels to ensure background is not 0
    markers = markers + 1
    
    # Mark the region of edges as unknown (0)
    markers[edges_dilated == 255] = 0
    
    # Convert HSV to BGR for watershed
    bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    # Apply watershed
    cv2.watershed(bgr, markers)
    
    # Create a mask where each pencil has a different value
    segmentation = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for i in range(2, ret+1):  # Skip background (1) and edges (0)
        # Limit the value to 255 (max for uint8)
        label_value = min(i * 10, 250)  # Use smaller multiplier and cap at 250
        segmentation[markers == i] = label_value
    
    return segmentation

def create_color_mask(
    hsv_image: np.ndarray, 
    lower_bound: np.ndarray, 
    upper_bound: np.ndarray,
    kernel_size: int = 9,
    iterations: int = 2
) -> np.ndarray:
    """
    Create a binary mask for a given HSV color range with enhanced processing.
    
    Args:
        hsv_image: Input image in HSV format
        lower_bound: Lower HSV bound for the color range
        upper_bound: Upper HSV bound for the color range
        kernel_size: Size of morphological kernel
        iterations: Number of iterations for morphological operations
        
    Returns:
        Binary mask where the color is white (255)
    """
    # Create initial mask using HSV thresholds
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # Create kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply morphological opening to remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Apply morphological closing to fill gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # Apply dilation to expand the mask slightly
    mask = cv2.dilate(mask, kernel, iterations=iterations)
    
    return mask

def get_green_mask(hsv_image: np.ndarray) -> np.ndarray:
    """
    Create a binary mask for green color with enhanced processing.
    
    Args:
        hsv_image: Input image in HSV format
        
    Returns:
        Binary mask where green pixels are white (255)
    """
    # Get green color range from config
    green_range = COLOR_RANGES['green']
    lower_green = np.array(green_range['lower'])
    upper_green = np.array(green_range['upper'])
    
    # Create initial green mask
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    # Create kernel for morphological operations
    kernel = np.ones((9, 9), np.uint8)
    
    # Apply morphological opening to remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Apply morphological closing to fill gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Apply dilation to expand the mask slightly
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask
