"""
Module for contour detection and filtering.
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

def find_pencils_in_mask(
    mask: np.ndarray, 
    min_area: int = 500, 
    max_area: int = 100000,
    min_aspect_ratio: float = 2.5
) -> List[Dict[str, Any]]:
    """
    Find pencil contours in a binary mask and filter by area and aspect ratio.
    
    Args:
        mask: Binary mask where pencils are white
        min_area: Minimum contour area to consider
        max_area: Maximum contour area to consider
        min_aspect_ratio: Minimum aspect ratio (length/width) for a pencil
        
    Returns:
        List of dictionaries with contour, bounding box, center and area information
    """
    # Apply morphological operations to improve contour detection
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours with hierarchy to handle nested contours better
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and aspect ratio
    pencils = []
    for i, contour in enumerate(contours):
        # Skip child contours (inner contours)
        if hierarchy is not None and hierarchy[0][i][3] != -1:
            continue
            
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            # Get rotated rectangle which better fits elongated objects
            rect = cv2.minAreaRect(contour)
            (center_x, center_y), (width, height), angle = rect
            
            # Get regular bounding box for compatibility
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio - pencils are typically elongated
            aspect_ratio = max(width, height) / max(1, min(width, height))  # Avoid division by zero
            
            # Filter by aspect ratio - pencils should be elongated
            if aspect_ratio >= min_aspect_ratio:
                # Calculate solidity (area / convex hull area) to filter irregular shapes
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # Only keep contours with reasonable solidity (not too irregular)
                if solidity > 0.5:
                    pencil_info = {
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'center': (int(center_x), int(center_y)),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'solidity': solidity,
                        'angle': angle
                    }
                    
                    pencils.append(pencil_info)
    
    return pencils

def create_contour_mask(image_shape: Tuple[int, int], contour: np.ndarray) -> np.ndarray:
    """
    Create a binary mask for a single contour.
    
    Args:
        image_shape: Shape of the original image (height, width)
        contour: Contour points
        
    Returns:
        Binary mask where the contour area is white (255)
    """
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    return mask

def detect_edges(image: np.ndarray) -> np.ndarray:
    """
    Detect edges in an image using a combination of adaptive thresholding and Canny edge detection.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        Binary edge mask
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply adaptive thresholding with larger block size for better pencil detection
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 21, 2)
    
    # Apply Canny edge detection with lower thresholds to catch more edges
    edges = cv2.Canny(blurred, 20, 100)
    
    # Combine thresholding and edge detection
    combined = cv2.bitwise_or(thresh, edges)
    
    # Apply morphological operations to clean up and connect edges
    kernel_close = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Dilate edges to connect gaps
    kernel_dilate = np.ones((7, 7), np.uint8)
    dilated_edges = cv2.dilate(closed, kernel_dilate, iterations=2)
    
    return dilated_edges
