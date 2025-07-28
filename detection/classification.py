"""
Module for color extraction and classification.
"""
import cv2
import numpy as np
from typing import Tuple, Dict, Any

def get_dominant_color(image: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
    """
    Extract the dominant color from a masked region of an image.
    
    Args:
        image: Input image in BGR format
        mask: Binary mask where the region of interest is white (255)
        
    Returns:
        Dominant color as a BGR tuple (B, G, R)
    """
    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Get non-zero pixels (pixels that are part of the mask)
    non_zero_pixels = masked_image[mask > 0]
    
    # If no pixels are found, return black
    if len(non_zero_pixels) == 0:
        return (0, 0, 0)
    
    # Calculate average color
    average_color = np.mean(non_zero_pixels, axis=0).astype(int)
    
    # Return as BGR tuple
    return (int(average_color[0]), int(average_color[1]), int(average_color[2]))

def classify_hsv_color(hsv_color: Tuple[int, int, int]) -> str:
    """
    Classify an HSV color into a named color category.
    
    Args:
        hsv_color: HSV color tuple (H, S, V)
        
    Returns:
        Color name as a string
    """
    h, s, v = hsv_color
    
    # Very low saturation or value indicates black, gray, or white
    if v < 50:
        return "black"
    if s < 30 and v > 150:
        return "white"
    if s < 50:
        return "gray"
    
    # Classify based on hue
    if (0 <= h < 10) or (160 <= h <= 180):
        return "red"
    elif 10 <= h < 25:
        return "orange"
    elif 25 <= h < 35:
        return "yellow"
    elif 35 <= h < 85:
        return "green"
    elif 85 <= h < 130:
        return "blue"
    elif 130 <= h < 145:
        return "purple"
    elif 145 <= h < 165:
        return "pink"
    else:
        return "unknown"

def extract_pencil_color(image: np.ndarray, pencil: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract color information for a pencil and add it to the pencil dictionary.
    
    Args:
        image: Input image in BGR format
        pencil: Pencil information dictionary with contour
        
    Returns:
        Updated pencil dictionary with color information
    """
    # Create a mask for just this pencil
    height, width = image.shape[:2]
    single_pencil_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(single_pencil_mask, [pencil['contour']], 0, 255, -1)
    
    # Get dominant color in BGR
    bgr_color = get_dominant_color(image, single_pencil_mask)
    pencil['bgr_color'] = bgr_color
    
    # Convert to HSV
    hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
    pencil['hsv_color'] = tuple(hsv_color)
    
    # Classify color
    pencil['color_name'] = classify_hsv_color(pencil['hsv_color'])
    
    return pencil
