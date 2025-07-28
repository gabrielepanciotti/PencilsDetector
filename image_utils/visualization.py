"""
Module for visualizing detection results.
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

def draw_bounding_boxes(
    image: np.ndarray, 
    pencils_by_color: Dict[str, List[Dict[str, Any]]], 
    green_pencils: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Draw bounding boxes and labels for detected pencils.
    
    Args:
        image: Original image
        pencils_by_color: Dictionary of pencils grouped by color
        green_pencils: List of green pencils
        
    Returns:
        Image with bounding boxes and labels
    """
    # Create a copy of the image to avoid modifying the original
    vis_image = image.copy()
    
    # Define colors for different pencil types (BGR format)
    color_map = {
        'green': (0, 255, 0),    # Green
        'red': (0, 0, 255),      # Red
        'blue': (255, 0, 0),     # Blue
        'yellow': (0, 255, 255), # Yellow
        'orange': (0, 165, 255), # Orange
        'purple': (255, 0, 255), # Purple
        'pink': (203, 192, 255), # Pink
        'brown': (42, 42, 165),  # Brown
        'black': (0, 0, 0),      # Black
        'gray': (128, 128, 128), # Gray
        'white': (255, 255, 255) # White
    }
    
    # Draw bounding boxes for all pencils
    for color_name, pencils in pencils_by_color.items():
        # Get the color for drawing (default to white if not in map)
        draw_color = color_map.get(color_name, (255, 255, 255))
        
        for pencil in pencils:
            x, y, w, h = pencil['bbox']
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), draw_color, 2)
            
            # Draw label with color name
            label = f"{color_name}"
            cv2.putText(vis_image, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
    
    return vis_image

def save_visualization(image: np.ndarray, output_path: str) -> None:
    """
    Save visualization image to disk.
    
    Args:
        image: Image to save
        output_path: Path to save the image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Visualization saved to {output_path}")

def save_debug_images(
    images: Dict[str, np.ndarray], 
    debug_dir: str,
    process_name: str = ""
) -> None:
    """
    Save debug images to disk with numbered labels and process descriptions.
    
    Args:
        images: Dictionary of images to save (name -> image)
        debug_dir: Directory to save debug images
        process_name: Name of the processing step (optional)
    """
    # Create debug directory if it doesn't exist
    os.makedirs(debug_dir, exist_ok=True)
    
    # Save each image with numbered labels
    for i, (name, image) in enumerate(images.items(), 1):
        # Create a copy of the image for adding text
        labeled_image = image.copy()
        
        # Prepare the label text
        if process_name:
            label_text = f"{i}. {process_name} - {name}"
        else:
            label_text = f"{i}. {name}"
        
        # Add the label to the image if it's not a binary mask
        if len(labeled_image.shape) == 2:  # Binary mask
            # Convert to 3 channel for text
            labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_GRAY2BGR)
        
        # Add text at the top of the image
        cv2.putText(labeled_image, label_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Save the image
        output_path = os.path.join(debug_dir, f"{i:02d}_{name}.jpg")
        cv2.imwrite(output_path, labeled_image)
