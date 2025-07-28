"""
Main module for pencil detection.
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

# Import from image_utils
from image_utils.loader import load_image
from image_utils.preprocessing import preprocess_image
from image_utils.visualization import draw_bounding_boxes, save_visualization, save_debug_images

# Import from detection
from detection.color_masks import get_green_mask, create_color_segmentation
from detection.contour_utils import find_pencils_in_mask, detect_edges
from detection.classification import extract_pencil_color, classify_hsv_color, extract_pencil_color

# Import from results
from results.exporter import generate_results_json
from results.summary import print_detection_summary, print_debug_info, group_pencils_by_color

# Import configuration
from config import COLOR_RANGES, CONTOUR_PARAMS, MORPH_PARAMS

def process_image(image: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process image once for all detection functions.
    
    Args:
        image: Input image in BGR format
        debug: If True, save debug images
        
    Returns:
        Tuple of (hsv_image, blurred_image, edge_mask, segmentation_mask)
    """
    # Preprocess image
    hsv, blurred = preprocess_image(image)
    
    # Create color-based segmentation mask
    segmentation_mask = create_color_segmentation(hsv)
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding with larger block size for better pencil detection
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 2)  # Increased block size from 11 to 21
    
    # Apply Canny edge detection with lower thresholds to catch more edges
    edges = cv2.Canny(gray, 20, 100)  # Lowered thresholds
    
    # Combine thresholding and edge detection
    edge_mask = cv2.bitwise_or(thresh, edges)
    
    # Apply dilation to connect broken edges
    kernel = np.ones((5, 5), np.uint8)
    edge_mask = cv2.dilate(edge_mask, kernel, iterations=2)
    
    # Save debug images if requested
    if debug:
        debug_dir = os.path.join(os.getcwd(), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create a colorized version of the segmentation mask for better visualization
        segmentation_vis = cv2.applyColorMap(segmentation_mask, cv2.COLORMAP_JET)
        
        # Create an overlay of the segmentation on the original image
        segmentation_overlay = image.copy()
        segmentation_overlay = cv2.addWeighted(segmentation_vis, 0.5, segmentation_overlay, 0.5, 0)
        
        debug_images = {
            "gray": gray,
            "thresh": thresh,
            "edges": edges,
            "edge_mask": edge_mask,
            "edge_overlay": cv2.bitwise_and(image, image, mask=edge_mask),
            "segmentation": segmentation_mask,
            "segmentation_vis": segmentation_vis,
            "segmentation_overlay": segmentation_overlay
        }
        save_debug_images(debug_images, debug_dir, "Image Processing")
    
    return hsv, blurred, edge_mask, segmentation_mask

def detect_green_pencils(image: np.ndarray, hsv: np.ndarray = None, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Detect green pencils in the image.
    
    Args:
        image: Input image in BGR format
        hsv: HSV image (optional, will be computed if not provided)
        debug: If True, save debug images
        
    Returns:
        List of dictionaries containing green pencil information
    """
    # Preprocess image if HSV not provided
    if hsv is None:
        hsv, _ = preprocess_image(image)
    
    # Create mask for green color
    green_mask = get_green_mask(hsv)
    
    # Apply morphological operations to improve the mask
    kernel = np.ones((9, 9), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    green_mask = cv2.dilate(green_mask, kernel, iterations=2)
    
    # Save debug images if requested
    if debug:
        debug_dir = os.path.join(os.getcwd(), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        debug_images = {
            "green_mask": green_mask,
            "green_mask_overlay": cv2.bitwise_and(image, image, mask=green_mask)
        }
        save_debug_images(debug_images, debug_dir, "Green Pencil Detection")
    
    # Find pencils in the green mask
    green_pencils = find_pencils_in_mask(
        green_mask, 
        min_area=CONTOUR_PARAMS['min_area'], 
        max_area=CONTOUR_PARAMS['max_area'],
        min_aspect_ratio=CONTOUR_PARAMS['min_aspect_ratio']
    )
    
    # For each green pencil, extract its dominant color
    for i, pencil in enumerate(green_pencils):
        # Extract color information
        extract_pencil_color(image, pencil)
        
        # Override color name since we know it's green
        pencil['color_name'] = "green"
        
        # Save debug image if requested
        if debug:
            x, y, w, h = pencil['bbox']
            # Extract the pencil region
            pencil_img = image[y:y+h, x:x+w].copy()
            cv2.imwrite(os.path.join(debug_dir, f"green_pencil_{i}.jpg"), pencil_img)
    
    print(f"Found {len(green_pencils)} green pencils")
    return green_pencils

def detect_all_pencils(image: np.ndarray, hsv: np.ndarray = None, edge_mask: np.ndarray = None, segmentation_mask: np.ndarray = None, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Detect all pencils in the image and classify them by color using color segmentation.
    
    Args:
        image: Input image in BGR format
        hsv: HSV image (optional, will be computed if not provided)
        edge_mask: Edge mask (optional, will be computed if not provided)
        segmentation_mask: Segmentation mask (optional, will be computed if not provided)
        debug: If True, save debug images
        
    Returns:
        List of dictionaries containing pencil information with color classification
    """
    # If segmentation_mask not provided, compute it
    if segmentation_mask is None or hsv is None:
        hsv, _, _, segmentation_mask = process_image(image, debug)
    
    # Find unique labels in the segmentation mask (each represents a potential pencil)
    unique_labels = np.unique(segmentation_mask)
    unique_labels = unique_labels[unique_labels > 0]  # Skip background (0)
    
    pencils = []
    for i, label in enumerate(unique_labels):
        # Create a mask for this specific segment
        segment_mask = np.zeros(segmentation_mask.shape, dtype=np.uint8)
        segment_mask[segmentation_mask == label] = 255
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        segment_mask = cv2.morphologyEx(segment_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours in this segment
        contours, _ = cv2.findContours(segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Skip if no contours found
        if not contours:
            continue
            
        # Get the largest contour (should be the pencil)
        contour = max(contours, key=cv2.contourArea)
        
        # Filter by area
        area = cv2.contourArea(contour)
        if area < CONTOUR_PARAMS['min_area'] or area > CONTOUR_PARAMS['max_area']:
            continue
            
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio
        aspect_ratio = max(h, w) / max(1, min(h, w))  # Avoid division by zero
        
        # Filter by aspect ratio - pencils should be elongated
        if aspect_ratio < CONTOUR_PARAMS['min_aspect_ratio']:
            continue
            
        # Calculate center
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Create pencil info
        pencil_info = {
            'contour': contour,
            'bbox': (x, y, w, h),
            'center': (center_x, center_y),
            'area': area,
            'aspect_ratio': aspect_ratio,
            'label': int(label)
        }
        
        # Extract color information
        extract_pencil_color(image, pencil_info)
        
        pencils.append(pencil_info)
        
        # Save debug image if requested
        if debug:
            debug_dir = os.path.join(os.getcwd(), "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Extract the pencil region
            pencil_img = image[y:y+h, x:x+w].copy()
            cv2.imwrite(os.path.join(debug_dir, f"pencil_{i}_{pencil_info['color_name']}.jpg"), pencil_img)
            
            # Save the segment mask
            cv2.imwrite(os.path.join(debug_dir, f"segment_{i}_mask.jpg"), segment_mask)
    
    print(f"Found {len(pencils)} pencils in total")
    return pencils

def visualize_results(
    image: np.ndarray, 
    green_pencils: List[Dict[str, Any]], 
    all_pencils: List[Dict[str, Any]], 
    output_path: str
) -> np.ndarray:
    """
    Visualize detection results and save to file.
    
    Args:
        image: Original image
        green_pencils: List of detected green pencils
        all_pencils: List of all detected pencils
        output_path: Path to save the visualization
        
    Returns:
        Visualization image
    """
    # Group pencils by color
    pencils_by_color = group_pencils_by_color(all_pencils)
    
    # Draw bounding boxes
    vis_image = draw_bounding_boxes(image, pencils_by_color, green_pencils)
    
    # Save visualization
    save_visualization(vis_image, output_path)
    
    return vis_image
