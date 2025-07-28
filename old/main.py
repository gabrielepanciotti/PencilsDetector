#!/usr/bin/env python
"""
Pencil Counter and Localizer
----------------------------
This script analyzes an image of pencils to:
1. Count green pencils
2. Localize green pencils
3. Count and localize other pencils grouped by color

Uses traditional computer vision techniques (no AI/ML).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
from typing import List, Tuple, Dict, Any


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from the specified path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image in BGR format
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    print(f"Loaded image with shape: {image.shape}")
    return image


def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the image for pencil detection.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        Tuple of (HSV image, blurred image)
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Apply slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    return hsv, blurred


def create_mask_for_color(hsv_image: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    """
    Create a binary mask for a given HSV color range.
    
    Args:
        hsv_image: Input image in HSV format
        lower_bound: Lower HSV bound for the color range
        upper_bound: Upper HSV bound for the color range
        
    Returns:
        Binary mask where the color is present
    """
    # Create initial mask
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((9, 9), np.uint8)  # Increased kernel size for better connectivity
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Additional dilation to connect fragmented parts
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask


def find_pencils_in_mask(mask: np.ndarray, min_area: int = 500, max_area: int = 100000) -> List[Dict[str, Any]]:
    """
    Find pencil contours in a binary mask and filter by area and aspect ratio.
    
    Args:
        mask: Binary mask where pencils are white
        min_area: Minimum contour area to consider
        max_area: Maximum contour area to consider
        
    Returns:
        List of dictionaries with contour, bounding box, center and area information
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and aspect ratio
    pencils = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio - pencils are typically elongated
            aspect_ratio = max(h, w) / min(h, w)
            
            # Filter by aspect ratio - pencils should be elongated
            if aspect_ratio >= 2.5:
                # Calculate center
                center_x = x + w // 2
                center_y = y + h // 2
                
                pencil_info = {
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': area,
                    'aspect_ratio': aspect_ratio
                }
                
                pencils.append(pencil_info)
    
    return pencils


def get_dominant_color(image: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
    """
    Extract the dominant color from a region defined by a mask.
    
    Args:
        image: Input image in BGR format
        mask: Binary mask defining the region of interest
        
    Returns:
        Dominant color as BGR tuple
    """
    # Apply mask to the original image
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    
    # Get non-zero pixels (pixels that belong to the pencil)
    non_zero_pixels = masked_img[mask > 0]
    
    # If no pixels, return black
    if len(non_zero_pixels) == 0:
        return (0, 0, 0)
    
    # Calculate average color
    average_color = np.mean(non_zero_pixels, axis=0)
    
    return tuple(map(int, average_color))


def classify_color(hsv_color: Tuple[int, int, int]) -> str:
    """
    Classify a color based on its HSV values.
    
    Args:
        hsv_color: Color in HSV format
        
    Returns:
        Color name as string
    """
    h, s, v = hsv_color
    
    # Define color ranges (hue values)
    # Hue ranges from 0 to 179 in OpenCV
    if s < 50 or v < 50:
        return "gray"
    
    if 0 <= h <= 10 or 170 <= h <= 180:
        return "red"
    elif 11 <= h <= 25:
        return "orange"
    elif 26 <= h <= 35:
        return "yellow"
    elif 36 <= h <= 70:
        return "green"
    elif 71 <= h <= 85:
        return "light_blue"
    elif 86 <= h <= 130:
        return "blue"
    elif 131 <= h <= 160:
        return "purple"
    else:
        return "pink"


def detect_green_pencils(image: np.ndarray, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Detect green pencils in the image.
    
    Args:
        image: Input image in BGR format
        debug: If True, save debug images
        
    Returns:
        List of dictionaries containing green pencil information
    """
    # Preprocess image
    hsv, blurred = preprocess_image(image)
    
    # Define green color range in HSV - expanded range for better detection
    lower_green = np.array([35, 40, 40])  # Adjusted lower bound
    upper_green = np.array([85, 255, 255])  # Adjusted upper bound
    
    # Create mask for green color
    green_mask = create_mask_for_color(hsv, lower_green, upper_green)
    
    # Save debug image if requested
    if debug:
        debug_dir = os.path.join(os.getcwd(), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "green_mask.jpg"), green_mask)
        
        # Create a visualization of the mask on the original image
        mask_overlay = image.copy()
        mask_overlay[green_mask > 0] = [0, 255, 0]  # Green overlay
        cv2.imwrite(os.path.join(debug_dir, "green_mask_overlay.jpg"), mask_overlay)
    
    # Find pencils in the green mask
    green_pencils = find_pencils_in_mask(green_mask)
    
    # For each green pencil, extract its dominant color in BGR and HSV
    for i, pencil in enumerate(green_pencils):
        # Create a mask for just this pencil
        single_pencil_mask = np.zeros_like(green_mask)
        cv2.drawContours(single_pencil_mask, [pencil['contour']], 0, 255, -1)
        
        # Get dominant color in BGR
        bgr_color = get_dominant_color(image, single_pencil_mask)
        pencil['bgr_color'] = bgr_color
        
        # Convert to HSV
        hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
        pencil['hsv_color'] = tuple(hsv_color)
        
        # Classify color
        pencil['color_name'] = "green"  # We know it's green since we used green mask
        
        # Save debug image if requested
        if debug:
            x, y, w, h = pencil['bbox']
            # Extract the pencil region
            pencil_img = image[y:y+h, x:x+w].copy()
            cv2.imwrite(os.path.join(debug_dir, f"green_pencil_{i}.jpg"), pencil_img)
    
    print(f"Found {len(green_pencils)} green pencils")
    return green_pencils


def detect_all_pencils(image: np.ndarray, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Detect all pencils in the image and classify them by color.
    
    Args:
        image: Input image in BGR format
        debug: If True, save debug images
        
    Returns:
        List of dictionaries containing pencil information with color classification
    """
    # Preprocess image
    hsv, blurred = preprocess_image(image)
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for better segmentation
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply Canny edge detection with adjusted parameters
    edges = cv2.Canny(gray, 30, 120)  # Lowered thresholds to detect more edges
    
    # Combine thresholding and edge detection for better results
    combined = cv2.bitwise_or(thresh, edges)
    
    # Dilate edges to connect gaps
    kernel = np.ones((9, 9), np.uint8)  # Larger kernel
    dilated_edges = cv2.dilate(combined, kernel, iterations=3)  # More iterations
    
    # Save debug images if requested
    if debug:
        debug_dir = os.path.join(os.getcwd(), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "gray.jpg"), gray)
        cv2.imwrite(os.path.join(debug_dir, "thresh.jpg"), thresh)
        cv2.imwrite(os.path.join(debug_dir, "edges.jpg"), edges)
        cv2.imwrite(os.path.join(debug_dir, "combined.jpg"), combined)
        cv2.imwrite(os.path.join(debug_dir, "dilated_edges.jpg"), dilated_edges)
    
    # Find contours in the edge map
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size and shape to find pencils
    pencils = []
    for i, contour in enumerate(contours):
        # Calculate area and filter by size
        area = cv2.contourArea(contour)
        if 500 <= area <= 100000:  # Adjusted thresholds
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (pencils are elongated)
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 2.5:  # Pencils typically have high aspect ratio
                # Calculate center
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Create a mask for this pencil
                pencil_mask = np.zeros_like(gray)
                cv2.drawContours(pencil_mask, [contour], 0, 255, -1)
                
                # Get dominant color in BGR
                bgr_color = get_dominant_color(image, pencil_mask)
                
                # Convert to HSV for better color classification
                hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
                
                # Classify color
                color_name = classify_color(hsv_color)
                
                # Store pencil information
                pencil_info = {
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'bgr_color': bgr_color,
                    'hsv_color': tuple(hsv_color),
                    'color_name': color_name
                }
                
                pencils.append(pencil_info)
                
                # Save debug image if requested
                if debug:
                    # Extract the pencil region
                    pencil_img = image[y:y+h, x:x+w].copy()
                    cv2.imwrite(os.path.join(debug_dir, f"pencil_{i}_{color_name}.jpg"), pencil_img)
    
    print(f"Found {len(pencils)} pencils in total")
    return pencils


def visualize_results(image: np.ndarray, 
                    green_pencils: List[Dict[str, Any]], 
                    all_pencils: List[Dict[str, Any]],
                    output_path: str = None) -> np.ndarray:
    """
    Visualize detection results by drawing bounding boxes and labels.
    
    Args:
        image: Original input image
        green_pencils: List of detected green pencils
        all_pencils: List of all detected pencils
        output_path: Path to save the visualization (optional)
        
    Returns:
        Annotated image
    """
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Define colors for visualization (BGR format)
    color_map = {
        "red": (0, 0, 255),
        "orange": (0, 165, 255),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
        "light_blue": (255, 255, 0),
        "blue": (255, 0, 0),
        "purple": (255, 0, 255),
        "pink": (203, 192, 255),
        "gray": (128, 128, 128)
    }
    
    # Group pencils by color
    pencils_by_color = {}
    for pencil in all_pencils:
        color = pencil['color_name']
        if color not in pencils_by_color:
            pencils_by_color[color] = []
        pencils_by_color[color].append(pencil)
    
    # Draw all pencils with their color classification
    for color_name, pencils in pencils_by_color.items():
        box_color = color_map.get(color_name, (255, 255, 255))  # Default to white if color not in map
        
        for i, pencil in enumerate(pencils):
            x, y, w, h = pencil['bbox']
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), box_color, 2)
            
            # Draw label with color name and index
            label = f"{color_name} {i+1}"
            cv2.putText(vis_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    
    # Highlight green pencils with thicker boxes
    for i, pencil in enumerate(green_pencils):
        x, y, w, h = pencil['bbox']
        
        # Draw thicker bounding box for green pencils
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 4)
        
        # Draw label with index
        label = f"GREEN {i+1}"
        cv2.putText(vis_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Add summary text
    summary_text = []
    summary_text.append(f"Total pencils: {len(all_pencils)}")
    summary_text.append(f"Green pencils: {len(green_pencils)}")
    
    for i, line in enumerate(summary_text):
        cv2.putText(vis_image, line, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save the visualization if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to {output_path}")
    
    return vis_image


def generate_results_json(green_pencils: List[Dict[str, Any]], 
                        all_pencils: List[Dict[str, Any]],
                        output_path: str = None) -> Dict[str, Any]:
    """
    Generate a JSON with detection results.
    
    Args:
        green_pencils: List of detected green pencils
        all_pencils: List of all detected pencils
        output_path: Path to save the JSON (optional)
        
    Returns:
        Results dictionary
    """
    # Group pencils by color
    pencils_by_color = {}
    for pencil in all_pencils:
        color = pencil['color_name']
        if color not in pencils_by_color:
            pencils_by_color[color] = []
            
        # Extract relevant information (exclude contour which is not JSON serializable)
        pencil_info = {
            'bbox': pencil['bbox'],
            'center': pencil['center'],
            'area': float(pencil['area']),  # Convert numpy types to native Python types
            'bgr_color': [int(c) for c in pencil['bgr_color']]
        }
        
        pencils_by_color[color].append(pencil_info)
    
    # Create results dictionary
    results = {
        'total_pencils': len(all_pencils),
        'pencils_by_color': {}
    }
    
    # Add count and positions for each color
    for color, pencils in pencils_by_color.items():
        results['pencils_by_color'][color] = {
            'count': len(pencils),
            'positions': [p['center'] for p in pencils]
        }
    
    # Add specific section for green pencils
    green_positions = [{'center': p['center'], 'bbox': p['bbox']} for p in green_pencils]
    results['green_pencils'] = {
        'count': len(green_pencils),
        'positions': green_positions
    }
    
    # Save to JSON file if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return results


def main():
    """Main function to run the pencil detection pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect and count pencils by color')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save output files')
    parser.add_argument('--show', action='store_true', help='Show visualization')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional visualizations')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create debug directory if debug mode is enabled
    if args.debug:
        debug_dir = os.path.join(os.getcwd(), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug mode enabled. Debug images will be saved to {debug_dir}")
    
    # Load image
    image = load_image(args.image)
    print(f"Loaded image with shape: {image.shape}")
    
    # Save original image for reference if in debug mode
    if args.debug:
        cv2.imwrite(os.path.join(debug_dir, "original.jpg"), image)
    
    # Detect green pencils (Task 1 & 2)
    green_pencils = detect_green_pencils(image, debug=args.debug)
    
    # Detect all pencils (Task 3)
    all_pencils = detect_all_pencils(image, debug=args.debug)
    
    # Generate visualization
    output_image_path = os.path.join(args.output_dir, 'pencils_detected.jpg')
    vis_image = visualize_results(image, green_pencils, all_pencils, output_image_path)
    
    # Generate JSON results
    output_json_path = os.path.join(args.output_dir, 'pencils_results.json')
    results = generate_results_json(green_pencils, all_pencils, output_json_path)
    
    # Print summary
    print("\n=== Pencil Detection Results ===")
    print(f"Total pencils found: {len(all_pencils)}")
    print(f"Green pencils found: {len(green_pencils)}")
    
    # Print green pencil positions
    if green_pencils:
        print("\nGreen pencil positions:")
        for i, pencil in enumerate(green_pencils):
            x, y, w, h = pencil['bbox']
            center_x, center_y = pencil['center']
            print(f"  Green pencil {i+1}: Center at ({center_x}, {center_y}), Bounding box: ({x}, {y}, {w}, {h})")
    
    # Print counts by color
    color_counts = {}
    for pencil in all_pencils:
        color = pencil['color_name']
        if color not in color_counts:
            color_counts[color] = 0
        color_counts[color] += 1
    
    if color_counts:
        print("\nPencil counts by color:")
        for color, count in sorted(color_counts.items()):
            print(f"  {color.capitalize()}: {count}")
    
    # Print HSV values for debugging color classification if in debug mode
    if args.debug:
        print("\nHSV values for color classification debugging:")
        for i, pencil in enumerate(all_pencils):
            h, s, v = pencil['hsv_color']
            color = pencil['color_name']
            print(f"  Pencil {i+1}: Color={color}, HSV=({h}, {s}, {v})")
    
    # Show visualization if requested
    if args.show:
        cv2.imshow('Pencil Detection', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
