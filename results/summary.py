"""
Module for generating summary reports of pencil detection results.
"""
from typing import Dict, List, Any

def print_detection_summary(green_pencils: List[Dict[str, Any]], all_pencils: List[Dict[str, Any]]) -> None:
    """
    Print a summary of the pencil detection results.
    
    Args:
        green_pencils: List of detected green pencils
        all_pencils: List of all detected pencils
    """
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

def print_debug_info(all_pencils: List[Dict[str, Any]]) -> None:
    """
    Print debug information for detected pencils.
    
    Args:
        all_pencils: List of all detected pencils
    """
    print("\nHSV values for color classification debugging:")
    for i, pencil in enumerate(all_pencils):
        h, s, v = pencil['hsv_color']
        color = pencil['color_name']
        print(f"  Pencil {i+1}: Color={color}, HSV=({h}, {s}, {v})")

def group_pencils_by_color(pencils: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group pencils by their classified color.
    
    Args:
        pencils: List of pencil dictionaries
        
    Returns:
        Dictionary mapping color names to lists of pencils
    """
    result = {}
    for pencil in pencils:
        color = pencil['color_name']
        if color not in result:
            result[color] = []
        result[color].append(pencil)
    
    return result
