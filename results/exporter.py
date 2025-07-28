"""
Module for exporting detection results to JSON format.
"""
import os
import json
import numpy as np
from typing import Dict, List, Any

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def prepare_pencil_for_json(pencil: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a pencil dictionary for JSON serialization by removing non-serializable items.
    
    Args:
        pencil: Pencil information dictionary
        
    Returns:
        JSON-serializable pencil dictionary
    """
    # Create a copy to avoid modifying the original
    result = pencil.copy()
    
    # Remove contour as it's not needed in the JSON output
    if 'contour' in result:
        del result['contour']
    
    return result

def save_json(results: Dict[str, Any], output_path: str) -> None:
    """
    Save detection results to a JSON file.
    
    Args:
        results: Dictionary of detection results
        output_path: Path to save the JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write JSON file
    with open(output_path, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)
    
    print(f"Results saved to {output_path}")

def generate_results_json(green_pencils: List[Dict[str, Any]], all_pencils: List[Dict[str, Any]], output_path: str) -> Dict[str, Any]:
    """
    Generate and save JSON results from pencil detection.
    
    Args:
        green_pencils: List of detected green pencils
        all_pencils: List of all detected pencils
        output_path: Path to save the JSON file
        
    Returns:
        Dictionary of results that was saved
    """
    # Prepare pencils for JSON serialization
    green_pencils_json = [prepare_pencil_for_json(p) for p in green_pencils]
    all_pencils_json = [prepare_pencil_for_json(p) for p in all_pencils]
    
    # Group pencils by color
    pencils_by_color = {}
    for pencil in all_pencils_json:
        color = pencil['color_name']
        if color not in pencils_by_color:
            pencils_by_color[color] = []
        pencils_by_color[color].append(pencil)
    
    # Create results dictionary
    results = {
        'green_pencils': {
            'count': len(green_pencils_json),
            'items': green_pencils_json
        },
        'all_pencils': {
            'count': len(all_pencils_json),
            'items': all_pencils_json
        },
        'pencils_by_color': {
            color: {
                'count': len(pencils),
                'items': pencils
            } for color, pencils in pencils_by_color.items()
        }
    }
    
    # Save to JSON file
    save_json(results, output_path)
    
    return results
