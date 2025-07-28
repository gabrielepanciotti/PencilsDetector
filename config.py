"""
Configuration parameters for pencil detection.
"""
import numpy as np
from typing import Dict, Tuple

# HSV color ranges for different colors
COLOR_RANGES = {
    'red_lower': (np.array([0, 100, 100]), np.array([10, 255, 255])),
    'red_upper': (np.array([160, 100, 100]), np.array([180, 255, 255])),
    'green': (np.array([35, 40, 40]), np.array([85, 255, 255])),
    'blue': (np.array([90, 50, 50]), np.array([130, 255, 255])),
    'yellow': (np.array([20, 100, 100]), np.array([35, 255, 255])),
    'orange': (np.array([10, 100, 100]), np.array([25, 255, 255])),
    'purple': (np.array([130, 50, 50]), np.array([160, 255, 255])),
    'pink': (np.array([145, 30, 150]), np.array([165, 120, 255])),
    'brown': (np.array([10, 50, 20]), np.array([30, 255, 200])),
}

# Contour filtering parameters
CONTOUR_PARAMS = {
    'min_area': 500,
    'max_area': 100000,
    'min_aspect_ratio': 2.5
}

# Morphological operation parameters
MORPH_PARAMS = {
    'kernel_size': 9,
    'iterations': 2
}

# Debug parameters
DEBUG = {
    'save_masks': True,
    'save_contours': True,
    'save_pencil_crops': True
}

# Visualization parameters
VIS_PARAMS = {
    'line_thickness': 2,
    'font_scale': 0.5,
    'font_thickness': 2,
    'font': 0  # FONT_HERSHEY_SIMPLEX
}

# Color mapping for visualization (BGR format)
COLOR_MAP = {
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
