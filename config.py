"""
Configuration parameters for pencil detection.
"""
import numpy as np
from typing import Dict, Tuple

# HSV color ranges for different colors
COLOR_RANGES = {
    'green': {
        'lower': np.array([35, 50, 50]),
        'upper': np.array([85, 255, 255])
    },
    'red': {
        # Il rosso in HSV è diviso in due intervalli (vicino a 0° e vicino a 180°)
        'lower1': np.array([0, 170, 120]),
        'upper1': np.array([2, 255, 255]),
        'lower2': np.array([178, 170, 120]),
        'upper2': np.array([180, 255, 255])
    },
    'blue': {
        'lower': np.array([105, 100, 100]),
        'upper': np.array([115, 255, 255])
    },
    'yellow': {
        'lower': np.array([20, 100, 100]),
        'upper': np.array([30, 255, 255])
    },
    'purple': {
        'lower': np.array([130, 20, 20]),
        'upper': np.array([160, 255, 255])
    },
    'orange': {
        'lower': np.array([3, 150, 150]),
        'upper': np.array([15, 255, 255])
    },
    'pink': {
        # Includiamo sia il rosa tradizionale che il rosa carne
        'lower1': np.array([161, 40, 100]),
        'upper1': np.array([170, 150, 255]),
        'lower2': np.array([5, 50, 150]),
        'upper2': np.array([15, 150, 255])
    },
    'brown': {
        'lower': np.array([5, 50, 50]),
        'upper': np.array([20, 150, 150])
    },
    'light_blue': {
        'lower': np.array([85, 80, 120]),
        'upper': np.array([100, 255, 255])
    },
    'black': {
        'lower': np.array([0, 0, 0]),
        'upper': np.array([180, 50, 60])
    }
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
    'green': (0, 255, 0),     # Green
    'red': (0, 0, 255),       # Red
    'blue': (255, 0, 0),      # Blue
    'yellow': (0, 255, 255),  # Yellow
    'orange': (0, 165, 255),  # Orange
    'purple': (255, 0, 255),  # Purple
    'pink': (203, 192, 255),  # Pink
    'brown': (42, 42, 165),   # Brown
    'black': (0, 0, 0),       # Black
    'gray': (128, 128, 128),  # Gray
    'white': (255, 255, 255), # White
    'light_blue': (255, 191, 0) # Light Blue
}
