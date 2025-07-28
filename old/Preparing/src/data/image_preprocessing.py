"""
Advanced image preprocessing functions for anomaly detection in computer vision.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Union, Optional, Any
import matplotlib.pyplot as plt
from skimage import exposure, filters, morphology, feature, segmentation, color


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                keep_aspect_ratio: bool = True, pad_color: int = 0) -> np.ndarray:
    """
    Resize an image to the target size with optional aspect ratio preservation.
    
    Args:
        image: Input image
        target_size: Target size as (height, width)
        keep_aspect_ratio: Whether to preserve aspect ratio
        pad_color: Padding color for preserved aspect ratio
        
    Returns:
        Resized image
    """
    if keep_aspect_ratio:
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scaling factor
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas with target size
        if len(image.shape) == 3:
            canvas = np.ones((target_h, target_w, image.shape[2]), dtype=image.dtype) * pad_color
        else:
            canvas = np.ones((target_h, target_w), dtype=image.dtype) * pad_color
            
        # Center image on canvas
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        if len(image.shape) == 3:
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized
        else:
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
        return canvas
    else:
        return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.
    
    Args:
        image: Input image (can be color or already grayscale)
        
    Returns:
        Grayscale image
    """
    if len(image.shape) == 2:
        return image  # Already grayscale
    elif len(image.shape) == 3 and image.shape[2] == 1:
        return image[:, :, 0]  # Single channel
    else:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def apply_blur(image: np.ndarray, method: str = 'gaussian', kernel_size: int = 5, 
              sigma: float = 1.0) -> np.ndarray:
    """
    Apply blur to an image using different methods.
    
    Args:
        image: Input image
        method: Blur method ('gaussian', 'median', 'bilateral')
        kernel_size: Size of the kernel
        sigma: Sigma parameter for Gaussian and bilateral blur
        
    Returns:
        Blurred image
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    elif method == 'median':
        return cv2.medianBlur(image, kernel_size)
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, kernel_size, sigma, sigma)
    else:
        raise ValueError(f"Unknown blur method: {method}")


def detect_edges(image: np.ndarray, method: str = 'canny', 
                threshold1: float = 100, threshold2: float = 200,
                aperture_size: int = 3) -> np.ndarray:
    """
    Detect edges in an image using various methods.
    
    Args:
        image: Input image
        method: Edge detection method ('canny', 'sobel', 'laplacian', 'scharr')
        threshold1: First threshold for Canny edge detection
        threshold2: Second threshold for Canny edge detection
        aperture_size: Aperture size for gradient operators
        
    Returns:
        Edge map
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Ensure uint8 format for some operations
    if gray.dtype != np.uint8:
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)
    
    if method == 'canny':
        return cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture_size)
    
    elif method == 'sobel':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=aperture_size)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=aperture_size)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        # Normalize to 0-255
        sobel = np.uint8(255 * sobel / np.max(sobel))
        return sobel
    
    elif method == 'laplacian':
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=aperture_size)
        # Normalize to 0-255
        laplacian = np.uint8(255 * np.absolute(laplacian) / np.max(np.absolute(laplacian)))
        return laplacian
    
    elif method == 'scharr':
        scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        scharr = np.sqrt(scharrx**2 + scharry**2)
        # Normalize to 0-255
        scharr = np.uint8(255 * scharr / np.max(scharr))
        return scharr
    
    else:
        raise ValueError(f"Unknown edge detection method: {method}")


def normalize_image(image: np.ndarray, method: str = 'minmax', 
                   target_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """
    Normalize image values using different methods.
    
    Args:
        image: Input image
        method: Normalization method ('minmax', 'zscore', 'percentile')
        target_range: Target range for the normalized values
        
    Returns:
        Normalized image
    """
    if method == 'minmax':
        # Min-max normalization
        min_val, max_val = np.min(image), np.max(image)
        if max_val == min_val:
            normalized = np.zeros_like(image, dtype=np.float32)
        else:
            normalized = (image - min_val) / (max_val - min_val)
        
        # Scale to target range
        if target_range != (0, 1):
            normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
            
        return normalized
    
    elif method == 'zscore':
        # Z-score normalization
        mean, std = np.mean(image), np.std(image)
        if std == 0:
            normalized = np.zeros_like(image, dtype=np.float32)
        else:
            normalized = (image - mean) / std
            
        # Scale to target range if needed
        if target_range != (0, 1):
            # First scale to [0, 1] (assuming typical z-scores are in [-3, 3])
            normalized = (normalized + 3) / 6
            # Then scale to target range
            normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
            # Clip to ensure values are within range
            normalized = np.clip(normalized, target_range[0], target_range[1])
            
        return normalized
    
    elif method == 'percentile':
        # Percentile-based normalization (robust to outliers)
        p_low, p_high = 2, 98
        low, high = np.percentile(image, p_low), np.percentile(image, p_high)
        if high == low:
            normalized = np.zeros_like(image, dtype=np.float32)
        else:
            normalized = np.clip((image - low) / (high - low), 0, 1)
            
        # Scale to target range
        if target_range != (0, 1):
            normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
            
        return normalized
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, 
               tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        CLAHE-enhanced image
    """
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Handle different image types
    if len(image.shape) == 2:
        # Grayscale image
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return clahe.apply(image)
    
    elif len(image.shape) == 3:
        # Color image
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge((l, a, b))
        
        # Convert back to RGB
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    else:
        raise ValueError("Unsupported image format")


def visualize_anomalies(original: np.ndarray, anomaly_map: np.ndarray, 
                       threshold: Optional[float] = None,
                       colormap: str = 'jet', alpha: float = 0.5,
                       figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Visualize anomalies in an image using an anomaly map.
    
    Args:
        original: Original image
        anomaly_map: Anomaly map (higher values indicate anomalies)
        threshold: Optional threshold for binary anomaly detection
        colormap: Colormap for the anomaly visualization
        alpha: Transparency of the anomaly overlay
        figsize: Figure size
    """
    # Normalize anomaly map to [0, 1]
    if anomaly_map.min() != anomaly_map.max():
        anomaly_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    else:
        anomaly_norm = np.zeros_like(anomaly_map)
    
    # Resize anomaly map to match original image if needed
    if anomaly_norm.shape[:2] != original.shape[:2]:
        anomaly_norm = cv2.resize(anomaly_norm, (original.shape[1], original.shape[0]))
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')
    
    # Plot anomaly heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(anomaly_norm, cmap=colormap)
    plt.title("Anomaly Map")
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Plot overlay
    plt.subplot(1, 3, 3)
    plt.imshow(original)
    plt.imshow(anomaly_norm, cmap=colormap, alpha=alpha)
    
    # If threshold is provided, highlight anomalies
    if threshold is not None:
        # Create binary mask
        mask = anomaly_norm > threshold
        # Highlight contours
        if np.any(mask):
            plt.contour(mask, colors='r', linewidths=2)
            plt.title(f"Overlay (threshold={threshold:.2f})")
        else:
            plt.title("Overlay (no anomalies)")
    else:
        plt.title("Overlay")
    
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def extract_features(image: np.ndarray, feature_type: str = 'hog',
                    visualize: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract features from an image.
    
    Args:
        image: Input image
        feature_type: Type of features ('hog', 'lbp', 'orb', 'sift')
        visualize: Whether to return visualization
        
    Returns:
        Features or tuple of (features, visualization)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Ensure proper format
    if gray.dtype != np.uint8:
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)
    
    if feature_type == 'hog':
        # Histogram of Oriented Gradients
        from skimage.feature import hog
        features, hog_image = hog(
            gray, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys'
        )
        
        if visualize:
            return features, hog_image
        else:
            return features
    
    elif feature_type == 'lbp':
        # Local Binary Patterns
        from skimage.feature import local_binary_pattern
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Compute histogram of LBP
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        
        if visualize:
            return hist, lbp
        else:
            return hist
    
    elif feature_type == 'orb':
        # Oriented FAST and Rotated BRIEF
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        if visualize:
            vis_image = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return descriptors, vis_image
        else:
            return descriptors if descriptors is not None else np.array([])
    
    elif feature_type == 'sift':
        # Scale-Invariant Feature Transform
        try:
            sift = cv2.SIFT_create()
        except AttributeError:
            # For older OpenCV versions
            sift = cv2.xfeatures2d.SIFT_create()
            
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if visualize:
            vis_image = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return descriptors, vis_image
        else:
            return descriptors if descriptors is not None else np.array([])
    
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def create_preprocessing_pipeline(steps: List[Dict[str, Any]]) -> callable:
    """
    Create a preprocessing pipeline from a list of steps.
    
    Args:
        steps: List of preprocessing steps, each a dict with 'function' and 'params'
        
    Returns:
        Function that applies the pipeline to an image
    """
    def pipeline(image: np.ndarray) -> np.ndarray:
        result = image.copy()
        
        for step in steps:
            function_name = step['function']
            params = step.get('params', {})
            
            # Get the function from this module
            if function_name == 'resize_image':
                result = resize_image(result, **params)
            elif function_name == 'convert_to_grayscale':
                result = convert_to_grayscale(result)
            elif function_name == 'apply_blur':
                result = apply_blur(result, **params)
            elif function_name == 'detect_edges':
                result = detect_edges(result, **params)
            elif function_name == 'normalize_image':
                result = normalize_image(result, **params)
            elif function_name == 'apply_clahe':
                result = apply_clahe(result, **params)
            else:
                raise ValueError(f"Unknown function: {function_name}")
                
        return result
    
    return pipeline


def visualize_preprocessing_steps(image: np.ndarray, steps: List[Dict[str, Any]],
                                figsize: Tuple[int, int] = (15, 10)) -> np.ndarray:
    """
    Visualize the steps in a preprocessing pipeline.
    
    Args:
        image: Input image
        steps: List of preprocessing steps
        figsize: Figure size
        
    Returns:
        Final processed image
    """
    n_steps = len(steps) + 1  # Original + each step
    
    # Create figure
    fig, axes = plt.subplots(1, n_steps, figsize=figsize)
    if n_steps == 1:
        axes = [axes]
    
    # Display original image
    axes[0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Apply each step and display result
    result = image.copy()
    
    for i, step in enumerate(steps):
        function_name = step['function']
        params = step.get('params', {})
        
        # Apply step
        if function_name == 'resize_image':
            result = resize_image(result, **params)
        elif function_name == 'convert_to_grayscale':
            result = convert_to_grayscale(result)
        elif function_name == 'apply_blur':
            result = apply_blur(result, **params)
        elif function_name == 'detect_edges':
            result = detect_edges(result, **params)
        elif function_name == 'normalize_image':
            result = normalize_image(result, **params)
        elif function_name == 'apply_clahe':
            result = apply_clahe(result, **params)
        else:
            raise ValueError(f"Unknown function: {function_name}")
        
        # Display result
        if len(result.shape) == 2 or (len(result.shape) == 3 and result.shape[2] == 1):
            axes[i+1].imshow(result, cmap='gray')
        else:
            axes[i+1].imshow(result)
            
        axes[i+1].set_title(f"Step {i+1}: {function_name}")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return result
