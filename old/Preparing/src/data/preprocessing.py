"""
Image preprocessing utilities for computer vision and anomaly detection.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Union, Optional, Any
from skimage import exposure, filters, morphology, feature, segmentation
import matplotlib.pyplot as plt


def load_image(image_path: str, grayscale: bool = False) -> np.ndarray:
    """
    Load an image from a file path.
    
    Args:
        image_path: Path to the image file
        grayscale: Whether to load as grayscale
        
    Returns:
        Loaded image as numpy array
    """
    if grayscale:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
        
    return img


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                 keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize an image to the target size.
    
    Args:
        image: Input image
        target_size: Target size as (height, width)
        keep_aspect_ratio: Whether to preserve aspect ratio
        
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
            canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        else:
            canvas = np.zeros((target_h, target_w), dtype=image.dtype)
            
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


def normalize_image(image: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize image values.
    
    Args:
        image: Input image
        method: Normalization method ('minmax', 'zscore', 'clahe')
        
    Returns:
        Normalized image
    """
    if method == 'minmax':
        # Scale to [0, 1]
        return (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    elif method == 'zscore':
        # Standardize to zero mean and unit variance
        mean = image.mean()
        std = image.std() + 1e-8
        return (image - mean) / std
    
    elif method == 'clahe':
        # Contrast Limited Adaptive Histogram Equalization
        if len(image.shape) == 3:
            # For color images, apply CLAHE to L channel in LAB color space
            lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(np.uint8(l * 255)) / 255.0
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(np.uint8(image * 255)) / 255.0
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def extract_edges(image: np.ndarray, method: str = 'canny', 
                 threshold1: float = 100, threshold2: float = 200) -> np.ndarray:
    """
    Extract edges from an image.
    
    Args:
        image: Input image
        method: Edge detection method ('canny', 'sobel', 'prewitt', 'scharr')
        threshold1, threshold2: Thresholds for Canny edge detection
        
    Returns:
        Edge map
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    if method == 'canny':
        return cv2.Canny(np.uint8(gray * 255), threshold1, threshold2)
    
    elif method == 'sobel':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx**2 + sobely**2)
    
    elif method == 'prewitt':
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewittx = cv2.filter2D(gray, -1, kernelx)
        prewitty = cv2.filter2D(gray, -1, kernely)
        return np.sqrt(prewittx**2 + prewitty**2)
    
    elif method == 'scharr':
        scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        return np.sqrt(scharrx**2 + scharry**2)
    
    else:
        raise ValueError(f"Unknown edge detection method: {method}")


def segment_image(image: np.ndarray, method: str = 'otsu') -> np.ndarray:
    """
    Segment an image using various methods.
    
    Args:
        image: Input image
        method: Segmentation method ('otsu', 'kmeans', 'watershed')
        
    Returns:
        Segmented image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    if method == 'otsu':
        # Otsu's thresholding
        _, thresh = cv2.threshold(np.uint8(gray * 255), 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    elif method == 'kmeans':
        # K-means clustering
        if len(image.shape) == 3:
            # For color images
            pixels = image.reshape(-1, 3).astype(np.float32)
        else:
            # For grayscale images
            pixels = gray.reshape(-1, 1).astype(np.float32)
            
        # Define criteria and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 2  # Number of clusters
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8 and reshape
        segmented = centers[labels.flatten()].reshape(image.shape)
        return segmented
    
    elif method == 'watershed':
        # Watershed algorithm
        # Apply threshold
        _, thresh = cv2.threshold(np.uint8(gray * 255), 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that background is 1, not 0
        markers = markers + 1
        
        # Mark the unknown region with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        if len(image.shape) == 3:
            markers = cv2.watershed(np.uint8(image * 255), markers)
        else:
            # Convert to BGR for watershed
            img_color = cv2.cvtColor(np.uint8(gray * 255), cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(img_color, markers)
        
        # Mark boundaries in the original image
        result = np.zeros_like(gray)
        result[markers == -1] = 1  # Boundaries
        
        return result
    
    else:
        raise ValueError(f"Unknown segmentation method: {method}")


def extract_features(image: np.ndarray, feature_type: str = 'hog') -> np.ndarray:
    """
    Extract features from an image.
    
    Args:
        image: Input image
        feature_type: Type of features to extract ('hog', 'lbp', 'orb')
        
    Returns:
        Extracted features
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    if feature_type == 'hog':
        # Histogram of Oriented Gradients
        fd, hog_image = feature.hog(
            gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
            block_norm='L2-Hys', visualize=True
        )
        return hog_image
    
    elif feature_type == 'lbp':
        # Local Binary Patterns
        radius = 3
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
        return lbp
    
    elif feature_type == 'orb':
        # ORB (Oriented FAST and Rotated BRIEF)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(np.uint8(gray * 255), None)
        
        # Draw keypoints
        orb_image = cv2.drawKeypoints(
            np.uint8(gray * 255), keypoints, None, color=(0, 255, 0), 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return orb_image
    
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def remove_noise(image: np.ndarray, method: str = 'gaussian', kernel_size: int = 5) -> np.ndarray:
    """
    Remove noise from an image.
    
    Args:
        image: Input image
        method: Noise removal method ('gaussian', 'median', 'bilateral')
        kernel_size: Size of the kernel for filtering
        
    Returns:
        Denoised image
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif method == 'median':
        return cv2.medianBlur(np.uint8(image * 255), kernel_size) / 255.0
    
    elif method == 'bilateral':
        # Bilateral filter preserves edges while removing noise
        return cv2.bilateralFilter(np.uint8(image * 255), kernel_size, 75, 75) / 255.0
    
    else:
        raise ValueError(f"Unknown noise removal method: {method}")


def enhance_contrast(image: np.ndarray, method: str = 'clahe') -> np.ndarray:
    """
    Enhance contrast in an image.
    
    Args:
        image: Input image
        method: Contrast enhancement method ('clahe', 'histeq', 'gamma')
        
    Returns:
        Contrast-enhanced image
    """
    if method == 'clahe':
        # Contrast Limited Adaptive Histogram Equalization
        if len(image.shape) == 3:
            # For color images, apply CLAHE to L channel in LAB color space
            lab = cv2.cvtColor(np.uint8(image * 255), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) / 255.0
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(np.uint8(image * 255)) / 255.0
    
    elif method == 'histeq':
        # Histogram equalization
        if len(image.shape) == 3:
            # For color images, apply histogram equalization to each channel
            result = np.zeros_like(image, dtype=np.float32)
            for i in range(3):
                result[:, :, i] = cv2.equalizeHist(np.uint8(image[:, :, i] * 255)) / 255.0
            return result
        else:
            # For grayscale images
            return cv2.equalizeHist(np.uint8(image * 255)) / 255.0
    
    elif method == 'gamma':
        # Gamma correction
        gamma = 1.5  # Adjust gamma value as needed
        return np.power(image, 1/gamma)
    
    else:
        raise ValueError(f"Unknown contrast enhancement method: {method}")


def create_preprocessing_pipeline(operations: List[Dict[str, Any]]) -> callable:
    """
    Create a preprocessing pipeline from a list of operations.
    
    Args:
        operations: List of dictionaries, each containing:
                   - 'function': The preprocessing function to apply
                   - 'params': Parameters for the function
                   
    Returns:
        A function that applies the pipeline to an image
    """
    def pipeline(image: np.ndarray) -> np.ndarray:
        result = image.copy()
        
        for op in operations:
            func = op['function']
            params = op.get('params', {})
            result = func(result, **params)
            
        return result
    
    return pipeline


def visualize_preprocessing_steps(image: np.ndarray, operations: List[Dict[str, Any]], 
                                 figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Visualize the steps in a preprocessing pipeline.
    
    Args:
        image: Input image
        operations: List of preprocessing operations
        figsize: Figure size
    """
    n_steps = len(operations) + 1  # Original image + each step
    fig, axes = plt.subplots(1, n_steps, figsize=figsize)
    
    # Display original image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Apply each operation and display result
    result = image.copy()
    for i, op in enumerate(operations):
        func = op['function']
        params = op.get('params', {})
        result = func(result, **params)
        
        # Display result
        if len(result.shape) == 2 or result.shape[2] == 1:
            # Grayscale image
            axes[i+1].imshow(result, cmap='gray')
        else:
            # Color image
            axes[i+1].imshow(result)
            
        # Set title
        title = op.get('name', func.__name__)
        axes[i+1].set_title(title)
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return result  # Return the final processed image
