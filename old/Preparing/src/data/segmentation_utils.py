"""
Traditional image segmentation utilities using OpenCV.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union
from skimage import segmentation, color, morphology, measure
from scipy import ndimage


def threshold_segmentation(image: np.ndarray, 
                         method: str = 'otsu',
                         threshold: Optional[int] = None) -> np.ndarray:
    """
    Segment image using thresholding.
    
    Args:
        image: Input image
        method: Thresholding method ('otsu', 'adaptive', 'binary')
        threshold: Manual threshold value (only for 'binary')
        
    Returns:
        Binary mask
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply thresholding
    if method == 'otsu':
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    elif method == 'adaptive':
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    elif method == 'binary':
        if threshold is None:
            threshold = 127
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return mask


def kmeans_segmentation(image: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment image using K-means clustering.
    
    Args:
        image: Input image
        k: Number of clusters
        
    Returns:
        Tuple of (segmented image, labels)
    """
    # Reshape image for clustering
    pixel_values = image.reshape((-1, 3)).astype(np.float32)
    
    # Define criteria and apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    
    # Reshape labels to match image dimensions
    labels = labels.reshape(image.shape[:2])
    
    return segmented_image, labels


def watershed_segmentation(image: np.ndarray, 
                         marker_threshold: int = 127) -> np.ndarray:
    """
    Segment image using watershed algorithm.
    
    Args:
        image: Input image
        marker_threshold: Threshold for marker generation
        
    Returns:
        Segmented image with labels
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(gray, marker_threshold, 255, cv2.THRESH_BINARY)
    
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add 1 to all labels so that background is not 0, but 1
    markers = markers + 1
    
    # Mark the unknown region with 0
    markers[unknown == 255] = 0
    
    # Apply watershed
    if len(image.shape) == 3:
        markers = cv2.watershed(image, markers)
    else:
        # Convert to BGR for watershed
        image_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(image_color, markers)
    
    return markers


def grab_cut_segmentation(image: np.ndarray, 
                        rect: Optional[Tuple[int, int, int, int]] = None,
                        iterations: int = 5) -> np.ndarray:
    """
    Segment image using GrabCut algorithm.
    
    Args:
        image: Input image
        rect: Rectangle containing the object (x, y, width, height)
        iterations: Number of iterations
        
    Returns:
        Binary mask
    """
    # Create mask
    mask = np.zeros(image.shape[:2], np.uint8)
    
    # Create temporary arrays for the algorithm
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Define rectangle if not provided
    if rect is None:
        h, w = image.shape[:2]
        rect = (w // 10, h // 10, w * 8 // 10, h * 8 // 10)
    
    # Apply GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    
    # Modify mask: 0 and 2 for background, 1 and 3 for foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    return mask2 * 255


def felzenszwalb_segmentation(image: np.ndarray, 
                            scale: float = 100,
                            sigma: float = 0.5,
                            min_size: int = 50) -> np.ndarray:
    """
    Segment image using Felzenszwalb's algorithm.
    
    Args:
        image: Input image
        scale: Free parameter. Higher means larger clusters.
        sigma: Width of Gaussian kernel for preprocessing
        min_size: Minimum component size
        
    Returns:
        Segmented image with labels
    """
    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if BGR (OpenCV default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Apply segmentation
    segments = segmentation.felzenszwalb(
        image_rgb, scale=scale, sigma=sigma, min_size=min_size
    )
    
    return segments


def slic_segmentation(image: np.ndarray, 
                    n_segments: int = 100,
                    compactness: float = 10.0) -> np.ndarray:
    """
    Segment image using SLIC superpixels.
    
    Args:
        image: Input image
        n_segments: Approximate number of segments
        compactness: Balances color and space proximity
        
    Returns:
        Segmented image with labels
    """
    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if BGR (OpenCV default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Apply segmentation
    segments = segmentation.slic(
        image_rgb, n_segments=n_segments, compactness=compactness
    )
    
    return segments


def quickshift_segmentation(image: np.ndarray, 
                          kernel_size: float = 3,
                          max_dist: float = 6,
                          ratio: float = 0.5) -> np.ndarray:
    """
    Segment image using quickshift clustering.
    
    Args:
        image: Input image
        kernel_size: Width of Gaussian kernel
        max_dist: Cut-off point for data distances
        ratio: Balances color and space proximity
        
    Returns:
        Segmented image with labels
    """
    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if BGR (OpenCV default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Apply segmentation
    segments = segmentation.quickshift(
        image_rgb, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio
    )
    
    return segments


def morphological_segmentation(binary_image: np.ndarray,
                             operation: str = 'open',
                             kernel_size: int = 5,
                             iterations: int = 1) -> np.ndarray:
    """
    Apply morphological operations for segmentation refinement.
    
    Args:
        binary_image: Binary input image
        operation: Morphological operation ('open', 'close', 'dilate', 'erode')
        kernel_size: Size of the structuring element
        iterations: Number of times to apply the operation
        
    Returns:
        Processed binary image
    """
    # Create kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply operation
    if operation == 'open':
        result = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'close':
        result = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == 'dilate':
        result = cv2.dilate(binary_image, kernel, iterations=iterations)
    elif operation == 'erode':
        result = cv2.erode(binary_image, kernel, iterations=iterations)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return result


def connected_component_analysis(binary_image: np.ndarray,
                               min_size: int = 100,
                               connectivity: int = 8) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Perform connected component analysis on binary image.
    
    Args:
        binary_image: Binary input image
        min_size: Minimum component size to keep
        connectivity: Connectivity for connected components (4 or 8)
        
    Returns:
        Tuple of (labeled image, list of component properties)
    """
    # Ensure binary image
    if binary_image.dtype != np.uint8:
        binary_image = binary_image.astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=connectivity
    )
    
    # Filter small components
    filtered_labels = np.zeros_like(labels)
    component_props = []
    
    # Start from 1 to skip background (label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area >= min_size:
            # Keep this component
            filtered_labels[labels == i] = i
            
            # Extract properties
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]
            
            component_props.append({
                'label': i,
                'area': area,
                'centroid': (cx, cy),
                'bounding_box': (x, y, w, h),
                'width': w,
                'height': h
            })
    
    return filtered_labels, component_props


def visualize_segmentation(image: np.ndarray, 
                         segmentation: np.ndarray,
                         alpha: float = 0.5,
                         random_colors: bool = True,
                         figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualize segmentation results.
    
    Args:
        image: Original image
        segmentation: Segmentation mask or labels
        alpha: Transparency for overlay
        random_colors: Whether to use random colors for labels
        figsize: Figure size
    """
    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if BGR (OpenCV default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    # Plot segmentation
    plt.subplot(1, 3, 2)
    
    if segmentation.max() == 1 or segmentation.max() == 255:
        # Binary segmentation
        plt.imshow(segmentation, cmap='gray')
        plt.title("Segmentation Mask")
    else:
        # Multi-label segmentation
        if random_colors:
            # Use random colors for visualization
            segmentation_rgb = color.label2rgb(segmentation, image=image_rgb, bg_label=0)
            plt.imshow(segmentation_rgb)
        else:
            plt.imshow(segmentation, cmap='nipy_spectral')
        plt.title(f"Segmentation ({segmentation.max()} regions)")
    
    plt.axis('off')
    
    # Plot overlay
    plt.subplot(1, 3, 3)
    
    if segmentation.max() == 1 or segmentation.max() == 255:
        # Binary segmentation
        mask = segmentation.astype(bool)
        masked_image = np.copy(image_rgb)
        
        # Create colored overlay
        if len(image_rgb.shape) == 3:
            masked_image[mask] = (0, 255, 0)  # Green for the segmented region
        else:
            masked_image[mask] = 255
        
        # Blend with original image
        overlay = cv2.addWeighted(image_rgb, 1 - alpha, masked_image, alpha, 0)
        plt.imshow(overlay)
        plt.title("Segmentation Overlay")
    else:
        # Multi-label segmentation
        if random_colors:
            # Use random colors for visualization
            boundaries = segmentation.astype(np.uint8)
            boundaries = segmentation.copy()
            
            # Find boundaries between regions
            for i in range(1, boundaries.shape[0]):
                for j in range(1, boundaries.shape[1]):
                    if boundaries[i, j] != boundaries[i-1, j] or boundaries[i, j] != boundaries[i, j-1]:
                        boundaries[i, j] = 0
            
            # Create overlay
            overlay = np.copy(image_rgb)
            if len(image_rgb.shape) == 3:
                for i in range(1, segmentation.max() + 1):
                    mask = segmentation == i
                    overlay[mask] = [0, 255, 0]  # Green for all segments
            
            # Blend with original image
            overlay = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)
            plt.imshow(overlay)
            plt.title("Segmentation Overlay")
        else:
            # Use label2rgb for overlay
            overlay = color.label2rgb(segmentation, image=image_rgb, bg_label=0, alpha=alpha)
            plt.imshow(overlay)
            plt.title("Segmentation Overlay")
    
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def create_segmentation_pipeline(steps: List[Dict[str, Any]]) -> callable:
    """
    Create a segmentation pipeline from a list of steps.
    
    Args:
        steps: List of segmentation steps, each a dict with 'function' and 'params'
        
    Returns:
        Function that applies the pipeline to an image
    """
    def pipeline(image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Apply segmentation pipeline to an image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (final segmentation, list of intermediate results)
        """
        result = image.copy()
        intermediate_results = []
        
        for i, step in enumerate(steps):
            function_name = step['function']
            params = step.get('params', {})
            
            # Apply function
            if function_name == 'threshold_segmentation':
                result = threshold_segmentation(result, **params)
            elif function_name == 'kmeans_segmentation':
                result, _ = kmeans_segmentation(result, **params)
            elif function_name == 'watershed_segmentation':
                result = watershed_segmentation(result, **params)
            elif function_name == 'grab_cut_segmentation':
                result = grab_cut_segmentation(result, **params)
            elif function_name == 'felzenszwalb_segmentation':
                result = felzenszwalb_segmentation(result, **params)
            elif function_name == 'slic_segmentation':
                result = slic_segmentation(result, **params)
            elif function_name == 'quickshift_segmentation':
                result = quickshift_segmentation(result, **params)
            elif function_name == 'morphological_segmentation':
                result = morphological_segmentation(result, **params)
            elif function_name == 'connected_component_analysis':
                result, props = connected_component_analysis(result, **params)
                intermediate_results.append({
                    'step': i,
                    'name': function_name,
                    'result': result.copy(),
                    'properties': props
                })
                continue
            else:
                raise ValueError(f"Unknown function: {function_name}")
            
            # Store intermediate result
            intermediate_results.append({
                'step': i,
                'name': function_name,
                'result': result.copy()
            })
        
        return result, intermediate_results
    
    return pipeline


def visualize_segmentation_pipeline(image: np.ndarray, 
                                 steps: List[Dict[str, Any]],
                                 figsize: Tuple[int, int] = (15, 10)) -> np.ndarray:
    """
    Visualize the steps in a segmentation pipeline.
    
    Args:
        image: Input image
        steps: List of segmentation steps
        figsize: Figure size
        
    Returns:
        Final segmentation result
    """
    # Create pipeline
    pipeline_func = create_segmentation_pipeline(steps)
    
    # Apply pipeline
    result, intermediate_results = pipeline_func(image)
    
    # Create figure
    n_steps = len(intermediate_results) + 1  # Original + each step
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(n_steps)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    # Display original image
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if BGR (OpenCV default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(image_rgb)
    else:
        axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Display intermediate results
    for i, step_result in enumerate(intermediate_results):
        step_name = step_result['name']
        step_image = step_result['result']
        
        if step_image.max() == 1 or step_image.max() == 255:
            # Binary segmentation
            axes[i+1].imshow(step_image, cmap='gray')
        else:
            # Multi-label segmentation
            axes[i+1].imshow(color.label2rgb(step_image, bg_label=0))
            
        axes[i+1].set_title(f"Step {i+1}: {step_name}")
        axes[i+1].axis('off')
    
    # Hide unused subplots
    for i in range(n_steps, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return result
