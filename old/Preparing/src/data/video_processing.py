"""
Video processing and time-series analysis for anomaly detection in computer vision.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import os
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import pandas as pd


def extract_frames(video_path: str, output_dir: Optional[str] = None, 
                 frame_interval: int = 1, max_frames: Optional[int] = None,
                 resize: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames (None to not save)
        frame_interval: Extract every nth frame
        max_frames: Maximum number of frames to extract (None for all)
        resize: Optional size to resize frames to (width, height)
        
    Returns:
        List of extracted frames as numpy arrays
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"FPS: {fps:.2f}, Frames: {frame_count}, Duration: {duration:.2f}s")
    
    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames
    frames = []
    frame_idx = 0
    extracted_count = 0
    
    with tqdm(total=frame_count if max_frames is None else min(frame_count, max_frames * frame_interval),
             desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
                
            pbar.update(1)
            
            # Process every nth frame
            if frame_idx % frame_interval == 0:
                # Resize if needed
                if resize is not None:
                    frame = cv2.resize(frame, resize)
                
                # Save frame if output directory is provided
                if output_dir is not None:
                    frame_path = os.path.join(output_dir, f"frame_{extracted_count:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                
                # Add to list
                frames.append(frame)
                extracted_count += 1
                
                # Check if we've reached the maximum number of frames
                if max_frames is not None and extracted_count >= max_frames:
                    break
            
            frame_idx += 1
    
    # Release the video capture object
    cap.release()
    
    print(f"Extracted {len(frames)} frames")
    return frames


def extract_color_features(frame: np.ndarray, bins: int = 32) -> np.ndarray:
    """
    Extract color histogram features from a frame.
    
    Args:
        frame: Input frame
        bins: Number of bins for the histogram
        
    Returns:
        Color histogram features
    """
    # Convert to RGB if needed
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        # Check if BGR (OpenCV default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    
    # Split into channels
    channels = cv2.split(frame_rgb)
    
    # Compute histograms for each channel
    hist_features = []
    for i, channel in enumerate(channels):
        hist = cv2.calcHist([channel], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)
    
    return np.array(hist_features)


def extract_motion_features(frames: List[np.ndarray], 
                          method: str = 'flow',
                          step: int = 1) -> np.ndarray:
    """
    Extract motion features from a sequence of frames.
    
    Args:
        frames: List of frames
        method: Method to use ('flow' for optical flow, 'diff' for frame differencing)
        step: Step size between frames to analyze
        
    Returns:
        Array of motion features for each frame pair
    """
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames to extract motion features")
    
    # Convert frames to grayscale
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    
    features = []
    
    if method == 'flow':
        # Optical flow parameters
        flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        for i in range(0, len(gray_frames) - step, 1):
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i], gray_frames[i + step],
                None, **flow_params
            )
            
            # Calculate magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Calculate statistics of the flow
            mean_magnitude = np.mean(magnitude)
            std_magnitude = np.std(magnitude)
            max_magnitude = np.max(magnitude)
            
            # Calculate flow in different regions (3x3 grid)
            h, w = magnitude.shape
            region_features = []
            
            for y in range(3):
                for x in range(3):
                    y_start, y_end = int(y * h / 3), int((y + 1) * h / 3)
                    x_start, x_end = int(x * w / 3), int((x + 1) * w / 3)
                    
                    region_mag = magnitude[y_start:y_end, x_start:x_end]
                    region_features.extend([
                        np.mean(region_mag),
                        np.std(region_mag),
                        np.max(region_mag)
                    ])
            
            # Combine features
            frame_features = np.array([
                mean_magnitude, std_magnitude, max_magnitude,
                *region_features
            ])
            
            features.append(frame_features)
    
    elif method == 'diff':
        for i in range(0, len(gray_frames) - step, 1):
            # Calculate absolute difference
            diff = cv2.absdiff(gray_frames[i], gray_frames[i + step])
            
            # Calculate statistics
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            max_diff = np.max(diff)
            
            # Calculate difference in different regions (3x3 grid)
            h, w = diff.shape
            region_features = []
            
            for y in range(3):
                for x in range(3):
                    y_start, y_end = int(y * h / 3), int((y + 1) * h / 3)
                    x_start, x_end = int(x * w / 3), int((x + 1) * w / 3)
                    
                    region_diff = diff[y_start:y_end, x_start:x_end]
                    region_features.extend([
                        np.mean(region_diff),
                        np.std(region_diff),
                        np.max(region_diff)
                    ])
            
            # Combine features
            frame_features = np.array([
                mean_diff, std_diff, max_diff,
                *region_features
            ])
            
            features.append(frame_features)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return np.array(features)


def extract_keypoint_features(frame: np.ndarray, 
                            method: str = 'sift',
                            max_keypoints: int = 100) -> np.ndarray:
    """
    Extract keypoint features from a frame.
    
    Args:
        frame: Input frame
        method: Method to use ('sift', 'orb', 'surf')
        max_keypoints: Maximum number of keypoints to use
        
    Returns:
        Keypoint features
    """
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    if method == 'sift':
        try:
            # SIFT detector
            detector = cv2.SIFT_create()
        except AttributeError:
            # For older OpenCV versions
            detector = cv2.xfeatures2d.SIFT_create()
    
    elif method == 'orb':
        # ORB detector
        detector = cv2.ORB_create(nfeatures=max_keypoints)
    
    elif method == 'surf':
        try:
            # SURF detector
            detector = cv2.xfeatures2d.SURF_create()
        except AttributeError:
            raise ValueError("SURF is not available in this OpenCV version")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
    # If no keypoints found, return empty array
    if descriptors is None:
        return np.array([])
    
    # Limit number of keypoints
    if len(keypoints) > max_keypoints:
        # Sort keypoints by response (strength)
        keypoints_with_response = [(kp.response, i) for i, kp in enumerate(keypoints)]
        keypoints_with_response.sort(reverse=True)
        
        # Get indices of top keypoints
        top_indices = [idx for _, idx in keypoints_with_response[:max_keypoints]]
        descriptors = descriptors[top_indices]
    
    # Compute statistics of descriptors
    if len(descriptors) > 0:
        mean_desc = np.mean(descriptors, axis=0)
        std_desc = np.std(descriptors, axis=0)
        
        # Combine statistics
        features = np.concatenate([mean_desc, std_desc])
    else:
        # No descriptors, return zeros
        features = np.zeros(128 * 2)  # SIFT has 128-dim descriptors
    
    return features


def extract_area_features(frame: np.ndarray, 
                        threshold: int = 127,
                        min_area: int = 100) -> List[Dict[str, Any]]:
    """
    Extract features of connected areas in a frame.
    
    Args:
        frame: Input frame
        threshold: Threshold for binarization
        min_area: Minimum area size to consider
        
    Returns:
        List of area features (area, perimeter, centroid, etc.)
    """
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Apply threshold
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract features for each contour
    area_features = []
    
    for contour in contours:
        # Calculate area
        area = cv2.contourArea(contour)
        
        # Skip small areas
        if area < min_area:
            continue
        
        # Calculate perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Calculate extent
        extent = float(area) / (w * h) if w * h > 0 else 0
        
        # Store features
        area_features.append({
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'centroid': (cx, cy),
            'bounding_box': (x, y, w, h),
            'aspect_ratio': aspect_ratio,
            'extent': extent
        })
    
    return area_features


def detect_anomalies_isolation_forest(features: np.ndarray, 
                                    contamination: float = 0.1,
                                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect anomalies using Isolation Forest.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        contamination: Expected proportion of anomalies
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (anomaly labels, anomaly scores)
    """
    # Create and fit the model
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    clf.fit(features)
    
    # Predict anomalies
    # -1 for anomalies and 1 for normal points
    labels = clf.predict(features)
    
    # Convert to binary labels (1 for anomalies, 0 for normal)
    binary_labels = np.where(labels == -1, 1, 0)
    
    # Get anomaly scores
    scores = clf.score_samples(features)
    
    # Invert scores so that higher values indicate more anomalous points
    scores = -scores
    
    return binary_labels, scores


def detect_anomalies_dbscan(features: np.ndarray, 
                          eps: float = 0.5,
                          min_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect anomalies using DBSCAN clustering.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        eps: Maximum distance between samples for neighborhood
        min_samples: Minimum number of samples in a neighborhood
        
    Returns:
        Tuple of (anomaly labels, anomaly scores)
    """
    # Create and fit the model
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    clustering.fit(features)
    
    # Get cluster labels
    labels = clustering.labels_
    
    # Points labeled as -1 are anomalies
    binary_labels = np.where(labels == -1, 1, 0)
    
    # Calculate anomaly scores based on distance to nearest cluster
    scores = np.zeros(len(features))
    
    # Get unique cluster labels (excluding -1)
    clusters = np.unique(labels)
    clusters = clusters[clusters != -1]
    
    # For each point, calculate distance to nearest cluster center
    for i in range(len(features)):
        if labels[i] == -1:
            # For anomalies, calculate distance to nearest cluster center
            min_dist = float('inf')
            
            for cluster in clusters:
                # Get points in this cluster
                cluster_points = features[labels == cluster]
                
                # Calculate distance to cluster center
                cluster_center = np.mean(cluster_points, axis=0)
                dist = np.linalg.norm(features[i] - cluster_center)
                
                # Update minimum distance
                if dist < min_dist:
                    min_dist = dist
            
            scores[i] = min_dist
        else:
            # For normal points, use small score
            scores[i] = 0.1
    
    # Normalize scores to [0, 1]
    if np.max(scores) > np.min(scores):
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    
    return binary_labels, scores


def detect_anomalies_kmeans(features: np.ndarray, 
                          n_clusters: int = 5,
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect anomalies using K-means clustering.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (anomaly labels, anomaly scores)
    """
    # Create and fit the model
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(features)
    
    # Get cluster centers and labels
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Calculate distance to assigned cluster center
    distances = np.zeros(len(features))
    for i in range(len(features)):
        cluster_idx = labels[i]
        distances[i] = np.linalg.norm(features[i] - centers[cluster_idx])
    
    # Define anomalies as points with distance > threshold
    # Use 3 standard deviations as threshold
    threshold = np.mean(distances) + 3 * np.std(distances)
    binary_labels = np.where(distances > threshold, 1, 0)
    
    # Normalize distances to [0, 1] for scores
    scores = distances / np.max(distances) if np.max(distances) > 0 else distances
    
    return binary_labels, scores


def visualize_time_series_anomalies(features: np.ndarray, 
                                  anomaly_scores: np.ndarray,
                                  anomaly_threshold: Optional[float] = None,
                                  feature_names: Optional[List[str]] = None,
                                  figsize: Tuple[int, int] = (15, 10),
                                  smooth_window: int = 5) -> None:
    """
    Visualize time series data with anomaly scores.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        anomaly_scores: Anomaly scores for each sample
        anomaly_threshold: Threshold for anomaly detection (None for automatic)
        feature_names: Names of features
        figsize: Figure size
        smooth_window: Window size for smoothing
    """
    # Set default feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(features.shape[1])]
    
    # Limit number of features to display
    max_features = min(5, features.shape[1])
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot features
    for i in range(max_features):
        plt.subplot(max_features + 1, 1, i + 1)
        
        # Get feature values
        feature_values = features[:, i]
        
        # Smooth feature values
        if smooth_window > 0 and len(feature_values) > smooth_window:
            feature_values = savgol_filter(feature_values, smooth_window, 2)
        
        plt.plot(feature_values, label=feature_names[i])
        
        # Highlight anomalies if threshold is provided
        if anomaly_threshold is not None:
            anomaly_indices = np.where(anomaly_scores > anomaly_threshold)[0]
            if len(anomaly_indices) > 0:
                plt.scatter(anomaly_indices, feature_values[anomaly_indices], 
                           color='red', label='Anomalies')
        
        plt.legend()
        plt.title(feature_names[i])
        plt.grid(True)
    
    # Plot anomaly scores
    plt.subplot(max_features + 1, 1, max_features + 1)
    plt.plot(anomaly_scores, color='blue', label='Anomaly Score')
    
    # Add threshold line if provided
    if anomaly_threshold is not None:
        plt.axhline(y=anomaly_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({anomaly_threshold:.3f})')
    
    plt.legend()
    plt.title('Anomaly Scores')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def visualize_frame_anomalies(frames: List[np.ndarray], 
                            anomaly_scores: np.ndarray,
                            anomaly_threshold: Optional[float] = None,
                            num_frames: int = 10,
                            figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Visualize frames with their anomaly scores.
    
    Args:
        frames: List of frames
        anomaly_scores: Anomaly scores for each frame
        anomaly_threshold: Threshold for anomaly detection (None for automatic)
        num_frames: Number of frames to display
        figsize: Figure size
    """
    # Set automatic threshold if not provided
    if anomaly_threshold is None:
        anomaly_threshold = np.mean(anomaly_scores) + 2 * np.std(anomaly_scores)
    
    # Find anomalous frames
    anomaly_indices = np.where(anomaly_scores > anomaly_threshold)[0]
    
    # If no anomalies found, show frames with highest scores
    if len(anomaly_indices) == 0:
        top_indices = np.argsort(anomaly_scores)[-num_frames:]
    else:
        # If more anomalies than num_frames, select top ones
        if len(anomaly_indices) > num_frames:
            # Get indices of top anomalies
            top_anomaly_scores = anomaly_scores[anomaly_indices]
            top_indices = anomaly_indices[np.argsort(top_anomaly_scores)[-num_frames:]]
        else:
            top_indices = anomaly_indices
    
    # Sort indices
    top_indices = sorted(top_indices)
    
    # Create figure
    fig, axes = plt.subplots(2, len(top_indices) // 2 + len(top_indices) % 2, figsize=figsize)
    axes = axes.flatten()
    
    # Plot frames
    for i, idx in enumerate(top_indices):
        if idx < len(frames):
            # Convert BGR to RGB if needed
            frame = frames[idx]
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            axes[i].imshow(frame)
            axes[i].set_title(f"Frame {idx}\nScore: {anomaly_scores[idx]:.3f}")
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(top_indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def extract_video_features(video_path: str, 
                         frame_interval: int = 10,
                         max_frames: Optional[int] = None,
                         feature_types: List[str] = ['color', 'motion', 'keypoint'],
                         resize: Optional[Tuple[int, int]] = (320, 240)) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Extract features from a video for anomaly detection.
    
    Args:
        video_path: Path to the video file
        frame_interval: Extract every nth frame
        max_frames: Maximum number of frames to extract
        feature_types: Types of features to extract
        resize: Size to resize frames to
        
    Returns:
        Tuple of (frames, features)
    """
    # Extract frames
    frames = extract_frames(video_path, None, frame_interval, max_frames, resize)
    
    if len(frames) == 0:
        raise ValueError("No frames extracted from video")
    
    # Extract features
    all_features = []
    
    # Color features
    if 'color' in feature_types:
        color_features = np.array([extract_color_features(frame) for frame in frames])
        all_features.append(color_features)
    
    # Motion features
    if 'motion' in feature_types and len(frames) > 1:
        motion_features = extract_motion_features(frames)
        
        # Pad first frame with zeros for motion features
        padding = np.zeros((1, motion_features.shape[1]))
        motion_features = np.vstack([padding, motion_features])
        
        all_features.append(motion_features)
    
    # Keypoint features
    if 'keypoint' in feature_types:
        keypoint_features = np.array([extract_keypoint_features(frame) for frame in frames])
        
        # Check if any keypoints were found
        if keypoint_features.size > 0 and keypoint_features[0].size > 0:
            all_features.append(keypoint_features)
    
    # Area features
    if 'area' in feature_types:
        # Extract area features
        area_features_list = [extract_area_features(frame) for frame in frames]
        
        # Convert to fixed-size feature vectors
        area_feature_vectors = []
        
        for areas in area_features_list:
            if len(areas) > 0:
                # Calculate statistics of area features
                areas_array = np.array([
                    [area['area'], area['perimeter'], area['circularity'], 
                     area['aspect_ratio'], area['extent']] 
                    for area in areas
                ])
                
                # Calculate mean, std, min, max
                mean_features = np.mean(areas_array, axis=0)
                std_features = np.std(areas_array, axis=0)
                min_features = np.min(areas_array, axis=0)
                max_features = np.max(areas_array, axis=0)
                
                # Combine statistics
                area_feature_vector = np.concatenate([
                    mean_features, std_features, min_features, max_features
                ])
            else:
                # No areas found, use zeros
                area_feature_vector = np.zeros(20)  # 5 features * 4 statistics
            
            area_feature_vectors.append(area_feature_vector)
        
        all_features.append(np.array(area_feature_vectors))
    
    # Combine all features
    if all_features:
        # Ensure all feature arrays have the same number of samples
        min_samples = min(feat.shape[0] for feat in all_features)
        all_features = [feat[:min_samples] for feat in all_features]
        
        # Concatenate features
        combined_features = np.hstack(all_features)
        
        # Keep only frames that have features
        frames = frames[:min_samples]
    else:
        combined_features = np.array([])
    
    return frames, combined_features


def detect_video_anomalies(video_path: str, 
                         method: str = 'isolation_forest',
                         frame_interval: int = 10,
                         max_frames: Optional[int] = None,
                         feature_types: List[str] = ['color', 'motion', 'keypoint'],
                         resize: Optional[Tuple[int, int]] = (320, 240),
                         contamination: float = 0.1,
                         visualize: bool = True) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Detect anomalies in a video.
    
    Args:
        video_path: Path to the video file
        method: Anomaly detection method ('isolation_forest', 'dbscan', 'kmeans')
        frame_interval: Extract every nth frame
        max_frames: Maximum number of frames to extract
        feature_types: Types of features to extract
        resize: Size to resize frames to
        contamination: Expected proportion of anomalies
        visualize: Whether to visualize results
        
    Returns:
        Tuple of (frames, anomaly_labels, anomaly_scores)
    """
    # Extract features
    frames, features = extract_video_features(
        video_path, frame_interval, max_frames, feature_types, resize
    )
    
    if len(features) == 0:
        raise ValueError("No features extracted from video")
    
    # Apply PCA if feature dimension is high
    if features.shape[1] > 50:
        pca = PCA(n_components=min(50, features.shape[0]))
        features = pca.fit_transform(features)
    
    # Detect anomalies
    if method == 'isolation_forest':
        anomaly_labels, anomaly_scores = detect_anomalies_isolation_forest(
            features, contamination
        )
    elif method == 'dbscan':
        anomaly_labels, anomaly_scores = detect_anomalies_dbscan(
            features
        )
    elif method == 'kmeans':
        anomaly_labels, anomaly_scores = detect_anomalies_kmeans(
            features
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Visualize results
    if visualize:
        # Visualize time series
        visualize_time_series_anomalies(
            features, anomaly_scores,
            anomaly_threshold=np.percentile(anomaly_scores, 100 - contamination * 100)
        )
        
        # Visualize frames
        visualize_frame_anomalies(
            frames, anomaly_scores,
            anomaly_threshold=np.percentile(anomaly_scores, 100 - contamination * 100)
        )
    
    return frames, anomaly_labels, anomaly_scores
