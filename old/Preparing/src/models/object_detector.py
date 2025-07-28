"""
Object detection models for anomaly detection in computer vision.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union
import os
import time
from tqdm import tqdm
from PIL import Image
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.ops import box_iou


class ObjectDetector:
    """Base class for object detection models."""
    
    def __init__(self, num_classes: int, device: str = None):
        """
        Initialize the object detector.
        
        Args:
            num_classes: Number of classes (including background)
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.num_classes = num_classes
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = None
        self.optimizer = None
        self.history = {'train_loss': [], 'val_loss': []}
        
    def build_model(self):
        """Build the model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement build_model()")
    
    def train_epoch(self, data_loader: torch.utils.data.DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            data_loader: Training data loader
            
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0
        
        for images, targets in tqdm(data_loader, desc="Training", leave=False):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            
            total_loss += losses.item()
        
        return total_loss / len(data_loader)
    
    def validate_epoch(self, data_loader: torch.utils.data.DataLoader) -> float:
        """
        Validate for one epoch.
        
        Args:
            data_loader: Validation data loader
            
        Returns:
            Average loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="Validating", leave=False):
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                total_loss += losses.item()
        
        return total_loss / len(data_loader)
    
    def fit(self, train_loader: torch.utils.data.DataLoader, 
           val_loader: Optional[torch.utils.data.DataLoader] = None,
           epochs: int = 10, patience: int = 5, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            patience: Early stopping patience
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                          f"Time: {time.time() - start_time:.2f}s")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Time: {time.time() - start_time:.2f}s")
        
        # Load best model if validation was used
        if val_loader is not None and hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            
        return self.history
    
    def predict(self, images: List[torch.Tensor], confidence_threshold: float = 0.5) -> List[Dict[str, torch.Tensor]]:
        """
        Detect objects in images.
        
        Args:
            images: List of image tensors
            confidence_threshold: Confidence threshold for detections
            
        Returns:
            List of dictionaries with detection results
        """
        self.model.eval()
        
        device_images = [img.to(self.device) for img in images]
        
        with torch.no_grad():
            predictions = self.model(device_images)
            
        # Filter predictions by confidence threshold
        filtered_predictions = []
        for pred in predictions:
            scores = pred['scores']
            mask = scores >= confidence_threshold
            
            filtered_pred = {
                'boxes': pred['boxes'][mask].cpu(),
                'labels': pred['labels'][mask].cpu(),
                'scores': scores[mask].cpu()
            }
            filtered_predictions.append(filtered_pred)
            
        return filtered_predictions
    
    def predict_single_image(self, image: Union[np.ndarray, torch.Tensor], 
                           confidence_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Detect objects in a single image.
        
        Args:
            image: Image as numpy array or tensor
            confidence_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary with detection results
        """
        # Convert numpy array to tensor if needed
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            image = F.to_tensor(image)
        
        # Add batch dimension
        image = image.unsqueeze(0)
        
        # Get predictions
        predictions = self.predict(image, confidence_threshold)
        
        return predictions[0]
    
    def evaluate(self, data_loader: torch.utils.data.DataLoader, 
                iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate the model using mAP.
        
        Args:
            data_loader: Test data loader
            iou_threshold: IoU threshold for considering a detection as correct
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        # Initialize metrics
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'mAP': 0.0
        }
        
        all_gt_boxes = []
        all_gt_labels = []
        all_pred_boxes = []
        all_pred_labels = []
        all_pred_scores = []
        
        # Collect predictions and ground truth
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="Evaluating"):
                images = list(img.to(self.device) for img in images)
                
                # Get predictions
                predictions = self.model(images)
                
                # Store ground truth and predictions
                for target, pred in zip(targets, predictions):
                    all_gt_boxes.append(target['boxes'])
                    all_gt_labels.append(target['labels'])
                    
                    all_pred_boxes.append(pred['boxes'].cpu())
                    all_pred_labels.append(pred['labels'].cpu())
                    all_pred_scores.append(pred['scores'].cpu())
        
        # Calculate mAP
        ap_per_class = {}
        
        # For each class
        for class_id in range(1, self.num_classes):  # Skip background class (0)
            # Collect predictions and ground truth for this class
            class_preds = []
            class_gt_count = 0
            
            for i in range(len(all_pred_boxes)):
                # Get predictions for this class
                mask = all_pred_labels[i] == class_id
                boxes = all_pred_boxes[i][mask]
                scores = all_pred_scores[i][mask]
                
                # Sort by score
                sorted_indices = torch.argsort(scores, descending=True)
                boxes = boxes[sorted_indices]
                scores = scores[sorted_indices]
                
                # Add to class predictions
                for box, score in zip(boxes, scores):
                    class_preds.append({
                        'image_id': i,
                        'box': box,
                        'score': score,
                        'matched': False
                    })
                
                # Count ground truth for this class
                gt_mask = all_gt_labels[i] == class_id
                class_gt_count += gt_mask.sum().item()
            
            # Sort predictions by score
            class_preds.sort(key=lambda x: x['score'], reverse=True)
            
            # Calculate precision-recall curve
            tp = 0
            fp = 0
            precision_values = []
            recall_values = []
            
            for pred in class_preds:
                image_id = pred['image_id']
                pred_box = pred['box'].unsqueeze(0)
                
                # Get ground truth boxes for this image and class
                gt_mask = all_gt_labels[image_id] == class_id
                gt_boxes = all_gt_boxes[image_id][gt_mask]
                
                if len(gt_boxes) == 0:
                    # False positive
                    fp += 1
                else:
                    # Calculate IoU with all ground truth boxes
                    ious = box_iou(pred_box, gt_boxes)
                    max_iou, max_idx = torch.max(ious, dim=1)
                    
                    if max_iou >= iou_threshold:
                        # True positive
                        tp += 1
                    else:
                        # False positive
                        fp += 1
                
                # Calculate precision and recall
                precision = tp / (tp + fp)
                recall = tp / class_gt_count if class_gt_count > 0 else 0
                
                precision_values.append(precision)
                recall_values.append(recall)
            
            # Calculate AP using 11-point interpolation
            ap = 0.0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(np.array(recall_values) >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.array(precision_values)[np.array(recall_values) >= t])
                ap += p / 11
            
            ap_per_class[class_id] = ap
        
        # Calculate mAP
        metrics['mAP'] = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
        
        # Calculate overall precision, recall, and F1 score
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for i in range(len(all_pred_boxes)):
            pred_boxes = all_pred_boxes[i]
            pred_labels = all_pred_labels[i]
            pred_scores = all_pred_scores[i]
            
            gt_boxes = all_gt_boxes[i]
            gt_labels = all_gt_labels[i]
            
            # Match predictions to ground truth
            matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
            
            for j in range(len(pred_boxes)):
                pred_box = pred_boxes[j].unsqueeze(0)
                pred_label = pred_labels[j].item()
                
                # Find ground truth boxes with same label
                same_label_mask = gt_labels == pred_label
                
                if same_label_mask.sum() == 0:
                    # No ground truth with this label, false positive
                    total_fp += 1
                    continue
                
                # Calculate IoU with all ground truth boxes of same label
                gt_boxes_same_label = gt_boxes[same_label_mask]
                ious = box_iou(pred_box, gt_boxes_same_label)
                max_iou, max_idx = torch.max(ious, dim=1)
                
                if max_iou >= iou_threshold:
                    # Find original index in gt_boxes
                    orig_idx = torch.where(same_label_mask)[0][max_idx]
                    
                    if not matched_gt[orig_idx]:
                        # True positive
                        total_tp += 1
                        matched_gt[orig_idx] = True
                    else:
                        # Already matched, false positive
                        total_fp += 1
                else:
                    # No match, false positive
                    total_fp += 1
            
            # Count false negatives (unmatched ground truth)
            total_fn += (~matched_gt).sum().item()
        
        # Calculate metrics
        if total_tp + total_fp > 0:
            metrics['precision'] = total_tp / (total_tp + total_fp)
        else:
            metrics['precision'] = 0.0
            
        if total_tp + total_fn > 0:
            metrics['recall'] = total_tp / (total_tp + total_fn)
        else:
            metrics['recall'] = 0.0
            
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        return metrics
    
    def save(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'num_classes': self.num_classes,
            'history': self.history
        }
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the model.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Ensure model is built
        if self.model is None:
            self.num_classes = checkpoint['num_classes']
            self.build_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            
        print(f"Model loaded from {path}")
    
    def visualize_detections(self, image: np.ndarray, 
                           predictions: Dict[str, torch.Tensor],
                           class_names: List[str] = None,
                           figsize: Tuple[int, int] = (12, 8),
                           score_threshold: float = 0.5) -> None:
        """
        Visualize object detections.
        
        Args:
            image: Image as numpy array
            predictions: Predictions from model
            class_names: List of class names
            figsize: Figure size
            score_threshold: Score threshold for displaying detections
        """
        # Make a copy of the image
        img_copy = image.copy()
        
        # Get boxes, labels and scores
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        # Filter by score threshold
        mask = scores >= score_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        
        # Create figure
        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        
        # Plot each box
        for box, label, score in zip(boxes, labels, scores):
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, box)
            
            # Get class name
            if class_names is not None and 0 <= label < len(class_names):
                class_name = class_names[label]
            else:
                class_name = f"Class {label}"
            
            # Create label text
            label_text = f"{class_name}: {score:.2f}"
            
            # Draw rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img_copy, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(img_copy, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot training history.
        
        Args:
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        plt.plot(self.history['train_loss'], label='Train Loss')
        if 'val_loss' in self.history and self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Validation Loss')
            
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class FasterRCNNDetector(ObjectDetector):
    """Faster R-CNN object detector."""
    
    def __init__(self, num_classes: int, pretrained: bool = True,
                learning_rate: float = 0.005, device: str = None):
        """
        Initialize Faster R-CNN detector.
        
        Args:
            num_classes: Number of classes (including background)
            pretrained: Whether to use pretrained weights
            learning_rate: Learning rate
            device: Device to use
        """
        super().__init__(num_classes, device)
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        
        self.build_model()
    
    def build_model(self) -> None:
        """Build Faster R-CNN model."""
        # Load pre-trained model
        if self.pretrained:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn(weights=weights)
        else:
            self.model = fasterrcnn_resnet50_fpn(weights=None)
        
        # Replace the classifier with a new one for our number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set up optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params, 
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=0.0005
        )


class RetinaNetDetector(ObjectDetector):
    """RetinaNet object detector."""
    
    def __init__(self, num_classes: int, pretrained: bool = True,
                learning_rate: float = 0.005, device: str = None):
        """
        Initialize RetinaNet detector.
        
        Args:
            num_classes: Number of classes (excluding background)
            pretrained: Whether to use pretrained weights
            learning_rate: Learning rate
            device: Device to use
        """
        # RetinaNet counts classes differently (excludes background)
        super().__init__(num_classes, device)
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        
        self.build_model()
    
    def build_model(self) -> None:
        """Build RetinaNet model."""
        # Load pre-trained model
        if self.pretrained:
            weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
            self.model = retinanet_resnet50_fpn(weights=weights)
        else:
            self.model = retinanet_resnet50_fpn(weights=None)
        
        # Replace the classifier with a new one for our number of classes
        # RetinaNet uses num_classes + 1 for background
        self.model.head.classification_head.num_classes = self.num_classes
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set up optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params, 
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=0.0005
        )


def create_yolov8_detector(num_classes: int, model_size: str = 'n', 
                         pretrained: bool = True) -> Any:
    """
    Create a YOLOv8 detector using Ultralytics YOLOv8.
    
    Args:
        num_classes: Number of classes
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        pretrained: Whether to use pretrained weights
        
    Returns:
        YOLOv8 model
    """
    try:
        from ultralytics import YOLO
        
        if pretrained:
            # Load pretrained model
            model = YOLO(f'yolov8{model_size}.pt')
            
            # If custom number of classes, customize the model
            if num_classes != 80:  # COCO has 80 classes
                model = model.train(data=None)  # This initializes the model for training
                # The model will be trained with custom data that has the specified number of classes
        else:
            # Create new model with specified number of classes
            model = YOLO(f'yolov8{model_size}.yaml')
        
        return model
    
    except ImportError:
        print("Error: Ultralytics package not found. Install with: pip install ultralytics")
        return None


def run_batch_inference(model: Any, image_paths: List[str], 
                      output_dir: str, confidence: float = 0.25,
                      batch_size: int = 16) -> List[Dict[str, Any]]:
    """
    Run batch inference on multiple images.
    
    Args:
        model: Object detection model (YOLOv8 or ObjectDetector)
        image_paths: List of image paths
        output_dir: Directory to save results
        confidence: Confidence threshold
        batch_size: Batch size
        
    Returns:
        List of results for each image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if model is YOLOv8 (Ultralytics)
    is_yolo = hasattr(model, 'predict')
    
    if is_yolo:
        # Use YOLOv8's batch prediction
        results = model.predict(
            source=image_paths,
            conf=confidence,
            save=True,
            project=output_dir,
            exist_ok=True
        )
        return results
    
    else:
        # Use our ObjectDetector class
        results = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            # Load images
            for path in batch_paths:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = F.to_tensor(img)
                batch_images.append(tensor)
            
            # Get predictions
            predictions = model.predict(batch_images, confidence_threshold=confidence)
            
            # Save results
            for j, (path, pred) in enumerate(zip(batch_paths, predictions)):
                img = cv2.imread(path)
                
                # Draw boxes
                for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    label_text = f"Class {label.item()}: {score.item():.2f}"
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label_text, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save image
                filename = os.path.basename(path)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, img)
                
                # Store result
                results.append({
                    'path': path,
                    'predictions': pred,
                    'output_path': output_path
                })
        
        return results
