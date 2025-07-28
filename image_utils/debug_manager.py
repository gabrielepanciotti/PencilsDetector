"""
Utility module for managing debug images.
"""
import os
import shutil
import glob
import time
from datetime import datetime
from typing import Optional, Dict, Any

class DebugManager:
    """
    Class to manage debug images, including:
    - Moving old debug images to an 'old' folder
    - Adding chronological indices to new debug images
    - Creating organized debug directories
    """
    def __init__(self, base_dir: str = "debug"):
        """
        Initialize the debug manager.
        
        Args:
            base_dir: Base directory for debug images
        """
        self.base_dir = os.path.join(os.getcwd(), base_dir)
        self.execution_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.step_counter = 0
        self.step_timestamps = {}
        
        # Create the base directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        
    def prepare_debug_directory(self) -> str:
        """
        Prepare the debug directory by moving old files to an 'old' folder.
        
        Returns:
            Path to the debug directory
        """
        # Create old directory if it doesn't exist
        old_dir = os.path.join(self.base_dir, "old")
        os.makedirs(old_dir, exist_ok=True)
        
        # Move existing files to old directory with timestamp
        existing_files = glob.glob(os.path.join(self.base_dir, "*.jpg")) + \
                        glob.glob(os.path.join(self.base_dir, "*.png")) + \
                        glob.glob(os.path.join(self.base_dir, "*.csv"))
        
        if existing_files:
            # Create a timestamped folder in the old directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = os.path.join(old_dir, f"run_{timestamp}")
            os.makedirs(archive_dir, exist_ok=True)
            
            # Move files to the archive directory
            for file_path in existing_files:
                if os.path.isfile(file_path):
                    file_name = os.path.basename(file_path)
                    shutil.move(file_path, os.path.join(archive_dir, file_name))
                    
        return self.base_dir
    
    def get_debug_path(self, filename: str, step_name: Optional[str] = None) -> str:
        """
        Get a path for a debug file with chronological index.
        
        Args:
            filename: Original filename
            step_name: Optional step name to include in the filename
            
        Returns:
            Path to the debug file with chronological index
        """
        # Increment step counter
        self.step_counter += 1
        
        # Get file extension and base name
        base_name, ext = os.path.splitext(filename)
        
        # Create a timestamped filename with step counter
        if step_name:
            # Record timestamp for this step if it's new
            if step_name not in self.step_timestamps:
                self.step_timestamps[step_name] = len(self.step_timestamps) + 1
            
            step_index = self.step_timestamps[step_name]
            new_filename = f"{step_index:02d}_{self.step_counter:03d}_{step_name}_{base_name}{ext}"
        else:
            new_filename = f"{self.step_counter:03d}_{base_name}{ext}"
        
        return os.path.join(self.base_dir, new_filename)
    
    def save_debug_image(self, image, filename: str, step_name: Optional[str] = None) -> str:
        """
        Save a debug image with chronological index.
        
        Args:
            image: Image to save
            filename: Original filename
            step_name: Optional step name to include in the filename
            
        Returns:
            Path to the saved debug file
        """
        import cv2
        
        # Get path with chronological index
        path = self.get_debug_path(filename, step_name)
        
        # Save the image
        cv2.imwrite(path, image)
        
        return path
    
    def save_debug_csv(self, data, filename, step_name=None):
        """
        Save debug data as CSV with chronological index.
        
        Args:
            data: Dictionary with data to save or list of lists (where first list contains headers)
            filename: Original filename
            step_name: Optional step name to include in the filename
            
        Returns:
            Path to the saved CSV file
        """
        import csv
        
        # Get path with chronological index
        path = self.get_debug_path(filename, step_name)
        
        # Save the CSV
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Check if data is a dictionary or a list of lists
            if isinstance(data, dict):
                writer.writerow(data.keys())
                writer.writerows(zip(*data.values()))
            elif isinstance(data, list) and len(data) > 0:
                # Assume first row is headers, rest are data
                for row in data:
                    writer.writerow(row)
            else:
                raise ValueError("Data must be a dictionary or a non-empty list of lists")
        
        return path

# Global debug manager instance
debug_manager = DebugManager()

def prepare_debug_directory() -> str:
    """
    Prepare the debug directory by moving old files to an 'old' folder.
    
    Returns:
        Path to the debug directory
    """
    return debug_manager.prepare_debug_directory()

def get_debug_path(filename: str, step_name: Optional[str] = None) -> str:
    """
    Get a path for a debug file with chronological index.
    
    Args:
        filename: Original filename
        step_name: Optional step name to include in the filename
        
    Returns:
        Path to the debug file with chronological index
    """
    return debug_manager.get_debug_path(filename, step_name)

def save_debug_image(image, filename: str, step_name: Optional[str] = None) -> str:
    """
    Save a debug image with chronological index.
    
    Args:
        image: Image to save
        filename: Original filename
        step_name: Optional step name to include in the filename
        
    Returns:
        Path to the saved debug file
    """
    return debug_manager.save_debug_image(image, filename, step_name)

def save_debug_csv(data: Dict[str, Any], filename: str, step_name: Optional[str] = None) -> str:
    """
    Save debug data as CSV with chronological index.
    
    Args:
        data: Dictionary with data to save
        filename: Original filename
        step_name: Optional step name to include in the filename
        
    Returns:
        Path to the saved CSV file
    """
    return debug_manager.save_debug_csv(data, filename, step_name)
