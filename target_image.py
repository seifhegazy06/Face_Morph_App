"""
TargetImage class - Represents a single morph target with image, landmarks, and icon.
"""
import cv2
import numpy as np
import json
import os


class TargetImage:
    """Encapsulates a morph target image with its landmarks and icon."""
    
    def __init__(self, image_path, json_path, icon_size=70):
        """
        Initialize a target image.
        
        Args:
            image_path: Path to the target image file
            json_path: Path to the landmarks JSON file
            icon_size: Size of the circular icon for UI display
        """
        self.name = os.path.splitext(os.path.basename(image_path))[0]
        self.image_path = image_path
        self.json_path = json_path
        self.icon_size = icon_size
        
        # Load image and landmarks
        self.img = self._load_image(image_path)
        self.pts, self.width, self.height = self._load_landmarks(json_path)
        
        # Resize image to match landmark dimensions
        self.img = cv2.resize(self.img, (self.width, self.height))
        
        # Create circular icon for UI
        self.icon = self._make_circle_icon(self.img, icon_size)
    
    def _load_image(self, path):
        """Load image and convert RGBA to BGR if needed."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        
        # Convert RGBA to BGR if needed
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img
    
    def _load_landmarks(self, path):
        """Load landmarks from JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Landmarks file not found: {path}")
        
        with open(path, "r") as f:
            data = json.load(f)
        
        pts = np.array(data["points"], dtype=np.int32)
        w = int(data["width"])
        h = int(data["height"])
        
        return pts, w, h
    
    def _make_circle_icon(self, img, size):
        """Create a circular icon from the image."""
        # Convert to BGR if it has alpha channel
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        icon = cv2.resize(img, (size, size))
        mask = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.circle(mask, (size//2, size//2), size//2, (1, 1, 1), -1)
        
        return (icon * mask).astype(np.uint8)
    
    @staticmethod
    def load_all_from_folder(folder_path, icon_size=70):
        """
        Load all target images from a folder.
        
        Args:
            folder_path: Path to folder containing PNG images and JSON files
            icon_size: Size of circular icons
            
        Returns:
            List of TargetImage objects
        """
        import glob
        
        png_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        targets = []
        
        for png_path in png_files:
            name = os.path.splitext(os.path.basename(png_path))[0]
            json_path = os.path.join(folder_path, name + ".json")
            
            if not os.path.exists(json_path):
                print(f"WARNING: JSON for {name} not found, skipping...")
                continue
            
            try:
                target = TargetImage(png_path, json_path, icon_size)
                targets.append(target)
                print(f"Loaded target: {target.name}")
            except Exception as e:
                print(f"ERROR loading {name}: {e}")
        
        return targets
    
    def __repr__(self):
        return f"TargetImage(name='{self.name}', landmarks={len(self.pts)})"
