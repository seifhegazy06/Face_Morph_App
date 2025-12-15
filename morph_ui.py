"""
MorphUI class - Handles the user interface including display, icon bar, and mouse events.
"""
import cv2
import numpy as np


class MorphUI:
    """Manages the user interface for the morph application."""
    
    def __init__(self, window_name="Real-time Morph", frame_width=640, frame_height=480):
        """
        Initialize the UI.
        
        Args:
            window_name: Name of the OpenCV window
            frame_width: Width of the display window
            frame_height: Height of the display window
        """
        self.window_name = window_name
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # UI state
        self.active_target_index = 0
        self.targets = []
        self.alpha = 0.5
        
        # Icon bar settings
        self.icon_bar_height = 90
        self.icon_size = 70
        self.icon_spacing = 80
        self.icon_margin = 20
        
        # Mouse callback data
        self.on_target_change = None  # Callback function when target changes
        
        # Create window and controls
        self._create_window()
    
    def _create_window(self):
        """Create the OpenCV window and trackbar."""
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar("alpha", self.window_name, 50, 100, self._on_alpha_change)
    
    def _on_alpha_change(self, value):
        """Trackbar callback for alpha value."""
        self.alpha = value / 100.0
    
    def set_targets(self, targets):
        """
        Set the list of available targets.
        
        Args:
            targets: List of TargetImage objects
        """
        self.targets = targets
        self.active_target_index = 0
    
    def set_target_change_callback(self, callback):
        """
        Set a callback function to be called when the target changes.
        
        Args:
            callback: Function that takes the new target index as parameter
        """
        self.on_target_change = callback
    
    def get_active_target(self):
        """Get the currently active target."""
        if 0 <= self.active_target_index < len(self.targets):
            return self.targets[self.active_target_index]
        return None
    
    def get_alpha(self):
        """Get the current alpha value."""
        return self.alpha
    
    def enable_mouse_callback(self):
        """Enable mouse event handling."""
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for target selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            bar_y = self.frame_height - self.icon_bar_height
            
            # Check if click is in icon bar area
            if y >= bar_y:
                # Check which icon was clicked
                for i in range(len(self.targets)):
                    x0 = self.icon_margin + i * self.icon_spacing
                    x1 = x0 + self.icon_size
                    
                    if x0 <= x <= x1:
                        self.active_target_index = i
                        print(f"Switched to: {self.targets[i].name}")
                        
                        # Call the callback if set
                        if self.on_target_change is not None:
                            self.on_target_change(i)
                        
                        return
    
    def draw_icon_bar(self, frame):
        """
        Draw the icon bar at the bottom of the frame.
        
        Args:
            frame: Frame to draw on (modified in-place)
        """
        icon_y = self.frame_height - self.icon_bar_height
        x0 = self.icon_margin
        
        for i, target in enumerate(self.targets):
            center_x = x0 + 35  # Center of 70px icon
            center_y = icon_y + 35
            
            # Draw selection circle
            if i == self.active_target_index:
                cv2.circle(frame, (center_x, center_y), 38, (0, 255, 255), 3)
            
            # Create circular mask for icon
            icon_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
            cv2.circle(icon_mask, (center_x, center_y), 35, 255, -1)
            
            # Apply icon with circular mask
            icon_region = frame[icon_y:icon_y+self.icon_size, x0:x0+self.icon_size]
            icon_mask_region = icon_mask[icon_y:icon_y+self.icon_size, x0:x0+self.icon_size]
            icon_mask_3ch = cv2.merge([icon_mask_region, icon_mask_region, icon_mask_region]) / 255.0
            
            # Blend icon into frame
            icon_region[:] = (target.icon * icon_mask_3ch + icon_region * (1 - icon_mask_3ch)).astype(np.uint8)
            
            x0 += self.icon_spacing
    
    def draw_recording_indicator(self, frame):
        """
        Draw recording indicator in top-right corner.
        
        Args:
            frame: Frame to draw on (modified in-place)
        """
        cv2.circle(frame, (self.frame_width - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (self.frame_width - 70, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def show_frame(self, frame):
        """
        Display the frame in the window.
        
        Args:
            frame: Frame to display
        """
        cv2.imshow(self.window_name, frame)
    
    def wait_key(self, delay=1):
        """
        Wait for key press.
        
        Args:
            delay: Delay in milliseconds
            
        Returns:
            Key code or -1 if no key pressed
        """
        return cv2.waitKey(delay)
    
    def destroy_window(self):
        """Close the window."""
        cv2.destroyWindow(self.window_name)
    
    def __del__(self):
        """Destructor to ensure window cleanup."""
        try:
            self.destroy_window()
        except:
            pass
