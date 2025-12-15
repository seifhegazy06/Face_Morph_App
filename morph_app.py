"""
MorphApplication class - Main application that orchestrates all components.
"""
import cv2
import mediapipe as mp
from target_image import TargetImage
from face_morpher import FaceMorpher
from video_recorder import VideoRecorder
from morph_ui import MorphUI


class MorphApplication:
    """Main application class for real-time face morphing."""
    
    def __init__(self, target_folder="Targets", frame_width=640, frame_height=480):
        """
        Initialize the morph application.
        
        Args:
            target_folder: Folder containing target images and JSON files
            frame_width: Width of video capture
            frame_height: Height of video capture
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_folder = target_folder
        
        # Load targets
        print("Loading targets...")
        self.targets = TargetImage.load_all_from_folder(target_folder)
        
        if len(self.targets) == 0:
            raise Exception(f"No targets found in folder: {target_folder}")
        
        print(f"Loaded {len(self.targets)} targets: {[t.name for t in self.targets]}")
        
        # Initialize components
        self.ui = MorphUI("Real-time Morph", frame_width, frame_height)
        self.ui.set_targets(self.targets)
        self.ui.set_target_change_callback(self._on_target_change)
        self.ui.enable_mouse_callback()
        
        self.morpher = FaceMorpher(self.targets[0])
        self.recorder = VideoRecorder()
        
        # Initialize webcam
        print("Initializing webcam...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        
        # Initialize MediaPipe Face Mesh
        print("Initializing face detection...")
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Application state
        self.is_running = False
    
    def _on_target_change(self, new_index):
        """Callback when user selects a different target."""
        new_target = self.targets[new_index]
        self.morpher.update_target(new_target)
    
    def process_frame(self, frame):
        """
        Process a single frame: detect faces and apply morphing.
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            Processed frame with morphing applied
        """
        # Resize frame
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        # Start with original frame
        display = frame.copy()
        
        # Apply morphing if faces detected
        if results.multi_face_landmarks:
            # Get alpha from UI
            alpha = self.ui.get_alpha()
            
            # Process each detected face
            for face_landmarks in results.multi_face_landmarks:
                display = self.morpher.morph_face(
                    display, 
                    face_landmarks, 
                    alpha=alpha,
                    preserve_eyes=True,
                    preserve_mouth=True
                )
        
        return display
    
    def run(self):
        """Run the main application loop."""
        self.is_running = True
        print("\n" + "="*50)
        print("APPLICATION STARTED")
        print("="*50)
        print("Controls:")
        print("  ESC - Quit")
        print("  R   - Toggle recording")
        print("  Click icons to switch filters")
        print("="*50 + "\n")
        
        try:
            while self.is_running:
                # Read frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from webcam")
                    break
                
                # Process frame (detect faces and apply morphing)
                display = self.process_frame(frame)
                
                # Record frame if recording (BEFORE adding UI elements)
                if self.recorder.is_recording:
                    self.recorder.add_frame(display)
                
                # Add UI elements (icon bar and recording indicator)
                self.ui.draw_icon_bar(display)
                
                if self.recorder.is_recording:
                    self.ui.draw_recording_indicator(display)
                
                # Display frame
                self.ui.show_frame(display)
                
                # Handle keyboard input
                key = self.ui.wait_key(1)
                
                # Check if window was closed by clicking the X button
                try:
                    if cv2.getWindowProperty("Real-time Morph", cv2.WND_PROP_VISIBLE) < 1:
                        print("\nWindow closed - exiting...")
                        break
                except:
                    print("\nWindow closed - exiting...")
                    break
                
                if key == 27:  # ESC to quit
                    print("\nESC pressed - exiting...")
                    break
                
                elif key == ord('r') or key == ord('R'):  # R to toggle recording
                    if not self.recorder.is_recording:
                        self.recorder.start_recording(self.frame_width, self.frame_height)
                    else:
                        output_file = self.recorder.stop_recording()
                        if output_file:
                            print(f"Recording saved to: {output_file}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        
        if self.recorder.is_recording:
            print("Stopping recording...")
            self.recorder.stop_recording()
        
        self.recorder.cleanup()
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("Program closed.")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass


def main():
    """Main entry point."""
    try:
        app = MorphApplication(target_folder="Targets")
        app.run()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
