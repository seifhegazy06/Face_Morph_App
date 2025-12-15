# Face Morpher - OOP Version

A real-time face morphing application with clean object-oriented architecture.

## Project Structure

```
.
├── target_image.py      # TargetImage class - manages morph targets
├── face_morpher.py      # FaceMorpher class - handles morphing logic
├── video_recorder.py    # VideoRecorder class - manages recording
├── morph_ui.py          # MorphUI class - handles user interface
├── morph_app.py         # MorphApplication class - main orchestrator
├── requirements.txt     # Python dependencies
└── Targets/            # Folder with target images and JSON files
    ├── filter1.png
    ├── filter1.json
    ├── filter2.png
    └── filter2.json
```

## Class Responsibilities

### 1. **TargetImage** (`target_image.py`)
- Loads target images and their landmark data
- Creates circular icons for UI display
- Validates image and JSON file existence
- Static method to load all targets from a folder

### 2. **FaceMorpher** (`face_morpher.py`)
- Performs Delaunay triangulation on target landmarks
- Warps triangles from target to detected face
- Creates masks for face, eyes, and mouth areas
- Blends morphed target with original frame
- Preserves real eyes and teeth for natural look

### 3. **VideoRecorder** (`video_recorder.py`)
- Records video frames and audio simultaneously
- Runs audio recording in a separate thread
- Merges audio and video using moviepy
- Automatically creates timestamped output files
- Handles cleanup of temporary files

### 4. **MorphUI** (`morph_ui.py`)
- Creates and manages the OpenCV window
- Draws icon bar with target selection
- Handles mouse clicks for target switching
- Displays recording indicator
- Manages alpha blending trackbar

### 5. **MorphApplication** (`morph_app.py`)
- Orchestrates all components
- Initializes webcam and MediaPipe Face Mesh
- Runs main processing loop
- Handles keyboard commands
- Ensures proper cleanup on exit

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the `Targets` folder with your morph targets:
   - Each target needs a `.png` image file
   - Each target needs a corresponding `.json` file with landmarks

## Usage

Run the application:
```bash
python morph_app.py
```

### Controls
- **ESC** - Quit the application
- **R** - Start/stop recording
- **Click icons** - Switch between different morph targets
- **Alpha slider** - Adjust morph intensity (0-100%)

### Output
- Recordings are saved to the `Recordings/` folder
- Files are named with timestamp: `morph_YYYYMMDD_HHMMSS.mp4`
- Video includes both video and audio if microphone is available

## Key Features

### Object-Oriented Benefits
- **Separation of Concerns**: Each class has a single, clear responsibility
- **Testability**: Components can be tested independently
- **Reusability**: Classes can be used in other projects
- **Maintainability**: Easy to modify one component without affecting others
- **Extensibility**: Simple to add new features or target types

### Technical Features
- Real-time face detection using MediaPipe
- Delaunay triangulation for smooth morphing
- Preserves real eyes and mouth for natural appearance
- Multi-face support (up to 5 faces simultaneously)
- Simultaneous video and audio recording
- Clean UI with circular icon selector

## Customization

### Adding New Targets
1. Add a new `.png` image to the `Targets/` folder
2. Create a corresponding `.json` file with landmark data
3. The application will automatically load it on startup

### Adjusting Morph Settings
In `morph_app.py`, modify the `process_frame` method:
```python
display = self.morpher.morph_face(
    display, 
    face_landmarks, 
    alpha=alpha,
    preserve_eyes=True,    # Set to False to morph eyes
    preserve_mouth=True    # Set to False to morph mouth
)
```

### Changing Video Settings
In `morph_app.py` initialization:
```python
app = MorphApplication(
    target_folder="Targets",
    frame_width=640,    # Change resolution
    frame_height=480
)
```

## Dependencies
- OpenCV (cv2) - Image processing and display
- MediaPipe - Face landmark detection
- NumPy - Array operations
- SciPy - Delaunay triangulation
- PyAudio - Audio recording
- MoviePy - Video/audio merging

## Troubleshooting

### No webcam detected
- Check if your webcam is connected and working
- Try changing the camera index in `morph_app.py`: `cv2.VideoCapture(1)`

### Audio not recording
- Check microphone permissions
- Install PyAudio: `pip install pyaudio`
- On Linux, you may need: `sudo apt-get install portaudio19-dev`

### Targets not loading
- Ensure each `.png` has a matching `.json` file
- Check that JSON files contain "points", "width", and "height" keys
- Verify the `Targets/` folder path is correct

## License
MIT License - Feel free to use and modify as needed.
