# Class Diagram and Architecture

## Class Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                      MorphApplication                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Main Orchestrator                                          │ │
│  │ - Initializes all components                               │ │
│  │ - Runs main loop                                           │ │
│  │ - Handles keyboard input                                   │ │
│  │ - Manages cleanup                                          │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────┬──────────────┬──────────────┬──────────────┬───────────┘
         │              │              │              │
         │ uses         │ uses         │ uses         │ uses
         ▼              ▼              ▼              ▼
    ┌─────────┐   ┌───────────┐  ┌──────────┐  ┌────────────┐
    │ Target  │   │   Face    │  │  Video   │  │   Morph    │
    │ Image   │   │  Morpher  │  │ Recorder │  │    UI      │
    └─────────┘   └───────────┘  └──────────┘  └────────────┘
         │              │
         │ uses         │ uses
         └──────────────┘
```

## Detailed Class Structure

### MorphApplication
**Purpose**: Main controller that orchestrates all components

**Key Attributes**:
- `targets: List[TargetImage]` - All available morph targets
- `morpher: FaceMorpher` - Handles morphing operations
- `recorder: VideoRecorder` - Manages recording
- `ui: MorphUI` - Manages user interface
- `cap: cv2.VideoCapture` - Webcam capture
- `face_mesh: mediapipe.FaceMesh` - Face detection

**Key Methods**:
- `run()` - Main application loop
- `process_frame(frame)` - Process single frame
- `cleanup()` - Release resources

---

### TargetImage
**Purpose**: Encapsulates a single morph target with image and landmarks

**Key Attributes**:
- `name: str` - Target name
- `img: np.ndarray` - Target image (BGR)
- `pts: np.ndarray` - Landmark points
- `icon: np.ndarray` - Circular icon for UI
- `width: int` - Image width
- `height: int` - Image height

**Key Methods**:
- `__init__(image_path, json_path)` - Load single target
- `load_all_from_folder(folder_path)` - Static method to load all targets

---

### FaceMorpher
**Purpose**: Performs face morphing using Delaunay triangulation

**Key Attributes**:
- `target: TargetImage` - Current morph target
- `triangles: np.ndarray` - Delaunay triangulation indices

**Key Methods**:
- `morph_face(frame, face_landmarks, alpha)` - Main morphing function
- `update_target(new_target)` - Switch to new target
- `_warp_triangle(img_src, img_dst, t_src, t_dst)` - Warp single triangle
- `_create_face_mask()` - Create face region mask
- `_create_eyes_mask()` - Create eyes exclusion mask
- `_create_mouth_mask()` - Create mouth exclusion mask

---

### VideoRecorder
**Purpose**: Handles video and audio recording with merging

**Key Attributes**:
- `is_recording: bool` - Recording state
- `video_writer: cv2.VideoWriter` - Video writer
- `audio_frames: List[bytes]` - Audio data buffer
- `audio_thread: threading.Thread` - Audio recording thread
- `p_audio: pyaudio.PyAudio` - Audio interface

**Key Methods**:
- `start_recording(width, height)` - Begin recording
- `stop_recording()` - Stop and save recording
- `add_frame(frame)` - Add frame to video
- `_record_audio()` - Audio recording thread function
- `_merge_audio_video()` - Merge audio and video files

---

### MorphUI
**Purpose**: Manages user interface elements and interactions

**Key Attributes**:
- `window_name: str` - OpenCV window name
- `active_target_index: int` - Currently selected target
- `targets: List[TargetImage]` - Available targets
- `alpha: float` - Blend factor (0-1)
- `on_target_change: Callable` - Callback function

**Key Methods**:
- `draw_icon_bar(frame)` - Draw target selection icons
- `draw_recording_indicator(frame)` - Draw REC indicator
- `show_frame(frame)` - Display frame
- `_mouse_callback(event, x, y)` - Handle mouse clicks
- `set_target_change_callback(callback)` - Set callback for target changes

---

## Data Flow

```
┌──────────┐
│ Webcam   │
└────┬─────┘
     │ frame
     ▼
┌──────────────────┐
│ MorphApplication │
│  process_frame() │
└────┬─────────────┘
     │ frame + RGB conversion
     ▼
┌──────────────┐
│ MediaPipe    │
│ Face Mesh    │
└────┬─────────┘
     │ face_landmarks
     ▼
┌──────────────┐        ┌─────────────┐
│ FaceMorpher  │◄───────┤ TargetImage │
│ morph_face() │        │ (active)    │
└────┬─────────┘        └─────────────┘
     │ morphed_frame
     ▼
┌──────────────┐        ┌──────────────┐
│ VideoRecorder│        │   MorphUI    │
│  add_frame() │        │ draw_ui()    │
└──────────────┘        └─────┬────────┘
                              │ display_frame
                              ▼
                        ┌──────────┐
                        │ Screen   │
                        └──────────┘
```

## Workflow

### Application Startup
1. `MorphApplication.__init__()` creates all components
2. `TargetImage.load_all_from_folder()` loads targets
3. `MorphUI` creates window and trackbar
4. `FaceMorpher` computes triangulation for first target
5. Webcam and MediaPipe initialized

### Main Loop (each frame)
1. Read frame from webcam
2. Convert to RGB and detect faces with MediaPipe
3. For each detected face:
   - `FaceMorpher.morph_face()` applies morphing
   - Uses Delaunay triangulation
   - Warps each triangle
   - Creates masks to preserve eyes/mouth
   - Blends morphed with original
4. If recording, `VideoRecorder.add_frame()`
5. `MorphUI.draw_icon_bar()` adds UI elements
6. Display frame

### User Interactions
- **Click icon**: `MorphUI._mouse_callback()` → `MorphApplication._on_target_change()` → `FaceMorpher.update_target()`
- **Press 'R'**: Toggle `VideoRecorder.start/stop_recording()`
- **Move slider**: `MorphUI._on_alpha_change()` updates blend factor
- **Press ESC**: Exit main loop → `MorphApplication.cleanup()`

---

## Benefits of This Architecture

### 1. Single Responsibility Principle
Each class has one clear job:
- **TargetImage**: Manage target data
- **FaceMorpher**: Apply morphing algorithm
- **VideoRecorder**: Handle recording
- **MorphUI**: Manage interface
- **MorphApplication**: Coordinate everything

### 2. Open/Closed Principle
Easy to extend without modifying existing code:
- Add new morph algorithm → Create new class inheriting from `FaceMorpher`
- Add new UI → Create new class implementing same interface
- Add new recording format → Extend `VideoRecorder`

### 3. Dependency Injection
Components receive dependencies through constructors:
- Easy to test with mock objects
- Can swap implementations
- Clear dependencies

### 4. Testability
Each component can be tested independently:
```python
# Test FaceMorpher without UI or camera
target = TargetImage("test.png", "test.json")
morpher = FaceMorpher(target)
result = morpher.morph_face(test_frame, test_landmarks, alpha=0.5)
assert result.shape == test_frame.shape
```

### 5. Reusability
Components can be used in other projects:
- Use `FaceMorpher` in batch processing script
- Use `VideoRecorder` in different video app
- Use `TargetImage` loader in preprocessing tool

### 6. Maintainability
Changes are localized:
- Modify morphing algorithm → Only edit `FaceMorpher`
- Change UI layout → Only edit `MorphUI`
- Add recording features → Only edit `VideoRecorder`
