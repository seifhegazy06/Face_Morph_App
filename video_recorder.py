"""
VideoRecorder class - Manages video and audio recording with automatic merging.
"""
import cv2
import pyaudio
import wave
import threading
import time
import os
from moviepy import VideoFileClip, AudioFileClip


class VideoRecorder:
    """Handles video and audio recording, then merges them into a single file."""
    
    def __init__(self, output_folder="Recordings", fps=20.0):
        """
        Initialize the video recorder.
        
        Args:
            output_folder: Folder to save recordings
            fps: Frames per second for video recording
        """
        self.output_folder = output_folder
        self.fps = fps
        self.is_recording = False
        
        # Video recording
        self.video_writer = None
        self.temp_video_path = "temp_video.mp4"
        self.frame_width = None
        self.frame_height = None
        
        # Audio recording
        self.audio_frames = []
        self.audio_thread = None
        self.audio_stream = None
        self.p_audio = pyaudio.PyAudio()
        
        # Audio settings
        self.audio_chunk = 1024
        self.audio_format = pyaudio.paInt16
        self.audio_channels = 1
        self.audio_rate = 44100
        
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
    
    def start_recording(self, frame_width, frame_height):
        """
        Start recording video and audio.
        
        Args:
            frame_width: Width of video frames
            frame_height: Height of video frames
        """
        if self.is_recording:
            print("Already recording!")
            return
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Start video recording
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.temp_video_path, 
            fourcc, 
            self.fps, 
            (frame_width, frame_height)
        )
        
        # Start audio recording in separate thread
        self.is_recording = True
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.start()
        
        print(f"🔴 Recording started (with audio)")
    
    def stop_recording(self):
        """Stop recording and merge audio/video into final file."""
        if not self.is_recording:
            print("Not recording!")
            return None
        
        # Stop recording flag
        self.is_recording = False
        
        # Release video writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # Wait for audio thread to finish
        if self.audio_thread is not None:
            self.audio_thread.join()
        
        # Stop audio stream
        if self.audio_stream is not None:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        print("⏹️  Recording stopped. Processing...")
        
        # Generate output filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_output = os.path.join(self.output_folder, f"morph_{timestamp}.mp4")
        
        # Merge audio and video if audio was captured
        if len(self.audio_frames) > 0:
            temp_audio_path = "temp_audio.wav"
            
            try:
                # Save audio to temporary WAV file
                self._save_audio(temp_audio_path)
                
                # Merge audio and video
                self._merge_audio_video(self.temp_video_path, temp_audio_path, final_output)
                
                # Clean up temporary files
                if os.path.exists(self.temp_video_path):
                    os.remove(self.temp_video_path)
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                
                print(f"✅ Video with audio saved: {final_output}")
                
            except Exception as e:
                print(f"⚠️  Audio merge failed: {e}")
                
                # Save video without audio as fallback
                final_output = os.path.join(self.output_folder, f"morph_{timestamp}_no_audio.mp4")
                if os.path.exists(self.temp_video_path):
                    os.rename(self.temp_video_path, final_output)
                    print(f"✅ Video saved (no audio): {final_output}")
        else:
            # No audio captured, just move video
            final_output = os.path.join(self.output_folder, f"morph_{timestamp}_no_audio.mp4")
            if os.path.exists(self.temp_video_path):
                os.rename(self.temp_video_path, final_output)
                print(f"✅ Video saved (no audio captured): {final_output}")
        
        return final_output
    
    def add_frame(self, frame):
        """
        Add a frame to the recording.
        
        Args:
            frame: BGR frame to add
        """
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)
    
    def _record_audio(self):
        """Record audio in separate thread (runs while is_recording is True)."""
        try:
            self.audio_stream = self.p_audio.open(
                format=self.audio_format,
                channels=self.audio_channels,
                rate=self.audio_rate,
                input=True,
                frames_per_buffer=self.audio_chunk
            )
            
            self.audio_frames = []
            
            while self.is_recording:
                try:
                    data = self.audio_stream.read(self.audio_chunk, exception_on_overflow=False)
                    self.audio_frames.append(data)
                except Exception as e:
                    print(f"Audio recording error: {e}")
                    break
        except Exception as e:
            print(f"Failed to start audio recording: {e}")
    
    def _save_audio(self, output_path):
        """Save recorded audio frames to WAV file."""
        print(f"Saving audio to {output_path}...")
        
        wf = wave.open(output_path, 'wb')
        wf.setnchannels(self.audio_channels)
        wf.setsampwidth(self.p_audio.get_sample_size(self.audio_format))
        wf.setframerate(self.audio_rate)
        wf.writeframes(b''.join(self.audio_frames))
        wf.close()
        
        print(f"Audio saved. Size: {os.path.getsize(output_path)} bytes")
    
    def _merge_audio_video(self, video_path, audio_path, output_path):
        """Merge audio and video files using moviepy."""
        print("Merging audio and video...")
        
        video_clip = VideoFileClip(video_path)
        print(f"Video loaded: {video_clip.duration}s, {video_clip.fps} fps")
        
        audio_clip = AudioFileClip(audio_path)
        print(f"Audio loaded: {audio_clip.duration}s")
        
        final_clip = video_clip.with_audio(audio_clip)
        print(f"Writing merged video to {output_path}...")
        
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', logger=None)
        
        # Clean up clips
        final_clip.close()
        video_clip.close()
        audio_clip.close()
    
    def cleanup(self):
        """Clean up audio resources."""
        if self.is_recording:
            self.stop_recording()
        
        self.p_audio.terminate()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass
