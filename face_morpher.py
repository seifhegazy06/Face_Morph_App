"""
FaceMorpher class - Handles face morphing operations using Delaunay triangulation.
"""
import cv2
import numpy as np
from scipy.spatial import Delaunay


class FaceMorpher:
    """Handles face morphing from target image to live face landmarks."""
    
    def __init__(self, target):
        """
        Initialize the face morpher with a target image.
        
        Args:
            target: TargetImage object containing the morph target
        """
        self.target = target
        self.triangles = self._compute_triangulation()
    
    def _compute_triangulation(self):
        """Compute Delaunay triangulation for the target landmarks."""
        tri = Delaunay(self.target.pts)
        return tri.simplices
    
    def update_target(self, new_target):
        """
        Update the morph target.
        
        Args:
            new_target: New TargetImage object
        """
        self.target = new_target
        self.triangles = self._compute_triangulation()
    
    def morph_face(self, frame, face_landmarks, alpha=0.5, preserve_eyes=True, preserve_mouth=True):
        """
        Morph a face in the frame using the target image.
        
        Args:
            frame: Input frame (BGR image)
            face_landmarks: MediaPipe face landmarks
            alpha: Blend factor (0=original, 1=full morph)
            preserve_eyes: If True, preserve real eyes
            preserve_mouth: If True, preserve real mouth/teeth
            
        Returns:
            Morphed frame
        """
        frame_h, frame_w = frame.shape[:2]
        
        # Convert MediaPipe landmarks to pixel coordinates
        src_pts = self._landmarks_to_points(face_landmarks, frame_w, frame_h)
        
        # Create warped target
        warped_target = np.zeros_like(frame)
        
        # Warp each triangle from target to source
        for tri_idx in self.triangles:
            t_tgt = self.target.pts[tri_idx]
            t_src = src_pts[tri_idx]
            self._warp_triangle(self.target.img, warped_target, t_tgt, t_src)
        
        # Create face mask for blending
        face_mask = self._create_face_mask(src_pts, frame_w, frame_h)
        
        # Exclude eyes and mouth if requested
        if preserve_eyes:
            eyes_mask = self._create_eyes_mask(src_pts, frame_w, frame_h)
            face_mask = cv2.subtract(face_mask, eyes_mask)
        
        if preserve_mouth:
            mouth_mask = self._create_mouth_mask(src_pts, frame_w, frame_h)
            face_mask = cv2.subtract(face_mask, mouth_mask)
        
        # Blur mask edges for smooth blending
        face_mask = cv2.GaussianBlur(face_mask, (21, 21), 11)
        face_mask_3ch = cv2.merge([face_mask, face_mask, face_mask]) / 255.0
        
        # Alpha blend
        face_blend = cv2.addWeighted(frame, 1 - alpha, warped_target, alpha, 0)
        
        # Combine: blended face in mask area, original frame elsewhere
        result = (face_blend * face_mask_3ch + frame * (1 - face_mask_3ch)).astype(np.uint8)
        
        # Force original eyes and mouth to show (no blending)
        if preserve_eyes or preserve_mouth:
            final_exclusion_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            
            if preserve_eyes:
                left_eye_pts, right_eye_pts = self._get_eye_points(src_pts)
                if left_eye_pts is not None:
                    cv2.fillConvexPoly(final_exclusion_mask, cv2.convexHull(left_eye_pts), 255)
                if right_eye_pts is not None:
                    cv2.fillConvexPoly(final_exclusion_mask, cv2.convexHull(right_eye_pts), 255)
            
            if preserve_mouth:
                mouth_pts = self._get_mouth_points(src_pts)
                if mouth_pts is not None:
                    cv2.fillConvexPoly(final_exclusion_mask, cv2.convexHull(mouth_pts), 255)
            
            # Expand to ensure full coverage
            kernel = np.ones((5, 5), np.uint8)
            final_exclusion_mask = cv2.dilate(final_exclusion_mask, kernel, iterations=2)
            
            # Convert to 3 channels
            final_exclusion_mask_3ch = cv2.merge([final_exclusion_mask] * 3) / 255.0
            
            # Replace with original frame in these areas
            result = (frame * final_exclusion_mask_3ch + result * (1 - final_exclusion_mask_3ch)).astype(np.uint8)
        
        return result
    
    def _landmarks_to_points(self, face_landmarks, width, height):
        """Convert MediaPipe landmarks to pixel coordinates."""
        pts = []
        for lm in face_landmarks.landmark:
            pts.append([int(lm.x * width), int(lm.y * height)])
        return np.array(pts, dtype=np.int32)
    
    def _warp_triangle(self, img_src, img_dst, t_src, t_dst):
        """Warp one triangle from source to destination."""
        r1 = cv2.boundingRect(np.float32([t_src]))
        r2 = cv2.boundingRect(np.float32([t_dst]))
        
        # Safety checks
        if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
            return
        if r1[0] < 0 or r1[1] < 0 or r1[0]+r1[2] > img_src.shape[1] or r1[1]+r1[3] > img_src.shape[0]:
            return
        if r2[0] < 0 or r2[1] < 0 or r2[0]+r2[2] > img_dst.shape[1] or r2[1]+r2[3] > img_dst.shape[0]:
            return
        
        t1_rect = []
        t2_rect = []
        
        for i in range(3):
            t1_rect.append((t_src[i][0] - r1[0], t_src[i][1] - r1[1]))
            t2_rect.append((t_dst[i][0] - r2[0], t_dst[i][1] - r2[1]))
        
        src_crop = img_src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
        
        if src_crop.size == 0:
            return
        
        M = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
        warped = cv2.warpAffine(src_crop, M, (r2[2], r2[3]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_rect), (1, 1, 1), 16, 0)
        
        dst_area = img_dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]].astype(np.float32)
        
        if dst_area.shape != warped.shape or dst_area.shape != mask.shape:
            return
        
        dst_area = dst_area * (1 - mask) + warped.astype(np.float32) * mask
        img_dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = np.clip(dst_area, 0, 255).astype(np.uint8)
    
    def _create_face_mask(self, src_pts, width, height):
        """Create a mask for the face area."""
        mask = np.zeros((height, width), dtype=np.uint8)
        hull = cv2.convexHull(src_pts)
        cv2.fillConvexPoly(mask, hull, 255)
        return mask
    
    def _create_eyes_mask(self, src_pts, width, height):
        """Create a mask for the eyes area."""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        left_eye_pts, right_eye_pts = self._get_eye_points(src_pts)
        
        if left_eye_pts is not None:
            cv2.fillConvexPoly(mask, cv2.convexHull(left_eye_pts), 255)
        
        if right_eye_pts is not None:
            cv2.fillConvexPoly(mask, cv2.convexHull(right_eye_pts), 255)
        
        # Expand mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        return mask
    
    def _create_mouth_mask(self, src_pts, width, height):
        """Create a mask for the mouth/teeth area."""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        mouth_pts = self._get_mouth_points(src_pts)
        
        if mouth_pts is not None:
            cv2.fillConvexPoly(mask, cv2.convexHull(mouth_pts), 255)
            
            # Expand mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def _get_eye_points(self, src_pts):
        """Get left and right eye landmark points."""
        left_eye_indices = [
            33, 7, 163, 144, 145, 153, 154, 155, 133,
            173, 157, 158, 159, 160, 161, 246
        ]
        
        right_eye_indices = [
            362, 382, 381, 380, 374, 373, 390, 249,
            263, 466, 388, 387, 386, 385, 384, 398
        ]
        
        if len(src_pts) > max(left_eye_indices + right_eye_indices):
            left_eye_pts = src_pts[left_eye_indices]
            right_eye_pts = src_pts[right_eye_indices]
            return left_eye_pts, right_eye_pts
        
        return None, None
    
    def _get_mouth_points(self, src_pts):
        """Get mouth/teeth landmark points."""
        teeth_indices = [
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308
        ]
        
        if len(src_pts) > max(teeth_indices):
            return src_pts[teeth_indices]
        
        return None
