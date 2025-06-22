"""Skeleton extractor."""

# src/sign_flow/skeleton_extractor.py
from typing import List, Dict, Optional
from pathlib import Path
import json
import cv2
import mediapipe as mp
import numpy as np


class SkeletonExtractor:
    """SkeletonExtractor Class."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """Initialize MediaPipe Pose model."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Upper body landmarks indices (from waist up)
        # 11,12: shoulders, 13,14: elbows, 15,16: wrists, 23,24: hips
        self.upper_body_indices = [11, 12, 13, 14, 15, 16, 23, 24]

        # Define upper body connections (only between our saved landmarks)
        self.upper_body_connections = [
            (11, 12),  # left shoulder - right shoulder
            (11, 13),  # left shoulder - left elbow
            (13, 15),  # left elbow - left wrist
            (12, 14),  # right shoulder - right elbow
            (14, 16),  # right elbow - right wrist
            (11, 23),  # left shoulder - left hip
            (12, 24),  # right shoulder - right hip
            (23, 24),  # left hip - right hip
        ]

    def extract_frame_skeleton(self, frame):
        """Extract upper body skeleton from a frame and return raw results."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            upper_body_landmarks = {}
            for idx in self.upper_body_indices:
                lm = results.pose_landmarks.landmark[idx]
                upper_body_landmarks[idx] = (lm.x, lm.y, lm.z, lm.visibility)
            return upper_body_landmarks, results
        return None, results

    def extract_video_skeletons(self, video_path: str) -> List[Optional[Dict]]:
        """Extract skeletons from all frames in video."""
        print(f"🔍 Extracting skeletons from: {Path(video_path).name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frames_skeletons = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            skeleton, _ = self.extract_frame_skeleton(frame)
            frames_skeletons.append(skeleton)
            frame_count += 1

            if frame_count % 10 == 0:  # Progress indicator
                print(f"   Processed {frame_count} frames...")

        cap.release()

        # Count successful detections
        successful_detections = sum(
            1 for s in frames_skeletons if s is not None
        )
        success_rate = (successful_detections / len(frames_skeletons)) * 100

        print(
            f"✅ Extracted {len(frames_skeletons)} frames, "
            f"{successful_detections} with skeletons "
            f"({success_rate:.1f}%)"
        )
        return frames_skeletons

    def save_skeletons(
        self, skeletons: List[Optional[Dict]], output_path: str
    ):
        """Save skeletons to JSON file."""
        with open(output_path, "w") as f:
            json.dump(skeletons, f, indent=2)
        print(f"💾 Saved skeletons to: {Path(output_path).name}")

    def save_last_frame_with_skeleton(
        self, video_path: str, output_path: str
    ) -> bool:
        """Save the last frame of video with upper body skeleton overlay."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return False

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            print(f"❌ Video has no frames: {video_path}")
            cap.release()
            return False

        # Set position to last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"❌ Cannot read last frame from: {video_path}")
            return False

        # Extract skeleton and draw only upper body
        skeleton, results = self.extract_frame_skeleton(frame)
        annotated_frame = self.draw_upper_body_skeleton(frame, results)

        # Save frame
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated_frame)

        # Print skeleton info
        if skeleton:
            print(
                f"✅ Saved last frame with upper body skeleton to: "
                f"{out_path.name}"
            )
        else:
            print(
                f"✅ Saved last frame (no skeleton detected) to: "
                f"{out_path.name}"
            )

        return True

    def draw_upper_body_skeleton(self, frame, results) -> np.ndarray:
        """Draw only upper body skeleton landmarks that we save."""
        annotated_frame = frame.copy()

        if not results.pose_landmarks:
            return annotated_frame

        # Get frame dimensions
        height, width = frame.shape[:2]
        landmarks = results.pose_landmarks.landmark

        # Draw only upper body landmarks
        for idx in self.upper_body_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                # Convert normalized coordinates to pixel coordinates
                x = int(lm.x * width)
                y = int(lm.y * height)

                # Draw landmark point
                cv2.circle(
                    annotated_frame, (x, y), 5, (0, 255, 0), -1
                )  # Green filled circle
                cv2.circle(
                    annotated_frame, (x, y), 6, (0, 0, 255), 2
                )  # Red outline

                # Add landmark index label
                cv2.putText(
                    annotated_frame,
                    str(idx),
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

        # Draw connections between upper body landmarks
        for start_idx, end_idx in self.upper_body_connections:
            if (
                start_idx < len(landmarks)
                and end_idx < len(landmarks)
                and landmarks[start_idx].visibility > 0.5
                and landmarks[end_idx].visibility > 0.5
            ):
                # Get start and end points
                start_x = int(landmarks[start_idx].x * width)
                start_y = int(landmarks[start_idx].y * height)
                end_x = int(landmarks[end_idx].x * width)
                end_y = int(landmarks[end_idx].y * height)

                # Draw connection line
                cv2.line(
                    annotated_frame,
                    (start_x, start_y),
                    (end_x, end_y),
                    (255, 0, 0),
                    2,
                )  # Blue line

        return annotated_frame

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, "pose"):
            self.pose.close()
