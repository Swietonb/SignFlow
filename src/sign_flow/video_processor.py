"""VideoProcessor."""

# src/sign_flow/video_processor.py
from pathlib import Path
import cv2
from .config.settings import VideoProcessingConfig
from .skeleton_extractor import SkeletonExtractor


class VideoProcessor:
    """VideoProcessor Class."""

    def __init__(self, config: VideoProcessingConfig):
        """Initialize Video Processor."""
        self.config = config
        self.video1_path = Path(config.video1_path)
        self.video2_path = Path(config.video2_path)
        self.output_dir = Path(config.output_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize skeleton extractor
        self.skeleton_extractor = SkeletonExtractor()

    def verify_videos(self) -> bool:
        """Verify videos."""
        for name, path in [
            ("Video 1", self.video1_path),
            ("Video 2", self.video2_path),
        ]:
            if not path.exists():
                print(f"❌ {path}: file does not exist")
                return False

            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                print(f"❌ {name}: cannot open file")
                cap.release()
                return False

            # Video information
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            cap.release()
            print(f"✅ {name}: OK")
            print(
                f"   📊 {width}x{height}, {frame_count} frames, {fps:.1f} FPS,"
                f"{duration:.1f}s"
            )

        print("✅ Verification completed!")
        return True

    def get_video_info(self) -> dict:
        """Return basic information about the videos."""
        info = {}
        for name, path in [
            ("video1", self.video1_path),
            ("video2", self.video2_path),
        ]:
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                info[name] = {
                    "fps": cap.get(cv2.CAP_PROP_FPS),
                    "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                }
                cap.release()
        return info

    def save_verification_frames(self) -> bool:
        """Save last frames with skeleton overlays for verification."""
        print("\n=== SAVING VERIFICATION FRAMES ===")

        # Save last frame from each video with skeleton overlay
        frame1_path = self.output_dir / "verification_last_frame_video1.jpg"
        frame2_path = self.output_dir / "verification_last_frame_video2.jpg"

        success1 = self.skeleton_extractor.save_last_frame_with_skeleton(
            str(self.video1_path), str(frame1_path)
        )
        success2 = self.skeleton_extractor.save_last_frame_with_skeleton(
            str(self.video2_path), str(frame2_path)
        )

        if success1 and success2:
            print("✅ Verification frames saved successfully!")
            print(f"   📸 Check: {frame1_path.name}")
            print(f"   📸 Check: {frame2_path.name}")
            return True

        print("❌ Failed to save some verification frames")
        return False

    def extract_skeletons(self) -> tuple[list, list]:
        """Extract skeletons from both videos."""
        print("\n=== SKELETON EXTRACTION ===")

        # Extract skeletons from both videos
        skeletons1 = self.skeleton_extractor.extract_video_skeletons(
            str(self.video1_path)
        )
        skeletons2 = self.skeleton_extractor.extract_video_skeletons(
            str(self.video2_path)
        )

        # Save to files
        skeleton1_path = self.output_dir / "skeletons_video1.json"
        skeleton2_path = self.output_dir / "skeletons_video2.json"

        self.skeleton_extractor.save_skeletons(skeletons1, str(skeleton1_path))
        self.skeleton_extractor.save_skeletons(skeletons2, str(skeleton2_path))

        return skeletons1, skeletons2
