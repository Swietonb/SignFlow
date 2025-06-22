"""Main module."""

# src/main.py
from src.sign_flow.config.settings import VideoProcessingConfig
from src.sign_flow.video_processor import VideoProcessor


def main():
    """Run main procedure."""
    print("🎬 MOTION INTERPOLATION IN SIGN LANGUAGE")
    print("=" * 45)

    # Simple configuration
    config = VideoProcessingConfig()

    # Paths can be easily modified if needed
    # config.video1_path = "../vid/another_film1.mp4"
    # config.video2_path = "../vid/another_film2.mp4"

    # Step 1: Verify video files
    processor = VideoProcessor(config)
    if not processor.verify_videos():
        print("❌ Cannot continue")
        return

    processor.save_verification_frames()

    try:
        skeletons1, skeletons2 = processor.extract_skeletons()
        print("🎯 Step 2 completed!")
        print(f"   Video 1: {len(skeletons1)} frames processed")
        print(f"   Video 2: {len(skeletons2)} frames processed")
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        return


if __name__ == "__main__":
    main()
