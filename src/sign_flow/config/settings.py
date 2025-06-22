"""Settings."""

from dataclasses import dataclass


@dataclass
class VideoProcessingConfig:
    """Simple configuration for the video interpolation process."""

    video1_path: str = "vid/pociag.mp4"
    video2_path: str = "vid/do_stacji.mp4"
    output_dir: str = "output"
    n_frames_to_analyze: int = 5
    m_interpolated_frames: int = 5
