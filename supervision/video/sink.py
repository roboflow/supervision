import cv2
import numpy as np

from supervision.video.dataclasses import VideoInfo


class VideoSink:
    """
    A context manager that uses OpenCV to save video frames to a file.

    :param output_path: str : The path to the output file where the video will be saved.
    :param video_info: VideoInfo : An instance of VideoInfo containing information about the video resolution, fps, and total frame count.
    """

    def __init__(self, output_path: str, video_info: VideoInfo):
        """
        Initializes the VideoSink with the specified output path and video information.
        """
        self.output_path = output_path
        self.video_info = video_info
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = None

    def __enter__(self):
        """
        Opens the output file and returns the VideoSink instance.
        """
        self.writer = cv2.VideoWriter(
            self.output_path,
            self.fourcc,
            self.video_info.fps,
            self.video_info.resolution,
        )
        return self

    def write_frame(self, frame: np.ndarray):
        """
        Writes a frame to the output video file.

        :param frame: np.ndarray : The frame to be written.
        """
        self.writer.write(frame)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the output file.
        """
        self.writer.release()
