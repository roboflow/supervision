from typing import Generator

import cv2


def get_video_frames_generator(video_path: str) -> Generator[int, None, None]:
    """
    Returns a generator that yields the frames of the video.

    :param video_path: str : The path of the video file.
    :return: Generator[int, None, None] : Generator that yields the frames of the video.
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise Exception(f"Could not open video at {video_path}")
    success, frame = video.read()
    while success:
        yield frame
        success, frame = video.read()
    video.release()
