import json
from typing import Generator, List

import cv2
import numpy as np


def load_zones_config(file_path: str) -> List[np.ndarray]:
    """
    Load polygon zone configurations from a JSON file.

    This function reads a JSON file which contains polygon coordinates, and
    converts them into a list of NumPy arrays. Each polygon is represented as
    a NumPy array of coordinates.

    Args:
        file_path (str): The path to the JSON configuration file.

    Returns:
        List[np.ndarray]: A list of polygons, each represented as a NumPy array.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        return [np.array(polygon, np.int32) for polygon in data]


def find_in_list(array: np.ndarray, search_list: List[int]) -> np.ndarray:
    """Determines if elements of a numpy array are present in a list.

    Args:
        array (np.ndarray): The numpy array of integers to check.
        search_list (List[int]): The list of integers to search within.

    Returns:
        np.ndarray: A numpy array of booleans, where each boolean indicates whether
        the corresponding element in `array` is found in `search_list`.
    """
    if not search_list:
        return np.ones(array.shape, dtype=bool)
    else:
        return np.isin(array, search_list)


def get_stream_frames_generator(rtsp_url: str) -> Generator[np.ndarray, None, None]:
    """
    Generator function to yield frames from an RTSP stream.

    Args:
        rtsp_url (str): URL of the RTSP video stream.

    Yields:
        np.ndarray: The next frame from the video stream.
    """
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise Exception("Error: Could not open video stream.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or error reading frame.")
                break
            yield frame
    finally:
        cap.release()
