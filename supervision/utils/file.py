import json
import csv
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from supervision.detection.core import Detections

import numpy as np
import yaml

class JSONSink:
    """
    A utility class for saving detection data to a JSON file. This class is designed to
    efficiently serialize detection objects into a JSON format, allowing for the inclusion of
    bounding box coordinates and additional attributes like confidence, class ID, and tracker ID.
    
    The class supports the capability to include custom data alongside the detection fields,
    providing flexibility for logging various types of information in a structured JSON format.
    
    Args:
        filename (str): The name of the JSON file where the detections will be stored.
                        Defaults to 'output.json'.
    
    Usage:
        ```python
        from supervision.utils.detections import Detections
        # Initialize JSONSink with a filename
        json_sink = JSONSink('my_detections.json')
        
        # Assuming detections is an instance of Detections containing detection data
        detections = Detections(...)
        
        # Open the JSONSink context, append detection data, and close the file automatically
        with json_sink as sink:
            sink.append(detections, custom_data={'frame': 1})
        ```
    """
    def __init__(self, filename: str = 'output.json'):
        self.filename: str = filename
        self.file: Optional[open] = None
        self.data: List[Dict[str, Any]] = []

    def __enter__(self) -> 'JSONSink':
        self.open()
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        self.write_and_close()

    def open(self) -> None:
        self.file = open(self.filename, 'w')

    def write_and_close(self) -> None:
        if self.file:
            json.dump(self.data, self.file, indent=4)
            self.file.close()

    def append(self, detections: Detections, custom_data: Dict[str, Any] = None) -> None:
        for i in range(len(detections.xyxy)):
            detection_data = {
                'x_min': int(detections.xyxy[i][0]),
                'y_min': int(detections.xyxy[i][1]),
                'x_max': int(detections.xyxy[i][2]),
                'y_max': int(detections.xyxy[i][3]),
                'class_id': int(detections.class_id[i]),
                'confidence': float(detections.confidence[i]),
                'tracker_id': int(detections.tracker_id[i])
            }
            
            for key, value in detections.data.items():
                detection_data[key] = value[i] if hasattr(value, '__getitem__') else value
            if custom_data:
                detection_data.update(custom_data)
                
            self.data.append(detection_data)

class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJsonEncoder, self).default(obj)


def list_files_with_extensions(
    directory: Union[str, Path], extensions: Optional[List[str]] = None
) -> List[Path]:
    """
    List files in a directory with specified extensions or
        all files if no extensions are provided.

    Args:
        directory (Union[str, Path]): The directory path as a string or Path object.
        extensions (Optional[List[str]]): A list of file extensions to filter.
            Default is None, which lists all files.

    Returns:
        (List[Path]): A list of Path objects for the matching files.

    Examples:
        ```python
        import supervision as sv

        # List all files in the directory
        files = sv.list_files_with_extensions(directory='my_directory')

        # List only files with '.txt' and '.md' extensions
        files = sv.list_files_with_extensions(
            directory='my_directory', extensions=['txt', 'md'])
        ```
    """

    directory = Path(directory)
    files_with_extensions = []

    if extensions is not None:
        for ext in extensions:
            files_with_extensions.extend(directory.glob(f"*.{ext}"))
    else:
        files_with_extensions.extend(directory.glob("*"))

    return files_with_extensions


def read_txt_file(file_path: str, skip_empty: bool = False) -> List[str]:
    """
    Read a text file and return a list of strings without newline characters.
    Optionally skip empty lines.

    Args:
        file_path (str): The path to the text file.
        skip_empty (bool): If True, skip lines that are empty or contain only
            whitespace. Default is False.

    Returns:
        List[str]: A list of strings representing the lines in the text file.
    """
    with open(file_path, "r") as file:
        if skip_empty:
            lines = [line.rstrip("\n") for line in file if line.strip()]
        else:
            lines = [line.rstrip("\n") for line in file]

    return lines


def save_text_file(lines: List[str], file_path: str):
    """
    Write a list of strings to a text file, each string on a new line.

    Args:
        lines (List[str]): The list of strings to be written to the file.
        file_path (str): The path to the text file.
    """
    with open(file_path, "w") as file:
        for line in lines:
            file.write(line + "\n")


def read_json_file(file_path: str) -> dict:
    """
    Read a json file and return a dict.

    Args:
        file_path (str): The path to the json file.

    Returns:
        dict: A dict of annotations information
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def save_json_file(data: dict, file_path: str, indent: int = 3) -> None:
    """
    Write a dict to a json file.

    Args:
        indent:
        data (dict): dict with unique keys and value as pair.
        file_path (str): The path to the json file.
    """
    with open(file_path, "w") as fp:
        json.dump(data, fp, cls=NumpyJsonEncoder, indent=indent)


def read_yaml_file(file_path: str) -> dict:
    """
    Read a yaml file and return a dict.

    Args:
        file_path (str): The path to the yaml file.

    Returns:
        dict: A dict of content information
    """
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def save_yaml_file(data: dict, file_path: str) -> None:
    """
    Save a dict to a json file.

    Args:
        indent:
        data (dict): dict with unique keys and value as pair.
        file_path (str): The path to the json file.
    """

    with open(file_path, "w") as outfile:
        yaml.dump(data, outfile, sort_keys=False, default_flow_style=None)
