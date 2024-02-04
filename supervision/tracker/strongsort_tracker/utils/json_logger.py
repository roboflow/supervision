"""
References:
    https://medium.com/analytics-vidhya/creating-a-custom-logging-mechanism-for-real-time-object-detection-using-tdd-4ca2cfcd0a2f
"""
import json
from os import makedirs
from os.path import exists, join
from datetime import datetime


class JsonMeta(object):
    HOURS = 3
    MINUTES = 59
    SECONDS = 59
    PATH_TO_SAVE = 'LOGS'
    DEFAULT_FILE_NAME = 'remaining'


class BaseJsonLogger(object):
    """
    This is the base class that returns __dict__ of its own
    it also returns the dicts of objects in the attributes that are list instances

    """

    def dic(self):
        # returns dicts of objects
        out = {}
        for k, v in self.__dict__.items():
            if hasattr(v, 'dic'):
                out[k] = v.dic()
            elif isinstance(v, list):
                out[k] = self.list(v)
            else:
                out[k] = v
        return out

    @staticmethod
    def list(values):
        # applies the dic method on items in the list
        return [v.dic() if hasattr(v, 'dic') else v for v in values]


class Label(BaseJsonLogger):
    """
    For each bounding box there are various categories with confidences. Label class keeps track of that information.
    """

    def __init__(self, category: str, confidence: float):
        self.category = category
        self.confidence = confidence


class Bbox(BaseJsonLogger):
    """
    This module stores the information for each frame and use them in JsonParser
    Attributes:
        labels (list): List of label module.
        top (int):
        left (int):
        width (int):
        height (int):

    Args:
        bbox_id (float):
        top (int):
        left (int):
        width (int):
        height (int):

    References:
        Check Label module for better understanding.


    """

    def __init__(self, bbox_id, top, left, width, height):
        self.labels = []
        self.bbox_id = bbox_id
        self.top = top
        self.left = left
        self.width = width
        self.height = height

    def add_label(self, category, confidence):
        # adds category and confidence only if top_k is not exceeded.
        self.labels.append(Label(category, confidence))

    def labels_full(self, value):
        return len(self.labels) == value


class Frame(BaseJsonLogger):
    """
    This module stores the information for each frame and use them in JsonParser
    Attributes:
        timestamp (float): The elapsed time of captured frame
        frame_id (int): The frame number of the captured video
        bboxes (list of Bbox objects): Stores the list of bbox objects.

    References:
        Check Bbox class for better information

    Args:
        timestamp (float):
        frame_id (int):

    """

    def __init__(self, frame_id: int, timestamp: float = None):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.bboxes = []

    def add_bbox(self, bbox_id: int, top: int, left: int, width: int, height: int):
        bboxes_ids = [bbox.bbox_id for bbox in self.bboxes]
        if bbox_id not in bboxes_ids:
            self.bboxes.append(Bbox(bbox_id, top, left, width, height))
        else:
            raise ValueError("Frame with id: {} already has a Bbox with id: {}".format(self.frame_id, bbox_id))

    def add_label_to_bbox(self, bbox_id: int, category: str, confidence: float):
        bboxes = {bbox.id: bbox for bbox in self.bboxes}
        if bbox_id in bboxes.keys():
            res = bboxes.get(bbox_id)
            res.add_label(category, confidence)
        else:
            raise ValueError('the bbox with id: {} does not exists!'.format(bbox_id))


class BboxToJsonLogger(BaseJsonLogger):
    """
    ُ This module is designed to automate the task of logging jsons. An example json is used
    to show the contents of json file shortly
    Example:
          {
          "video_details": {
            "frame_width": 1920,
            "frame_height": 1080,
            "frame_rate": 20,
            "video_name": "/home/gpu/codes/MSD/pedestrian_2/project/public/camera1.avi"
          },
          "frames": [
            {
              "frame_id": 329,
              "timestamp": 3365.1254
              "bboxes": [
                {
                  "labels": [
                    {
                      "category": "pedestrian",
                      "confidence": 0.9
                    }
                  ],
                  "bbox_id": 0,
                  "top": 1257,
                  "left": 138,
                  "width": 68,
                  "height": 109
                }
              ]
            }],

    Attributes:
        frames (dict): It's a dictionary that maps each frame_id to json attributes.
        video_details (dict): information about video file.
        top_k_labels (int): shows the allowed number of labels
        start_time (datetime object): we use it to automate the json output by time.

    Args:
        top_k_labels (int): shows the allowed number of labels

    """

    def __init__(self, top_k_labels: int = 1):
        self.frames = {}
        self.video_details = self.video_details = dict(frame_width=None, frame_height=None, frame_rate=None,
                                                       video_name=None)
        self.top_k_labels = top_k_labels
        self.start_time = datetime.now()

    def set_top_k(self, value):
        self.top_k_labels = value

    def frame_exists(self, frame_id: int) -> bool:
        """
        Args:
            frame_id (int):

        Returns:
            bool: true if frame_id is recognized
        """
        return frame_id in self.frames.keys()

    def add_frame(self, frame_id: int, timestamp: float = None) -> None:
        """
        Args:
            frame_id (int):
            timestamp (float): opencv captured frame time property

        Raises:
             ValueError: if frame_id would not exist in class frames attribute

        Returns:
            None

        """
        if not self.frame_exists(frame_id):
            self.frames[frame_id] = Frame(frame_id, timestamp)
        else:
            raise ValueError("Frame id: {} already exists".format(frame_id))

    def bbox_exists(self, frame_id: int, bbox_id: int) -> bool:
        """
        Args:
            frame_id:
            bbox_id:

        Returns:
            bool: if bbox exists in frame bboxes list
        """
        bboxes = []
        if self.frame_exists(frame_id=frame_id):
            bboxes = [bbox.bbox_id for bbox in self.frames[frame_id].bboxes]
        return bbox_id in bboxes

    def find_bbox(self, frame_id: int, bbox_id: int):
        """

        Args:
            frame_id:
            bbox_id:

        Returns:
            bbox_id (int):

        Raises:
            ValueError: if bbox_id does not exist in the bbox list of specific frame.
        """
        if not self.bbox_exists(frame_id, bbox_id):
            raise ValueError("frame with id: {} does not contain bbox with id: {}".format(frame_id, bbox_id))
        bboxes = {bbox.bbox_id: bbox for bbox in self.frames[frame_id].bboxes}
        return bboxes.get(bbox_id)

    def add_bbox_to_frame(self, frame_id: int, bbox_id: int, top: int, left: int, width: int, height: int) -> None:
        """

        Args:
            frame_id (int):
            bbox_id (int):
            top (int):
            left (int):
            width (int):
            height (int):

        Returns:
            None

        Raises:
            ValueError: if bbox_id already exist in frame information with frame_id
            ValueError: if frame_id does not exist in frames attribute
        """
        if self.frame_exists(frame_id):
            frame = self.frames[frame_id]
            if not self.bbox_exists(frame_id, bbox_id):
                frame.add_bbox(bbox_id, top, left, width, height)
            else:
                raise ValueError(
                    "frame with frame_id: {} already contains the bbox with id: {} ".format(frame_id, bbox_id))
        else:
            raise ValueError("frame with frame_id: {} does not exist".format(frame_id))

    def add_label_to_bbox(self, frame_id: int, bbox_id: int, category: str, confidence: float):
        """
        Args:
            frame_id:
            bbox_id:
            category:
            confidence: the confidence value returned from yolo detection

        Returns:
            None

        Raises:
            ValueError: if labels quota (top_k_labels) exceeds.
        """
        bbox = self.find_bbox(frame_id, bbox_id)
        if not bbox.labels_full(self.top_k_labels):
            bbox.add_label(category, confidence)
        else:
            raise ValueError("labels in frame_id: {}, bbox_id: {} is fulled".format(frame_id, bbox_id))

    def add_video_details(self, frame_width: int = None, frame_height: int = None, frame_rate: int = None,
                          video_name: str = None):
        self.video_details['frame_width'] = frame_width
        self.video_details['frame_height'] = frame_height
        self.video_details['frame_rate'] = frame_rate
        self.video_details['video_name'] = video_name

    def output(self):
        output = {'video_details': self.video_details}
        result = list(self.frames.values())
        output['frames'] = [item.dic() for item in result]
        return output

    def json_output(self, output_name):
        """
        Args:
            output_name:

        Returns:
            None

        Notes:
            It creates the json output with `output_name` name.
        """
        if not output_name.endswith('.json'):
            output_name += '.json'
        with open(output_name, 'w') as file:
            json.dump(self.output(), file)
        file.close()

    def set_start(self):
        self.start_time = datetime.now()

    def schedule_output_by_time(self, output_dir=JsonMeta.PATH_TO_SAVE, hours: int = 0, minutes: int = 0,
                                seconds: int = 60) -> None:
        """
        Notes:
            Creates folder and then periodically stores the jsons on that address.

        Args:
            output_dir (str): the directory where output files will be stored
            hours (int):
            minutes (int):
            seconds (int):

        Returns:
            None

        """
        end = datetime.now()
        interval = 0
        interval += abs(min([hours, JsonMeta.HOURS]) * 3600)
        interval += abs(min([minutes, JsonMeta.MINUTES]) * 60)
        interval += abs(min([seconds, JsonMeta.SECONDS]))
        diff = (end - self.start_time).seconds

        if diff > interval:
            output_name = self.start_time.strftime('%Y-%m-%d %H-%M-%S') + '.json'
            if not exists(output_dir):
                makedirs(output_dir)
            output = join(output_dir, output_name)
            self.json_output(output_name=output)
            self.frames = {}
            self.start_time = datetime.now()

    def schedule_output_by_frames(self, frames_quota, frame_counter, output_dir=JsonMeta.PATH_TO_SAVE):
        """
        saves as the number of frames quota increases higher.
        :param frames_quota:
        :param frame_counter:
        :param output_dir:
        :return:
        """
        pass

    def flush(self, output_dir):
        """
        Notes:
            We use this function to output jsons whenever possible.
            like the time that we exit the while loop of opencv.

        Args:
            output_dir:

        Returns:
            None

        """
        filename = self.start_time.strftime('%Y-%m-%d %H-%M-%S') + '-remaining.json'
        output = join(output_dir, filename)
        self.json_output(output_name=output)
