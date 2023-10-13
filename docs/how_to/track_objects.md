## How To: Track Objects
This is a tutorial to track object in a video using [ByteTrack](https://supervision.roboflow.com/trackers/#supervision.tracker.byte_tracker.core.ByteTrack).
### What is Tracking?
Tracking is an essential application of computer vision, which is used in mainly in videos or set of images which contain a sequence of events(the set of images could be simply frames of a video). It takes an initial set of detections from a video as per the requirement of the project, provides some unique identification to the detections and tracks as the position changes with time in a video. Supervision provides some powerful tools to track object in videos. This tutorial would cover how to make detections in a video, track those detections and then label them for unique identification using Supervision. Further, it would also cover how to trace the path covered by the detections.
Before proceeding remember to install Supervision:
'''
pip install supervision
'''
and import it:
'''
import supervision as sv
'''
### Step 1:Building a detection pipeline for videos
First to track objects we need to load the data and make it usable. We are going to do it by generating the frames from the video and using an object detection model, using [sv.get_video_frames_generator()](https://supervision.roboflow.com/utils/video/#get_video_frames_generator). In this tutorial we are going to do it using YOLOv8. 
First install ultralytics, import it and initialize the frame_generator.
'''
pip install ultralytics
'''
'''
from ultralytics import YOLO
model = YOLO(...)
frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
'''
Replace '...' with the path of v8 model
### Step 2:Tracking the detections
To track the detections we are going to use [ByteTrack](https://supervision.roboflow.com/trackers/#supervision.tracker.byte_tracker.core.ByteTrack). Further, the detections in each frame are going tbe passed into [sv.Detections.from_ultralytics()](https://supervision.roboflow.com/detection/core/), whose output would be passed on to ByteTrack.
'''
tracker = sv.ByteTrack()

for frame in frame_generator:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
'''
### Step 3:Label the detections being tracked
The objects being tracked can be annotated with any [annotator provided by supervision](https://supervision.roboflow.com/annotators/#labelannotator). But, we are going to use label annotator and label the object being tracked with their tracking ids.
'''
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER) 
for frame in frame_generator:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    labels = detections.tracker_id
    annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
'''
### Step 4:Tracing the path of the detections
Further, the path of the tracked objects can be traced using [TraceAnnotator](https://supervision.roboflow.com/annotators/#traceannotator) of supervision.
'''

label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER) 
trace_annotator = sv.TraceAnnotator()
for frame in frame_generator:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    labels = detections.tracker_id
    label_annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
    trace_annotator.annotate(scene=frame.copy(), detections=detections)
### Step 5:Integrating the previous steps and processing the video
At the end, we want to see the output on our video. In order to do that, we'll be passing the annotations from previous iterations into a callback to be used by [sv.process_video](https://supervision.roboflow.com/utils/video/#process_video).
'''
from ultralytics import YOLO
model = YOLO(...)
frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
tracker = sv.ByteTrack()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER) 
trace_annotator = sv.TraceAnnotator()
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    for frame in frame_generator:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        labels = detections.tracker_id
        label_annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
        trace_annotator.annotate(scene=frame.copy(), detections=detections)
sv.process_video(source_path=VIDEO_PATH, target_path=f"result.mp4", callback=process_frame)
'''
Results with label annotator and trace annotator are show below in sequential frames from a video.
### Conclusion and further ideas
This brings us to the end of the tutorial. The users are strongly encouraged to try out other models for detection and play around with annotators. Check out this [post](https://blog.roboflow.com/yolov8-tracking-and-counting/#object-tracking-with-bytetrack)("Piotr Skalski." Roboflow Blog, Feb 1, 2023. https://blog.roboflow.com/yolov8-tracking-and-counting/) for some more cool stuff. The post was quite useful creation of this tutorial.
