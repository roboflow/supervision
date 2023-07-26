# Tracker

A tracker is a computer vision algorithm that aims to track objects across consecutive frames in a video sequence. Its primary objective is to maintain the identity of objects as they move, even in challenging conditions such as occlusions and appearance changes. By establishing associations between detections in subsequent frames, trackers play a crucial role in various applications like surveillance, robotics, and video analytics, enabling real-time monitoring and behavior analysis.

## Table of Contents
- [Tracker](#tracker)
  - [Table of Contents](#table-of-contents)
  - [Byte Tracker](#byte-tracker)
    - [Introduction](#introduction)
    - [Usage](#usage)
    - [update from sv result:](#update-from-sv-result)
    - [update from numpy array:](#update-from-numpy-array)

## Byte Tracker

Byte Tracker is a visual object tracking method designed to track objects in video sequences. This folder contains the implementation of the Byte Tracker algorithm.


### Introduction

Byte Tracker is a visual object tracking algorithm that leverages a combination of motion and appearance features to track objects in videos. The algorithm is robust to object occlusions, scale changes, and abrupt motion.

The folder contains the implementation of the Byte Tracker algorithm, which can be used in conjunction with object detection algorithms to track objects over time.
   
### Usage

1. Import the byte_tracker module in your Python script or Jupyter Notebook:

   ```python
   from supervision.tracker.byte_tracker import byte_tracker
   ```
2. Initialize the Byte Tracker with suitable parameters:
   
   ```python
   tracker = byte_tracker(
      track_thresh=0.25, 
      track_buffer=30, 
      match_thresh=0.8, 
      aspect_ratio_thresh=3.0, 
      min_box_area=1.0, 
      mot20=False, 
      frame_rate=30
   )
   ```
3. Update the tracker with new detections and frames for each video frame:
### update from sv result:
   ```python
   detections_res = byte_tracker.update_from_detections(
      detections=sv_results, img_info=frame.shape, img_size=frame.shape
   )
   ```
### update from numpy array:
  
   ```python
   tracks = byte_tracker.update_from_numpy(
      output_results=detections2boxes(detections=detections),
      img_info=frame.shape,
      img_size=frame.shape,
   )
   ```