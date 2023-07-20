# Examples

This repository also includes example code that demonstrates the usage of the computer vision library. Each example is located in a separate file under the "examples" directory. You can find the installation instructions and usage details for each example in README file.

## Example 1: Object Tracking with Byte Track

The "example" folder contains a code example that utilizes the Byte Track algorithm for object tracking. It demonstrates how to perform object detection and tracking on a video using the YOLO model and the BYTETracker algorithm.

To run the example, follow these steps:

1. Navigate to the example folder:
```
cd examples
```
2. Install Ultralytics

3. Prepare the source video:

- Ensure that the source video is in a compatible format.

- Set the `SOURCE_VIDEO_PATH` variable in the code to the path of your source video file.

4. Run the code:
```
python3 tracker.py
```
   The code will process the video frames, perform object detection and tracking, and save the output video with annotated bounding boxes.

5. Find the output video:
   - The processed video with annotated bounding boxes will be saved as `TARGET_VIDEO_PATH` in the example folder.