"""
Interactive SOURCE calibration tool for speed_estimation.

Usage:
    python calibrate_source.py --source_video_path <video_path>

Instructions:
    1. A window will open showing the first frame of the video.
    2. Click 4 points in order: top-left, top-right, bottom-right, bottom-left
       to define the road region for perspective transformation.
    3. The points should form a quadrilateral on the road surface,
       with the top edge being farther from the camera and the bottom edge closer.
    4. Press 'q' to quit, 'r' to reset points.
    5. After selecting 4 points, the result will be printed to console.

    Then update the SOURCE variable in ultralytics_example.py with the output.
"""

import argparse

import cv2
import numpy as np


def calibrate(video_path: str) -> None:
    """Open interactive window to pick 4 SOURCE points from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Cannot read first frame")
        return

    h, w = frame.shape[:2]
    print(f"Video resolution: {w}x{h}")

    points: list[tuple[int, int]] = []

    def redraw() -> None:
        """Redraw all points and lines on the frame."""
        display = frame.copy()
        for i, pt in enumerate(points):
            cv2.circle(display, pt, 8, (0, 255, 0), -1)
            cv2.putText(
                display,
                f"P{i+1}",
                (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        if len(points) > 1:
            cv2.polylines(
                display,
                [np.array(points, dtype=np.int32)],
                isClosed=False,
                color=(0, 255, 0),
                thickness=2,
            )
        if len(points) == 4:
            cv2.polylines(
                display,
                [np.array(points, dtype=np.int32)],
                isClosed=True,
                color=(0, 0, 255),
                thickness=2,
            )
        cv2.imshow("Calibrate SOURCE - Click 4 points", display)

    def mouse_callback(event: int, x: int, y: int, flags: int, param: None) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"  P{len(points)}: ({x}, {y})")
            redraw()

    cv2.namedWindow("Calibrate SOURCE - Click 4 points", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibrate SOURCE - Click 4 points", w, h)
    cv2.setMouseCallback("Calibrate SOURCE - Click 4 points", mouse_callback)
    cv2.imshow("Calibrate SOURCE - Click 4 points", frame)

    print("\nClick 4 points in order:")
    print("  P1: top-left     (far left of road)")
    print("  P2: top-right    (far right of road)")
    print("  P3: bottom-right (near right of road)")
    print("  P4: bottom-left  (near left of road)")
    print("\nPress 'r' to reset, 'q' to quit\n")

    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            points.clear()
            cv2.imshow("Calibrate SOURCE - Click 4 points", frame)
            print("\nPoints reset. Click again.\nClick 4 points in order:")
            print("  P1: top-left     (far left of road)")
            print("  P2: top-right    (far right of road)")
            print("  P3: bottom-right (near right of road)")
            print("  P4: bottom-left  (near left of road)")

    cv2.destroyAllWindows()

    if len(points) == 4:
        source_array = np.array(points)
        print("\n" + "=" * 60)
        print("Selected points summary:")
        for i, pt in enumerate(points):
            print(f"  P{i+1}: ({pt[0]}, {pt[1]})")
        print("=" * 60)
        print("Copy the following line to your ultralytics_example.py:")
        print("=" * 60)
        coords_str = ", ".join(
            [f"[{pt[0]}, {pt[1]}]" for pt in points]
        )
        print(f"SOURCE = np.array([{coords_str}])")
        print("=" * 60)

        # Also suggest TARGET_WIDTH and TARGET_HEIGHT
        # Calculate approximate real-world distance if user provides it
        print("\nNote: Also adjust TARGET_WIDTH and TARGET_HEIGHT")
        print("  TARGET_WIDTH  = real-world width of the road section (meters)")
        print("  TARGET_HEIGHT = real-world length of the road section (meters)")
    else:
        print(f"\nOnly {len(points)} points selected. Need exactly 4.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate SOURCE points for speed estimation"
    )
    parser.add_argument(
        "--source_video_path", required=True, help="Path to the source video"
    )
    args = parser.parse_args()
    calibrate(args.source_video_path)
