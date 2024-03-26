import argparse
import os
from typing import Any, Optional

import cv2
import numpy as np

import supervision as sv

THICKNESS = 2
COLORS = sv.ColorPalette.default()
WINDOW_NAME = "Draw Zones"
POINTS = []


def resolve_source(source_path: str) -> Optional[np.ndarray]:
    if not os.path.exists(source_path):
        return

    image = cv2.imread(source_path)
    if image is not None:
        return image
    else:
        frame_generator = sv.get_video_frames_generator(source_path=source_path)
        frame = next(frame_generator)
        return frame


def click_event(event: int, x: int, y: int, flags: int, param: Any) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        POINTS.append((x, y))
        print(f"Mouse coordinates: (X: {x}, Y: {y})")

        if len(POINTS) >= 2:
            cv2.line(
                img=param,
                pt1=POINTS[-2],
                pt2=POINTS[-1],
                color=COLORS.by_idx(0).as_bgr(),
                thickness=THICKNESS,
            )
        cv2.imshow(WINDOW_NAME, param)


def close_polygon(image: np.ndarray) -> None:
    if len(POINTS) >= 2:
        cv2.line(
            img=image,
            pt1=POINTS[-1],
            pt2=POINTS[0],
            color=COLORS.by_idx(0).as_bgr(),
            thickness=THICKNESS,
        )
        cv2.imshow(WINDOW_NAME, image)


def main(source_path: str, target_path: str) -> None:
    image = resolve_source(source_path=source_path)
    cv2.imshow(WINDOW_NAME, image)
    cv2.setMouseCallback(WINDOW_NAME, click_event, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 10:
            close_polygon(image)
        elif key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="...",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        required=True,
        help="...",
    )
    arguments = parser.parse_args()
    main(
        source_path=arguments.source_path,
        target_path=arguments.target_path,
    )
