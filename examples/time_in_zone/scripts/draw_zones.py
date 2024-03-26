import argparse
import os
from typing import Any, Optional

import cv2
import numpy as np

import supervision as sv

KEY_ENTER = 13
KEY_NEWLINE = 10
KEY_ESCAPE = 27
KEY_QUIT = ord("q")

THICKNESS = 2
COLORS = sv.ColorPalette.default()
WINDOW_NAME = "Draw Zones"
POLYGONS = [[]]


def resolve_source(source_path: str) -> Optional[np.ndarray]:
    if not os.path.exists(source_path):
        return None

    image = cv2.imread(source_path)
    if image is not None:
        return image

    frame_generator = sv.get_video_frames_generator(source_path=source_path)
    frame = next(frame_generator)
    return frame


def click_event(event: int, x: int, y: int, flags: int, param: Any) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        POLYGONS[-1].append((x, y))
        print(f"Mouse coordinates: (X: {x}, Y: {y})")

        if len(POLYGONS[-1]) >= 2:
            cv2.line(
                img=param,
                pt1=POLYGONS[-1][-2],
                pt2=POLYGONS[-1][-1],
                color=COLORS.by_idx(0).as_bgr(),
                thickness=THICKNESS,
            )
        cv2.imshow(WINDOW_NAME, param)


def close_and_finalize_polygon(image: np.ndarray, original_image: np.ndarray) -> None:
    if len(POLYGONS[-1]) > 2:
        cv2.line(
            img=image,
            pt1=POLYGONS[-1][-1],
            pt2=POLYGONS[-1][0],
            color=COLORS.by_idx(0).as_bgr(),
            thickness=THICKNESS,
        )
    POLYGONS.append([])
    image[:] = original_image.copy()
    redraw_polygons(image)
    cv2.imshow(WINDOW_NAME, image)


def redraw_polygons(image: np.ndarray) -> None:
    for polygon in POLYGONS[:-1]:
        if len(polygon) > 1:
            for i in range(len(polygon) - 1):
                cv2.line(
                    img=image,
                    pt1=polygon[i],
                    pt2=polygon[i + 1],
                    color=COLORS.by_idx(0).as_bgr(),
                    thickness=THICKNESS,
                )
            cv2.line(
                img=image,
                pt1=polygon[-1],
                pt2=polygon[0],
                color=COLORS.by_idx(0).as_bgr(),
                thickness=THICKNESS,
            )


def main(source_path: str, target_path: str) -> None:
    original_image = resolve_source(source_path=source_path)
    if original_image is None:
        print("Failed to load source image.")
        return

    image = original_image.copy()
    cv2.imshow(WINDOW_NAME, image)
    cv2.setMouseCallback(WINDOW_NAME, click_event, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == KEY_ENTER or key == KEY_NEWLINE:
            close_and_finalize_polygon(image, original_image)
        elif key == KEY_ESCAPE:
            POLYGONS[-1] = []
            image[:] = original_image.copy()
            redraw_polygons(image)
            cv2.imshow(WINDOW_NAME, image)
        elif key == KEY_QUIT:
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
