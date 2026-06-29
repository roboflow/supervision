import json
import os

import cv2
import numpy as np
from jsonargparse import auto_cli

import supervision as sv

KEY_ENTER = {"Return", "KP_Enter"}
KEY_ESCAPE = "Escape"
KEY_QUIT = "q"
KEY_SAVE = "s"

THICKNESS = 2
COLORS = sv.ColorPalette.DEFAULT
WINDOW_NAME = "Draw Zones"
POLYGONS = [[]]

current_mouse_position: tuple[int, int] | None = None


def resolve_source(source_path: str) -> np.ndarray | None:
    if not os.path.exists(source_path):
        return None

    image = cv2.imread(source_path)
    if image is not None:
        return image

    frame_generator = sv.get_video_frames_generator(source_path=source_path)
    frame = next(frame_generator)
    return frame


def mouse_event(x: int, y: int, event_type: str) -> None:
    global current_mouse_position
    if event_type == "move":
        current_mouse_position = (x, y)
    elif event_type == "down":
        POLYGONS[-1].append((x, y))


def redraw(
    image: np.ndarray, original_image: np.ndarray, window: sv.TkImageWindow
) -> None:
    global POLYGONS, current_mouse_position
    image[:] = original_image.copy()
    for idx, polygon in enumerate(POLYGONS):
        color = (
            COLORS.by_idx(idx).as_bgr()
            if idx < len(POLYGONS) - 1
            else sv.Color.WHITE.as_bgr()
        )

        if len(polygon) > 1:
            for i in range(1, len(polygon)):
                cv2.line(
                    img=image,
                    pt1=polygon[i - 1],
                    pt2=polygon[i],
                    color=color,
                    thickness=THICKNESS,
                )
            if idx < len(POLYGONS) - 1:
                cv2.line(
                    img=image,
                    pt1=polygon[-1],
                    pt2=polygon[0],
                    color=color,
                    thickness=THICKNESS,
                )
        if idx == len(POLYGONS) - 1 and current_mouse_position is not None and polygon:
            cv2.line(
                img=image,
                pt1=polygon[-1],
                pt2=current_mouse_position,
                color=color,
                thickness=THICKNESS,
            )
    window.show(image)


def close_and_finalize_polygon(
    image: np.ndarray, original_image: np.ndarray, window: sv.TkImageWindow
) -> None:
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
    window.show(image)


def redraw_polygons(image: np.ndarray) -> None:
    for idx, polygon in enumerate(POLYGONS[:-1]):
        if len(polygon) > 1:
            color = COLORS.by_idx(idx).as_bgr()
            for i in range(len(polygon) - 1):
                cv2.line(
                    img=image,
                    pt1=polygon[i],
                    pt2=polygon[i + 1],
                    color=color,
                    thickness=THICKNESS,
                )
            cv2.line(
                img=image,
                pt1=polygon[-1],
                pt2=polygon[0],
                color=color,
                thickness=THICKNESS,
            )


def save_polygons_to_json(
    polygons: list[list[tuple[int, int]]], target_path: str | os.PathLike[str]
) -> None:
    data_to_save = polygons if polygons[-1] else polygons[:-1]
    with open(target_path, "w") as f:
        json.dump(data_to_save, f)


def main(source_path: str, zone_configuration_path: str) -> None:
    """
    Interactively draw polygons on images or video frames and save the annotations.

    Args:
        source_path: Path to the source image or video file for drawing polygons.
        zone_configuration_path: Path where the polygon annotations saved as JSON file.
    """
    global current_mouse_position
    original_image = resolve_source(source_path=source_path)
    if original_image is None:
        print("Failed to load source image.")
        return

    image = original_image.copy()
    window = sv.TkImageWindow(WINDOW_NAME)
    window.set_mouse_callback(mouse_event)
    window.show(image)

    while True:
        key = window.wait_key(1)
        if key in KEY_ENTER:
            close_and_finalize_polygon(image, original_image, window)
        elif key == KEY_ESCAPE:
            POLYGONS[-1] = []
            current_mouse_position = None
        elif key == KEY_SAVE:
            save_polygons_to_json(POLYGONS, zone_configuration_path)
            print(f"Polygons saved to {zone_configuration_path}")
            break
        redraw(image, original_image, window)
        if key == KEY_QUIT:
            break

    window.close()


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
