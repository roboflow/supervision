from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import supervision as sv

DEFAULT_IMAGE_PATHS = (
    Path(__file__).with_name("input.jpg"),
    Path(__file__).with_name("input.png"),
    Path(__file__).with_name("input.jpeg"),
)
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("occupancy_result.jpg")
DEFAULT_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")


@dataclass(frozen=True)
class DetectionResult:
    label: str
    xyxy: list[float]
    confidence: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect objects in an image and calculate PolygonZone occupancy."
    )
    parser.add_argument(
        "image_path_arg",
        nargs="?",
        type=Path,
        help="Optional image path to test.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Path to the image to test.",
    )
    parser.add_argument(
        "--ask-path",
        action="store_true",
        help="Ask for the image path in PowerShell.",
    )
    parser.add_argument(
        "--choose-file",
        action="store_true",
        help="Open a Windows file picker to choose the image.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the annotated image will be saved.",
    )
    parser.add_argument(
        "--zone",
        choices=("full",),
        default="full",
        help="Use the full image as the occupancy zone.",
    )
    parser.add_argument(
        "--zone-rect",
        type=str,
        default=None,
        help='Zone rectangle as "x1,y1,x2,y2".',
    )
    parser.add_argument(
        "--zone-polygon",
        type=str,
        default=None,
        help='Zone polygon as "x1,y1 x2,y2 x3,y3 ...".',
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics YOLO model to use for automatic detections.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detector confidence.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="Object class names to count.",
    )
    parser.add_argument(
        "--manual-box",
        action="append",
        default=[],
        help='Manual fallback box as "label,x1,y1,x2,y2". Can be used more than once.',
    )
    parser.add_argument(
        "--skip-detector",
        action="store_true",
        help="Skip YOLO and use only --manual-box detections.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Draw the zone and object boxes with the mouse.",
    )
    parser.add_argument(
        "--no-interactive-fallback",
        action="store_true",
        help="Do not open the drawing tool when automatic detection finds no objects.",
    )
    return parser.parse_args()


def clean_path(raw_path: str) -> Path:
    return Path(raw_path.strip().strip("&").strip().strip('"').strip("'"))


def choose_image_path() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Choose an image to test",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
    except tk.TclError:
        return None

    return Path(selected) if selected else None


def ask_image_path() -> Path | None:
    try:
        raw_path = input("Paste the image path and press Enter: ").strip()
    except EOFError:
        return None

    return clean_path(raw_path) if raw_path else None


def resolve_image_path(args: argparse.Namespace) -> Path:
    image_path = args.image_path or args.image_path_arg
    if image_path is not None:
        return image_path

    if args.choose_file:
        selected_path = choose_image_path()
        if selected_path is not None:
            return selected_path

    if args.ask_path:
        selected_path = ask_image_path()
        if selected_path is not None:
            return selected_path

    for default_path in DEFAULT_IMAGE_PATHS:
        if default_path.exists():
            return default_path

    selected_path = ask_image_path()
    if selected_path is not None:
        return selected_path

    return DEFAULT_IMAGE_PATHS[0]


def parse_point(raw_point: str) -> list[int]:
    raw_x, raw_y = raw_point.split(",", maxsplit=1)
    return [int(float(raw_x)), int(float(raw_y))]


def parse_zone_polygon(
    args: argparse.Namespace, image_size: tuple[int, int]
) -> np.ndarray:
    if args.zone_polygon:
        points = [parse_point(point) for point in args.zone_polygon.split()]
        if len(points) < 3:
            raise ValueError("--zone-polygon must contain at least three points.")
        return np.array(points, dtype=np.int32)

    if args.zone_rect:
        values = [int(float(value.strip())) for value in args.zone_rect.split(",")]
        if len(values) != 4:
            raise ValueError('--zone-rect must use the format "x1,y1,x2,y2".')
        x1, y1, x2, y2 = values
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

    width, height = image_size
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.int32,
    )


def parse_manual_box(raw_box: str) -> DetectionResult:
    parts = [part.strip() for part in raw_box.split(",")]
    if len(parts) != 5:
        raise ValueError('--manual-box must use the format "label,x1,y1,x2,y2".')

    label = parts[0]
    xyxy = [float(value) for value in parts[1:]]
    return DetectionResult(label=label, xyxy=xyxy, confidence=None)


def select_zone_and_boxes(
    image: Image.Image,
) -> tuple[np.ndarray, list[DetectionResult]]:
    try:
        import tkinter as tk

        from PIL import ImageTk
    except ImportError as error:
        raise RuntimeError(
            "Interactive mode needs tkinter and Pillow ImageTk."
        ) from error

    max_width = 1100
    max_height = 750
    scale = min(max_width / image.width, max_height / image.height, 1.0)
    preview_size = (int(image.width * scale), int(image.height * scale))
    preview_image = image.resize(preview_size)

    root = tk.Tk()
    root.title("Draw occupancy zone and object boxes")

    status_var = tk.StringVar(
        value="Drag the zone rectangle first. Then drag one box per object. Press Done."
    )
    status = tk.Label(root, textvariable=status_var, anchor="w")
    status.pack(fill="x")

    photo = ImageTk.PhotoImage(preview_image)
    canvas = tk.Canvas(
        root, width=preview_size[0], height=preview_size[1], cursor="cross"
    )
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=photo)

    selections: list[tuple[str, tuple[int, int, int, int], int]] = []
    drag_start: tuple[int, int] | None = None
    preview_rect_id: int | None = None

    def normalize_rect(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def current_mode() -> str:
        return "zone" if not selections else "object"

    def on_mouse_down(event: tk.Event) -> None:
        nonlocal drag_start, preview_rect_id
        drag_start = (event.x, event.y)
        color = "lime" if current_mode() == "zone" else "red"
        preview_rect_id = canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline=color,
            width=3,
        )

    def on_mouse_move(event: tk.Event) -> None:
        if drag_start is None or preview_rect_id is None:
            return
        canvas.coords(preview_rect_id, drag_start[0], drag_start[1], event.x, event.y)

    def on_mouse_up(event: tk.Event) -> None:
        nonlocal drag_start, preview_rect_id
        if drag_start is None or preview_rect_id is None:
            return

        x1, y1, x2, y2 = normalize_rect(drag_start[0], drag_start[1], event.x, event.y)
        if abs(x2 - x1) < 4 or abs(y2 - y1) < 4:
            canvas.delete(preview_rect_id)
        else:
            mode = current_mode()
            selections.append((mode, (x1, y1, x2, y2), preview_rect_id))
            if mode == "zone":
                status_var.set(
                    "Zone selected. Now drag object boxes. Press Undo to fix mistakes."
                )
            else:
                status_var.set(
                    f"{len(selections) - 1} object box(es) selected. "
                    "Press Done when ready."
                )

        drag_start = None
        preview_rect_id = None

    def undo() -> None:
        if not selections:
            return
        _, _, rect_id = selections.pop()
        canvas.delete(rect_id)
        if not selections:
            status_var.set("Drag the zone rectangle first.")
        else:
            status_var.set(f"{len(selections) - 1} object box(es) selected.")

    def done() -> None:
        if not selections:
            status_var.set("Draw a zone rectangle before pressing Done.")
            return
        root.quit()

    controls = tk.Frame(root)
    controls.pack(fill="x")
    tk.Button(controls, text="Undo", command=undo).pack(side="left")
    tk.Button(controls, text="Done", command=done).pack(side="right")

    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)
    root.bind("<Return>", lambda _event: done())
    root.mainloop()
    root.destroy()

    if not selections:
        raise ValueError("No zone selected. Rerun --interactive and draw a zone first.")

    zone_rect = selections[0][1]
    object_rects = [selection[1] for selection in selections[1:]]

    def to_original(rect: tuple[int, int, int, int]) -> list[float]:
        return [coordinate / scale for coordinate in rect]

    x1, y1, x2, y2 = to_original(zone_rect)
    zone_polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    detections = [
        DetectionResult(label=f"object_{idx}", xyxy=to_original(rect))
        for idx, rect in enumerate(object_rects, start=1)
    ]
    return zone_polygon, detections


def detect_objects(
    image_path: Path,
    model_name: str,
    confidence: float,
    classes_to_count: set[str],
) -> list[DetectionResult]:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Automatic detection needs Ultralytics. Install it with:\n"
            r".\.venv\Scripts\python.exe -m pip install ultralytics"
        ) from error

    model = YOLO(model_name)
    result = model(str(image_path), conf=confidence, verbose=False)[0]
    names = result.names
    detections: list[DetectionResult] = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        label = names[class_id]
        if label not in classes_to_count:
            continue

        xyxy = box.xyxy[0].cpu().numpy().astype(float).tolist()
        detections.append(
            DetectionResult(
                label=label,
                xyxy=xyxy,
                confidence=float(box.conf.item()),
            )
        )

    return detections


def detections_to_supervision(detections: list[DetectionResult]) -> sv.Detections:
    if not detections:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array([detection.xyxy for detection in detections], dtype=np.float32),
        confidence=np.array(
            [
                0.0 if detection.confidence is None else detection.confidence
                for detection in detections
            ],
            dtype=np.float32,
        ),
    )


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: str = "yellow",
) -> None:
    font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox(xy, text, font=font)
    padding = 6
    draw.rectangle(
        [
            left - padding,
            top - padding,
            right + padding,
            bottom + padding,
        ],
        fill="black",
    )
    draw.text(xy, text, fill=fill, font=font)


def object_occupancy(zone: sv.PolygonZone, xyxy: list[float]) -> float:
    detections = sv.Detections(xyxy=np.array([xyxy], dtype=np.float32))
    return zone.get_occupancy(detections)


def detection_label(detection: DetectionResult) -> str:
    if detection.confidence is None:
        return detection.label

    return f"{detection.label} {detection.confidence:.0%}"


def main() -> None:
    args = parse_args()
    image_path = resolve_image_path(args)
    if not image_path.exists():
        raise FileNotFoundError(
            "Image not found. Put your test image in this folder as input.jpg, "
            "input.png, or input.jpeg, pass --image-path, use --ask-path, or use "
            "--choose-file."
        )

    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image_path}")
    print(f"Image size: {image.size[0]}x{image.size[1]}")

    if args.interactive:
        zone_polygon, detections = select_zone_and_boxes(image)
    else:
        zone_polygon = parse_zone_polygon(args, image.size)
        detections = [parse_manual_box(raw_box) for raw_box in args.manual_box]

    zone = sv.PolygonZone(polygon=zone_polygon)

    automatic_detection_ran = not args.skip_detector and not args.interactive
    if automatic_detection_ran:
        detections.extend(
            detect_objects(
                image_path=image_path,
                model_name=args.model,
                confidence=args.confidence,
                classes_to_count=set(args.classes),
            )
        )

    if automatic_detection_ran and not detections and not args.no_interactive_fallback:
        print(
            "Automatic detection found no matching objects. "
            "Opening the drawing tool so you can select the zone and objects."
        )
        zone_polygon, detections = select_zone_and_boxes(image)
        zone = sv.PolygonZone(polygon=zone_polygon)

    sv_detections = detections_to_supervision(detections)
    occupancy = zone.get_occupancy(sv_detections)

    print(f"Detected objects counted: {len(detections)}")
    print(f"Zone occupancy from detections: {occupancy:.2%}")
    if not detections:
        print(
            "No objects were detected. Try --interactive, add --manual-box, "
            "lower --confidence, or use a detector trained for this camera angle."
        )
    print("Objects that overlap the zone:")
    for detection in detections:
        contribution = object_occupancy(zone, detection.xyxy)
        if contribution > 0:
            print(f"- {detection_label(detection)}: {contribution:.2%} of the zone")

    annotated = image.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.polygon(
        [tuple(point) for point in zone_polygon],
        fill=(0, 255, 0, 45),
        outline=(0, 255, 0, 255),
    )
    annotated = Image.alpha_composite(annotated, overlay)
    draw = ImageDraw.Draw(annotated)

    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection.xyxy]
        contribution = object_occupancy(zone, detection.xyxy)
        color = "red" if contribution > 0 else "orange"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if contribution > 0:
            draw_label(
                draw,
                (x1, max(0, y1 - 22)),
                detection_label(detection),
                fill=color,
            )

    draw.line(
        [tuple(point) for point in zone_polygon] + [tuple(zone_polygon[0])],
        fill="lime",
        width=4,
    )
    draw_label(draw, (24, 24), f"Zone occupancy: {occupancy:.2%}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.convert("RGB").save(args.output_path)
    print(f"Saved annotated result to: {args.output_path}")


if __name__ == "__main__":
    main()
