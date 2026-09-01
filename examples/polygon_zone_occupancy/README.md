# Polygon Zone Occupancy Image Example

This example loads an image, detects objects with YOLO, and uses
`PolygonZone.get_occupancy` to calculate how much of a zone is covered by those
detections.

By default, the zone is the full image. You can also pass a rectangle or polygon
zone from PowerShell.

For random images, the most reliable option is `--interactive`: you choose the
image, draw the zone, draw object boxes, and the script calculates occupancy
without hard-coded coordinates.

## One-Time Detector Setup

Automatic detection needs Ultralytics:

```powershell
.\.venv\Scripts\python.exe -m pip install ultralytics
```

## Option 1: Use a Local Input File

Put an image in this folder named one of:

- `input.jpg`
- `input.png`
- `input.jpeg`

Then run:

```powershell
.\.venv\Scripts\python.exe examples\polygon_zone_occupancy\image_example.py
```

## Option 2: Paste a File Path in the Command

```powershell
.\.venv\Scripts\python.exe examples\polygon_zone_occupancy\image_example.py --image-path "C:\path\to\your\image.jpg"
```

You can also pass the image path without `--image-path`:

```powershell
.\.venv\Scripts\python.exe examples\polygon_zone_occupancy\image_example.py "C:\path\to\your\image.jpg"
```

If YOLO does not detect anything, the script opens the drawing tool so you can
select the zone and object boxes yourself.

## Option 3: Ask for the File Path

```powershell
.\.venv\Scripts\python.exe examples\polygon_zone_occupancy\image_example.py --ask-path
```

The script will ask you to paste the image path.

## Option 4: Use a File Picker

```powershell
.\.venv\Scripts\python.exe examples\polygon_zone_occupancy\image_example.py --choose-file
```

The script will open a Windows file picker.

## Option 5: Draw the Zone and Boxes

This avoids hard-coded coordinates:

```powershell
.\.venv\Scripts\python.exe examples\polygon_zone_occupancy\image_example.py --choose-file --interactive
```

In the window:

- drag the zone rectangle first
- drag one box around each object that should count
- click `Done` or press Enter

The script will calculate occupancy from what you drew.

## Custom Zone Options

Full image zone:

```powershell
.\.venv\Scripts\python.exe examples\polygon_zone_occupancy\image_example.py --image-path "C:\path\to\your\image.jpg"
```

Rectangle zone:

```powershell
.\.venv\Scripts\python.exe examples\polygon_zone_occupancy\image_example.py --image-path "C:\path\to\your\image.jpg" --zone-rect "30,0,1260,410"
```

Polygon zone:

```powershell
.\.venv\Scripts\python.exe examples\polygon_zone_occupancy\image_example.py --image-path "C:\path\to\your\image.jpg" --zone-polygon "30,0 1260,0 1260,410 30,410"
```

## Count Different Object Classes

By default, the example counts:

- `person`
- `bicycle`
- `car`
- `motorcycle`
- `bus`
- `truck`

To count only cars and trucks:

```powershell
.\.venv\Scripts\python.exe examples\polygon_zone_occupancy\image_example.py --classes car truck
```

## Manual Fallback

If you do not want to run YOLO, you can pass boxes manually:

```powershell
.\.venv\Scripts\python.exe examples\polygon_zone_occupancy\image_example.py --skip-detector --manual-box "car,42,58,177,363"
```

## Automatic Detection Limits

YOLO can detect common objects in many normal images, but it may miss unusual
camera angles, such as overhead parking-lot cars. If automatic detection returns
zero objects, this script falls back to `--interactive` automatically.

To keep the script from opening the drawing tool after zero detections:

```powershell
.\.venv\Scripts\python.exe examples\polygon_zone_occupancy\image_example.py --no-interactive-fallback
```

## Output

The script prints:

- the loaded image path
- image size
- number of detected objects counted
- zone occupancy percentage
- objects that overlap the zone

It also saves an annotated image to:

```text
examples\polygon_zone_occupancy\occupancy_result.jpg
```
