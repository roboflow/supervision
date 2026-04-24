---
comments: true
description: Frequently asked questions about installing Supervision, supported computer vision models, datasets, tracking, metrics, and licensing.
---

# Frequently Asked Questions

## What is Supervision?

Supervision is an open-source Python library by Roboflow for computer vision workflows. It provides a unified `Detections` class with converters for supported object detection, segmentation, and VLM outputs.

## How do I install Supervision?

Install the base package with:

```bash
pip install supervision
```

Use the `metrics` extra when you need optional metric dependencies:

```bash
pip install "supervision[metrics]"
```

Sample asset utilities are part of the base package under `supervision.assets`.

## Which object detection models work with Supervision?

Supervision is model agnostic. `sv.Detections` includes converters for Ultralytics YOLO, Roboflow Inference, Hugging Face Transformers outputs, SAM, Detectron2, MMDetection, YOLO-NAS, PaddleDet, NCNN, Azure AI Vision, and VLM parsers including Florence-2, PaliGemma, Qwen VL, Gemini, DeepSeek VL 2, and Moondream. Keypoint outputs have separate `sv.KeyPoints` converters, including MediaPipe.

## What can I do with Supervision?

You can annotate images and video, filter detections, track objects, count objects in zones or across lines, load and convert datasets, evaluate models with detection metrics, and export predictions for downstream analysis.

## How do I track objects across video frames?

Assign persistent tracker IDs before visualization. The built-in `sv.ByteTrack` wrapper accepts `Detections` through `update_with_detections()`. After tracking, combine the output with annotators such as `sv.TraceAnnotator`, `sv.BoxAnnotator`, and `sv.LabelAnnotator`.

## What dataset formats does Supervision support?

For detection datasets, Supervision supports YOLO, COCO JSON, and Pascal VOC. Use `DetectionDataset.from_yolo()`, `DetectionDataset.from_coco()`, or `DetectionDataset.from_pascal_voc()` to load datasets, and the matching `as_*` methods to export them.

## How do I count objects in a zone?

Use `sv.PolygonZone` for arbitrary polygon regions and `sv.LineZone` for line-crossing counts. Line crossing requires `detections.tracker_id`, so run a tracker before calling the line zone trigger.

## How do I benchmark a model?

Use `supervision.metrics.mean_average_precision.MeanAveragePrecision` for mAP and `sv.ConfusionMatrix` for confusion matrices. Accumulate predictions and ground-truth `Detections`, then call `compute()` to calculate metrics.

## Is Supervision free to use?

Yes. Supervision is free and open source under the MIT license.

## Where is the source code?

The source code is available at [github.com/roboflow/supervision](https://github.com/roboflow/supervision).
