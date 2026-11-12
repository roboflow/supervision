# depth showcase

## 👋 hello

This example colours a video with the depth predicted by several models. Each model runs on every frame, [`sv.DepthAnnotator`](https://supervision.roboflow.com/latest/depth/annotators/) paints its depth maps on one colour range shared by the whole clip, and each model gets its own output video as soon as it finishes, captioned with its measured speed. Compose the videos into a side-by-side comparison with any video editor.

| Model                                                                                       | Predicts               | Loaded with                     |
| ------------------------------------------------------------------------------------------- | ---------------------- | ------------------------------- |
| [Depth Anything V2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) | relative inverse depth | `sv.DepthMap.from_transformers` |
| [Apple Depth Pro](https://huggingface.co/apple/DepthPro-hf)                                 | metric depth in metres | `sv.DepthMap.from_transformers` |
| Ultralytics YOLO26 depth (`yolo26l-depth.pt`)                                               | metric depth in metres | `sv.DepthMap.from_ultralytics`  |

The input video is only a parameter, so the same command works for a road, a crowd or a beach.

## 💻 install

```bash
pip install -r requirements.txt
```

The minimum versions are `supervision>=0.31.0` for `sv.DepthMap` and `sv.DepthAnnotator`, `ultralytics>=8.4.104` for the YOLO26 depth task and `transformers>=4.49.0` for Depth Pro.

## ⚙️ run

Use any video of your own:

```bash
python depth_showcase.py --source_video_path my-video.mp4 --device mps
```

Without `--target_directory` the frames of each model are shown live in a window, and `q` stops. Every frame is predicted, coloured and shown or written in one pass. For a video file the colour range is fixed up front from `--range_samples` frames when frame-count metadata is usable; otherwise the first frames calibrate the range. If calibration finds no valid depth, the annotator uses its automatic range. For a live stream there is nothing ahead to sample, so the range follows the frames slowly. With `--target_directory outputs` it writes `outputs/my-video-depth-anything.mp4`, `outputs/my-video-depth-pro.mp4` and `outputs/my-video-yolo26.mp4`.

`--source_video_path` also takes the name of a sample video of supervision, any member of `sv.assets.VideoAssets` such as `VEHICLES`, `PEOPLE_WALKING` or `SKIING`, and downloads it to `data/` first. `./run_all.sh` does this for every sample video and runs every model on the full length of each. It selects CUDA, MPS or CPU from the installed PyTorch runtime, while a non-empty `DEVICE` value overrides that choice. Expect hours on a laptop, mostly for Depth Pro; set `MODELS` and `MAX_SECONDS` to shorten it, for example `MODELS='["yolo26"]' MAX_SECONDS=8 ./run_all.sh`.

| Option                    | Meaning                                                                                                                       |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `--target_directory`      | Directory for the output videos; the video is only shown live if not set                                                      |
| `--models`                | JSON list of models to run, such as `'["depth-anything", "yolo26"]'`; the choices are `depth-anything`, `depth-pro`, `yolo26` |
| `--max_seconds`           | Length of the video or stream to process, all of it by default                                                                |
| `--range_samples`         | Frames sampled from a video file to fix its colour range, 5 by default                                                        |
| `--colormap`, `--opacity` | Colour table, such as `turbo` or `viridis`, and its opacity                                                                   |
| `--device`                | `cpu`, `mps` or `cuda`                                                                                                        |

`--source_video_path` also takes a camera index such as `0` or a stream URL such as `rtsp://...` for a truly live view: `python depth_showcase.py --source_video_path 0 --models '["yolo26"]' --device mps`. Pick a fast model for it, since Depth Pro needs about 2 s per frame on a laptop.

Depth Pro is by far the slowest of the three; leave it out for a quick preview.

## © license

- supervision: [MIT license](https://github.com/roboflow/supervision/blob/develop/LICENSE.md).
- Depth Anything V2 Large: [CC-BY-NC-4.0](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf), non-commercial use only.
- Apple Depth Pro: licensed as `apple-amlr` on [`apple/DepthPro-hf`](https://huggingface.co/apple/DepthPro-hf); read it before commercial use.
- ultralytics: YOLO26 is distributed under the [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
