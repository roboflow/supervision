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

## ⚙️ run

Use any video of your own:

```bash
python depth_showcase.py --source_video_path my-video.mp4 --device mps
```

This writes `outputs/my-video-depth-anything.mp4`, `outputs/my-video-depth-pro.mp4` and `outputs/my-video-yolo26.mp4`.

`./run_all.sh` downloads the `VEHICLES` and `PEOPLE_WALKING` sample videos of supervision and runs every model on both. Other members of `sv.assets.VideoAssets` are `SUBWAY`, `BEACH` and `SKIING`.

| Option                    | Meaning                                                                                                                       |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `--target_directory`      | Directory for the output videos, `outputs` by default                                                                         |
| `--models`                | JSON list of models to run, such as `'["depth-anything", "yolo26"]'`; the choices are `depth-anything`, `depth-pro`, `yolo26` |
| `--max_seconds`           | Length of the video to process                                                                                                |
| `--colormap`, `--opacity` | Colour table, such as `turbo` or `viridis`, and its opacity                                                                   |
| `--device`                | `cpu`, `mps` or `cuda`                                                                                                        |

Depth Pro is by far the slowest of the three; leave it out for a quick preview.

## © license

- supervision: [MIT license](https://github.com/roboflow/supervision/blob/develop/LICENSE.md).
- Depth Anything V2 Large: [CC-BY-NC-4.0](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf), non-commercial use only.
- Apple Depth Pro: licensed as `apple-amlr` on [`apple/DepthPro-hf`](https://huggingface.co/apple/DepthPro-hf); read it before commercial use.
- ultralytics: YOLO26 is distributed under the [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
