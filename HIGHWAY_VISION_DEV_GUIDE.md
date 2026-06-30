# Supervision 项目介绍与高速公路视觉分析系统开发指南

## 一、项目介绍

### 1.1 项目概述

**supervision** 是由 Roboflow 维护的开源计算机视觉工具库，被誉为"你的计算机视觉必备工具包"。它是一个模型无关（model-agnostic）的 Python 库，核心设计理念是将来自不同检测、分割、分类和关键点模型的输出标准化为统一的数据容器，然后在此基础上提供丰富的标注、跟踪、区域计数、数据集管理和指标评估工具。

- **版本**: `0.29.0.dev`（开发版）
- **许可证**: MIT
- **Python**: `>=3.9`（支持 3.9 ~ 3.14）
- **上游仓库**: https://github.com/roboflow/supervision
- **Fork 仓库**: https://github.com/DavidWangW/supervision
- **官方文档**: https://supervision.roboflow.com/latest/

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| 模型无关 | 支持 10+ 模型家族的标准化输出转换（YOLO、Transformers、Detectron2、MMDetection 等） |
| 统一数据容器 | `Detections`、`KeyPoints`、`Classifications` 三大核心数据类 |
| 丰富的标注器 | 22+ 可组合的可视化标注器（框、掩码、标签、热力图、轨迹等） |
| 多目标跟踪 | 内置 ByteTrack 跟踪器（已废弃，推荐使用外部 `trackers` 包） |
| 区域计数 | `LineZone`（线穿越计数）和 `PolygonZone`（多边形区域计数） |
| 数据集管理 | 支持 COCO / YOLO / Pascal VOC 格式的加载、保存、拆分与合并 |
| 评估指标 | mAP、mAR、混淆矩阵、精确率、召回率、F1 |
| 视频处理 | 帧生成器、视频写入、FPS 监控等视频工具 |

### 1.3 项目结构

```
d:\MyGithubRepo\supervision\
├── .github/              # CI、CONTRIBUTING.md、工作流
├── docs/                 # MkDocs 文档源码
├── examples/             # 端到端示例脚本
│   ├── tracking/             # 目标跟踪示例
│   ├── count_people_in_zone/ # 区域计数示例
│   ├── traffic_analysis/     # 交通流分析示例 ⭐
│   ├── speed_estimation/     # 速度估算示例 ⭐
│   ├── time_in_zone/         # 区域停留时间示例
│   ├── heatmap_and_track/    # 热力图与轨迹示例
│   └── compact_mask/         # 紧凑掩码示例
├── src/
│   └── supervision/      # 主包源码
│       ├── detection/        # 核心检测容器 (Detections)、VLM、LineZone、PolygonZone、Sinks、Slicer
│       ├── annotators/       # 22+ 可视化标注器
│       ├── dataset/          # 数据集管理 (COCO/YOLO/Pascal VOC)
│       ├── draw/             # 颜色系统与底层绘图原语
│       ├── geometry/         # 几何原语: Point, Vector, Rect, Position
│       ├── tracker/          # ByteTrack 多目标跟踪器（已废弃）
│       ├── key_points/       # 关键点检测容器与标注器
│       ├── classification/   # 分类容器
│       ├── metrics/          # 评估指标: mAP, ConfusionMatrix, Precision, Recall, F1
│       ├── utils/            # 视频/图像/文件/转换等工具
│       └── validators/       # 内部验证函数
├── tests/                # pytest 测试套件
├── pyproject.toml        # 构建/依赖/lint 配置
├── mkdocs.yml            # 文档构建配置
└── uv.lock               # uv 锁文件
```

### 1.4 核心依赖

| 依赖 | 版本要求 | 用途 |
|------|---------|------|
| numpy | >=1.21.2 | 数组运算 |
| opencv-python | >=4.5.5.64 | 图像处理 |
| scipy | >=1.10 | 科学计算 |
| matplotlib | >=3.6 | 绘图/调色板 |
| pillow | >=9.4 | 图像处理 |
| tqdm | >=4.62.3 | 进度条 |
| pyyaml | >=5.3 | YAML 文件处理 |
| requests | >=2.26 | HTTP 请求（资源下载） |

---

## 二、开发环境配置

### 2.1 前置条件

- Python >= 3.9
- [uv](https://docs.astral.sh/uv/) 包管理器（已安装）
- Git

### 2.2 环境搭建（已完成）

你已经通过 `git clone` 和 `uv` 完成了环境配置，以下是完整步骤回顾：

```bash
# 1. 克隆 Fork 仓库
git clone https://github.com/DavidWangW/supervision.git
cd supervision

# 2. 使用 uv 创建虚拟环境并安装依赖
uv sync

# 3. 激活虚拟环境 (Windows PowerShell)
.venv\Scripts\Activate.ps1

# 4. 安装 pre-commit 钩子
uv run pre-commit install
```

### 2.3 验证安装

```bash
# 运行测试套件，确认环境正常
uv run pytest --cov=supervision

# 运行 pre-commit 检查
uv run pre-commit run --all-files
```

### 2.4 关键开发工具

| 工具 | 用途 | 命令 |
|------|------|------|
| uv | 包管理与虚拟环境 | `uv sync`, `uv run`, `uv add` |
| pytest | 测试运行 | `uv run pytest --cov=supervision` |
| pre-commit | 提交前自动检查 | `uv run pre-commit run --all-files` |
| ruff | 代码格式化与 lint | `uv run ruff check .`, `uv run ruff format .` |
| mypy | 类型检查 | `uv run mypy src/supervision` |

---

## 三、代码规范与提交约定

### 3.1 分支策略

- 从 `develop` 分支创建新分支，使用前缀：`feat/`、`fix/`、`docs/`、`refactor/`、`test/`、`chore/`
- PR 必须指向 `develop` 分支

```bash
# 示例：创建功能分支
git checkout develop
git pull origin develop
git checkout -b feat/highway-speed-display
```

### 3.2 提交信息规范（Conventional Commits）

```
feat: 新功能
fix: 修复 bug
docs: 文档更新
refactor: 重构（不改变行为）
perf: 性能优化
test: 测试相关
chore: 构建/工具/杂项
```

### 3.3 代码风格

- **格式化与 Lint**: 由 **ruff** 强制执行（目标: py39, 行宽 88, 双引号）
- **类型注解**: 所有新代码必须添加类型注解，鼓励使用 mypy 严格模式
- **文档字符串**: Google Python 风格，所有新函数和类必须包含，应包含可运行的示例

```python
def calculate_headway(distance: float, speed: float) -> float:
    """Calculate time headway between two vehicles.

    Args:
        distance: Distance between vehicles in meters.
        speed: Speed of the following vehicle in m/s.

    Returns:
        Time headway in seconds.

    Example:
        >>> calculate_headway(50.0, 25.0)
        2.0
    """
    if speed <= 0:
        return float("inf")
    return distance / speed
```

### 3.4 提交前必做检查

```bash
uv run pytest --cov=supervision
uv run pre-commit run --all-files
```

---

## 四、核心 API 速查

### 4.1 Detections — 检测数据容器

`Detections` 是整个库最核心的数据结构，统一封装所有模型的输出：

```python
import supervision as sv

# 从 Ultralytics YOLO 模型创建
detections = sv.Detections.from_ultralytics(results)

# 字段说明
detections.xyxy          # 边界框 (N, 4) — [x1, y1, x2, y2]
detections.mask          # 分割掩码 (N, H, W) 或 CompactMask
detections.confidence    # 置信度 (N,)
detections.class_id      # 类别 ID (N,)
detections.tracker_id    # 跟踪 ID (N,)
detections.data          # 自定义数据字典 (如 class_name)
detections.metadata      # 集合级元数据

# 切片与过滤
detections[0:5]                                    # 前 5 个检测
detections[detections.confidence > 0.5]            # 高置信度过滤
detections[detections.class_id == 2]               # 按类别过滤

# 锚点坐标（用于标注定位）
points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
```

### 4.2 Annotators — 可组合标注器

标注器遵循**可组合模式**，每个标注器独立，通过链式调用叠加效果：

```python
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5)
trace_annotator = sv.TraceAnnotator(trace_length=30)

# 链式标注
annotated = frame.copy()
annotated = trace_annotator.annotate(scene=annotated, detections=detections)
annotated = box_annotator.annotate(scene=annotated, detections=detections)
annotated = label_annotator.annotate(
    scene=annotated, detections=detections, labels=labels
)
```

**常用标注器一览：**

| 标注器 | 用途 | 高速公路场景应用 |
|--------|------|-----------------|
| `BoxAnnotator` | 矩形框 | 车辆检测框 |
| `RoundBoxAnnotator` | 圆角矩形框 | 美化检测框 |
| `LabelAnnotator` | 文本标签 | 显示车速、ID |
| `TraceAnnotator` | 运动轨迹 | 车辆行驶轨迹 |
| `HeatMapAnnotator` | 热力图 | 交通密度可视化 |
| `PolygonAnnotator` | 多边形标注 | 标注检测区域 |
| `BlurAnnotator` | 模糊处理 | 隐私保护 |
| `PercentageBarAnnotator` | 百分比条 | 置信度/速度比例条 |

### 4.3 Zone — 区域计数

#### LineZone — 线穿越计数

```python
# 定义计数线
start = sv.Point(x=0, y=1080)
end = sv.Point(x=1920, y=1080)
line_zone = sv.LineZone(start=start, end=end)

# 每帧更新
crossed_in, crossed_out = line_zone.trigger(detections)

# 可视化
line_annotator = sv.LineZoneAnnotator(thickness=2)
annotated = line_annotator.annotate(scene=annotated, line_counter=line_zone)
```

#### PolygonZone — 多边形区域计数

```python
polygon = np.array([[0, 0], [1920, 0], [1920, 500], [0, 500]])
zone = sv.PolygonZone(polygon=polygon)
inside = zone.trigger(detections)  # 返回布尔数组
detections_in_zone = detections[inside]
```

### 4.4 Tracker — 多目标跟踪

```python
# 内置 ByteTrack（已废弃但仍可用）
tracker = sv.ByteTrack(frame_rate=30)
detections = tracker.update_with_detections(detections)

# 推荐：外部 trackers 包
# pip install trackers
from trackers import ByteTrackTracker
tracker = ByteTrackTracker()
detections = tracker.update(detections)
```

### 4.5 Video — 视频处理工具

```python
# 视频信息
video_info = sv.VideoInfo.from_video_path("input.mp4")
# video_info.fps, video_info.width, video_info.height, video_info.total_frames

# 帧生成器
frame_generator = sv.get_video_frames_generator(source_path="input.mp4")

# 视频写入
with sv.VideoSink("output.mp4", video_info) as sink:
    for frame in frame_generator:
        # 处理帧...
        sink.write_frame(annotated_frame)
```

### 4.6 Sinks — 数据导出

```python
# CSV 导出
with sv.CSVSink("detections.csv") as sink:
    for frame in frame_generator:
        # ...检测...
        sink.append(detections, custom_data={"frame_id": i})

# JSON 导出
with sv.JSONSink("detections.json") as sink:
    # ...同上...
    sink.append(detections)
```

### 4.7 Color — 颜色系统

```python
# 预定义颜色
sv.Color.RED, sv.Color.GREEN, sv.Color.BLUE, sv.Color.WHITE, sv.Color.BLACK

# 自定义颜色
color = sv.Color.from_hex("#FF5733")

# 调色板
palette = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
# 默认调色板: sv.ColorPalette.DEFAULT
```

---

## 五、高速公路视觉分析系统开发指南

### 5.1 系统目标

基于 supervision 库，开发一款面向**高速公路场景**的智能视觉分析系统，实现以下核心功能：

| 功能模块 | 描述 | 对应 supervision 能力 |
|----------|------|----------------------|
| 车辆检测 | 实时检测画面中的车辆 | `Detections` + `from_ultralytics()` |
| 多目标跟踪 | 对检测到的车辆持续跟踪 | `ByteTrack` / `trackers` 包 |
| 车速显示 | 估算并标注每辆车的行驶速度 | `ViewTransformer` + `TraceAnnotator` + `LabelAnnotator` |
| 车流量统计 | 统计通过某截面的车辆数量 | `LineZone` + `LineZoneAnnotator` |
| 车头间距测量 | 计算同车道前后车辆之间的距离/时间间距 | 自定义逻辑 + 几何计算 |

### 5.2 推荐技术栈

```
检测模型:  Ultralytics YOLO (yolo11x.pt 或自定义训练的高速公路模型)
跟踪器:    ByteTrack (sv.ByteTrack 或 trackers 包)
标注:      supervision annotators (BoxAnnotator, LabelAnnotator, TraceAnnotator, ...)
区域分析:  LineZone, PolygonZone
透视变换:  cv2.getPerspectiveTransform (像素→真实坐标映射)
视频 I/O:  supervision VideoInfo + VideoSink + get_video_frames_generator
数据导出:  CSVSink, JSONSink
```

### 5.3 系统架构设计

```
┌──────────────────────────────────────────────────────┐
│                   高速公路视觉分析系统                    │
├──────────────┬───────────────┬───────────────────────┤
│   输入层      │   处理层       │   输出层               │
├──────────────┼───────────────┼───────────────────────┤
│ 视频流/摄像头  │ 车辆检测       │ 可视化视频输出          │
│ RTSP/本地文件 │ 多目标跟踪     │ 车速标注显示            │
│              │ 透视坐标变换    │ 车流量统计面板          │
│              │ 速度估算       │ 车头间距标注            │
│              │ 区域计数       │ 数据导出(CSV/JSON)     │
│              │ 间距计算       │ 实时预警               │
└──────────────┴───────────────┴───────────────────────┘
```

### 5.4 核心模块实现思路

#### 5.4.1 车辆检测

```python
from ultralytics import YOLO
import supervision as sv

model = YOLO("yolo11x.pt")  # 或自定义训练模型

def detect_vehicles(frame: np.ndarray) -> sv.Detections:
    """检测帧中的车辆（car=2, motorcycle=3, bus=5, truck=7）。"""
    results = model(frame, conf=0.3, iou=0.7)[0]
    detections = sv.Detections.from_ultralytics(results)
    # 过滤车辆类别 (COCO 类别 ID)
    vehicle_classes = [2, 3, 5, 7]
    if detections.class_id is not None:
        vehicle_mask = np.isin(detections.class_id, vehicle_classes)
        detections = detections[vehicle_mask]
    return detections
```

#### 5.4.2 多目标跟踪

```python
tracker = sv.ByteTrack(frame_rate=30, track_activation_threshold=0.3)

# 在每帧中更新跟踪
detections = tracker.update_with_detections(detections)
# 之后可通过 detections.tracker_id 访问每辆车的唯一 ID
```

#### 5.4.3 车速估算

参考 `examples/speed_estimation/` 的实现思路，核心是**透视变换**将像素坐标映射到真实世界坐标：

```python
class ViewTransformer:
    """透视变换器：将图像像素坐标映射到真实世界坐标。"""

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


# 使用方式：
# 1. SOURCE: 图像中 4 个标定点（如车道线的 4 个角点）
# 2. TARGET: 对应的真实世界坐标（米）
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
TARGET_WIDTH = 25    # 米
TARGET_HEIGHT = 250  # 米
TARGET = np.array([
    [0, 0], [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1],
])

view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

# 3. 速度计算：连续帧间跟踪点的位移 / 时间
from collections import defaultdict, deque

coordinates = defaultdict(lambda: deque(maxlen=int(fps)))

# 每帧更新坐标
points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
points = view_transformer.transform_points(points).astype(int)
for tracker_id, [_, y] in zip(detections.tracker_id, points):
    coordinates[tracker_id].append(y)

# 计算速度
for tracker_id in detections.tracker_id:
    if len(coordinates[tracker_id]) >= fps / 2:
        distance = abs(coordinates[tracker_id][-1] - coordinates[tracker_id][0])  # 米
        time = len(coordinates[tracker_id]) / fps  # 秒
        speed = distance / time * 3.6  # km/h
        labels.append(f"#{tracker_id} {int(speed)} km/h")
```

> **重要提示**: `SOURCE` 和 `TARGET` 必须根据实际摄像头的视角单独标定，不同机位需要不同配置。

#### 5.4.4 车流量统计

使用 `LineZone` 在画面中设置虚拟检测线：

```python
# 设置检测线（如高速公路横截面）
line_start = sv.Point(x=500, y=600)
line_end = sv.Point(x=1400, y=600)
line_zone = sv.LineZone(start=line_start, end=line_end)

# 每帧触发
crossed_in, crossed_out = line_zone.trigger(detections)

# 获取累计计数
in_count = line_zone.in_count
out_count = line_zone.out_count

# 可视化
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.5)
annotated = line_annotator.annotate(scene=annotated, line_counter=line_zone)
```

多车道场景可使用多条 `LineZone`：

```python
# 为每条车道创建独立的计数线
lane_lines = [
    sv.LineZone(start=sv.Point(200, 600), end=sv.Point(600, 600)),
    sv.LineZone(start=sv.Point(700, 600), end=sv.Point(1100, 600)),
    sv.LineZone(start=sv.Point(1200, 600), end=sv.Point(1600, 600)),
]
```

#### 5.4.5 车头间距测量

此功能需要自定义开发，supervision 不直接提供。实现思路：

```python
def calculate_headway(
    detections: sv.Detections,
    view_transformer: ViewTransformer,
    lane_boundaries: list[np.ndarray],
) -> dict[int, float]:
    """计算同车道前后车辆的车头时距（秒）。

    Args:
        detections: 包含 tracker_id 的检测数据。
        view_transformer: 透视变换器。
        lane_boundaries: 各车道的边界多边形。

    Returns:
        tracker_id 到车头时距的映射。
    """
    # 1. 将检测点转换到真实世界坐标
    points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    real_points = view_transformer.transform_points(points)

    # 2. 按车道分组
    lane_groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i, point in enumerate(real_points):
        for lane_id, boundary in enumerate(lane_boundaries):
            if _point_in_polygon(point, boundary):
                lane_groups[lane_id].append(
                    (detections.tracker_id[i], point[1])  # (id, y坐标)
                )

    # 3. 在每条车道内按 y 坐标排序，计算相邻车头时距
    headways: dict[int, float] = {}
    for lane_id, vehicles in lane_groups.items():
        vehicles.sort(key=lambda v: v[1])  # 按行驶方向排序
        for i in range(1, len(vehicles)):
            tracker_id, y_rear = vehicles[i - 1]
            _, y_front = vehicles[i]
            distance = abs(y_front - y_rear)  # 米
            # 假设后车速度已知（从速度估算模块获取）
            speed = get_speed(tracker_id)  # m/s
            if speed > 0:
                headways[tracker_id] = distance / speed  # 秒

    return headways


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """判断点是否在多边形内。"""
    result = cv2.pointPolygonTest(polygon.astype(np.float32), point, False)
    return result >= 0
```

### 5.5 完整系统示例框架

以下是一个整合所有功能的基础框架：

```python
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

import supervision as sv


# ============ 配置 ============

# 透视变换标定点（需根据实际摄像头标定）
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
TARGET_WIDTH = 25   # 米
TARGET_HEIGHT = 250  # 米
TARGET = np.array([
    [0, 0], [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1],
])

# 车流量统计检测线
LINE_START = sv.Point(x=500, y=600)
LINE_END = sv.Point(x=1400, y=600)

# 车辆类别 (COCO)
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


# ============ 透视变换器 ============

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


# ============ 主处理逻辑 ============

def main(
    source_video_path: str,
    target_video_path: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    """高速公路车辆检测、跟踪、测速、流量统计与间距分析。"""
    # 初始化模型和工具
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    model = YOLO("yolo11x.pt")
    tracker = sv.ByteTrack(
        frame_rate=video_info.fps,
        track_activation_threshold=confidence_threshold,
    )
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    # 初始化标注器
    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(
        resolution_wh=video_info.resolution_wh
    )
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(video_info.fps * 2),
        position=sv.Position.BOTTOM_CENTER,
    )

    # 初始化车流量统计
    line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
    line_annotator = sv.LineZoneAnnotator(thickness=thickness)

    # 速度追踪
    coordinates = defaultdict(lambda: deque(maxlen=int(video_info.fps)))
    speeds: dict[int, float] = {}

    # 视频处理循环
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            # 1. 检测
            results = model(frame, conf=confidence_threshold, iou=iou_threshold)[0]
            detections = sv.Detections.from_ultralytics(results)

            # 2. 过滤车辆类别
            if detections.class_id is not None:
                vehicle_mask = np.isin(detections.class_id, VEHICLE_CLASSES)
                detections = detections[vehicle_mask]

            # 3. 多目标跟踪
            detections = tracker.update_with_detections(detections=detections)

            # 4. 速度估算
            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = view_transformer.transform_points(points=points).astype(int)

            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            labels = []
            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                    speeds[tracker_id] = 0.0
                else:
                    distance = abs(
                        coordinates[tracker_id][-1] - coordinates[tracker_id][0]
                    )
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6  # km/h
                    speeds[tracker_id] = speed
                    labels.append(f"#{tracker_id} {int(speed)} km/h")

            # 5. 车流量统计
            line_zone.trigger(detections=detections)

            # 6. 可视化标注
            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )
            annotated_frame = line_annotator.annotate(
                scene=annotated_frame, line_counter=line_zone
            )

            sink.write_frame(annotated_frame)


if __name__ == "__main__":
    main(
        source_video_path="data/highway.mp4",
        target_video_path="data/highway_result.mp4",
    )
```

### 5.6 开发路线图

建议按以下阶段推进：

**阶段一：基础检测与跟踪**
- [ ] 搭建项目脚手架，创建 `highway_vision/` 模块目录
- [ ] 集成 YOLO 模型，实现车辆检测
- [ ] 集成 ByteTrack，实现多目标跟踪
- [ ] 基础可视化（检测框 + ID 标签 + 轨迹）

**阶段二：速度估算**
- [ ] 实现透视变换标定工具（交互式选点）
- [ ] 实现 `ViewTransformer` 坐标映射
- [ ] 实现速度估算算法
- [ ] 车速标签可视化

**阶段三：流量统计**
- [ ] 实现 `LineZone` 车道检测线配置
- [ ] 单车道流量计数
- [ ] 多车道独立计数
- [ ] 流量统计面板可视化

**阶段四：车头间距**
- [ ] 车道区域定义与车辆分组
- [ ] 基于透视坐标的间距计算
- [ ] 车头时距（THW）计算
- [ ] 危险间距预警

**阶段五：系统集成与优化**
- [ ] 实时视频流支持（RTSP）
- [ ] 数据导出（CSV/JSON）
- [ ] 性能优化（GPU 加速、帧采样）
- [ ] Web/API 接口（可选）

### 5.7 推荐的项目目录结构

```
highway_vision/
├── __init__.py
├── config.py              # 全局配置（模型路径、标定点、阈值等）
├── detector.py            # 车辆检测模块
├── tracker.py             # 多目标跟踪模块
├── speed_estimator.py     # 速度估算模块
├── traffic_counter.py     # 车流量统计模块
├── headway_calculator.py  # 车头间距计算模块
├── view_transform.py      # 透视变换模块
├── visualizer.py          # 可视化渲染模块
├── pipeline.py            # 主处理流水线
└── utils.py               # 工具函数
```

---

## 六、参考资源

### 6.1 项目内示例

| 示例 | 路径 | 与高速公路系统的关联 |
|------|------|---------------------|
| 速度估算 | `examples/speed_estimation/` | 直接参考 — 车速显示功能 |
| 交通流分析 | `examples/traffic_analysis/` | 直接参考 — 车流量统计功能 |
| 目标跟踪 | `examples/tracking/` | 基础参考 — 多目标跟踪流程 |
| 区域计数 | `examples/count_people_in_zone/` | 基础参考 — PolygonZone 用法 |
| 热力图与轨迹 | `examples/heatmap_and_track/` | 扩展参考 — 密度可视化 |

### 6.2 关键文件速查

| 需求 | 文件路径 |
|------|---------|
| Detections 数据类 | `src/supervision/detection/core.py` |
| LineZone / LineZoneAnnotator | `src/supervision/detection/line_zone.py` |
| PolygonZone / PolygonZoneAnnotator | `src/supervision/detection/tools/polygon_zone.py` |
| ByteTrack 跟踪器 | `src/supervision/tracker/byte_tracker/core.py` |
| 所有标注器 | `src/supervision/annotators/core.py` |
| 颜色系统 | `src/supervision/draw/color.py` |
| 绘图原语 | `src/supervision/draw/utils.py` |
| 视频工具 | `src/supervision/utils/video.py` |
| 几何原语 | `src/supervision/geometry/core.py` |
| CSV/JSON 导出 | `src/supervision/detection/tools/csv_sink.py`, `json_sink.py` |

### 6.3 外部资源

- [supervision 官方文档](https://supervision.roboflow.com/latest/)
- [Ultralytics YOLO 文档](https://docs.ultralytics.com/)
- [ByteTrack 论文](https://arxiv.org/abs/2110.06864)
- [透视变换教程](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
