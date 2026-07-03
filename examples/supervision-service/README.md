# Supervision Service

基于 [supervision](https://github.com/roboflow/supervision) 的视频分析 Web 服务，包含 **FastAPI 后端** 与 **Vue 3 前端**。

支持在浏览器中上传视频、在线处理、预览并下载结果。

## 功能

| 模块 | 路径 | 说明 |
|------|------|------|
| 概览 | `/` | 控制台首页 |
| 检测跟踪 | `/track` | YOLO 检测 + ByteTrack 跟踪 |
| 速度估算 | `/speed` | 路面四点标定 + 透视变换测速 |

## 环境要求

- **Python** >= 3.10
- **Node.js** >= 22.18（前端开发/构建）
- **NVIDIA GPU**（可选，建议；CPU 也可运行但较慢）
- **Conda** 或 **uv**（推荐用于 Python 环境管理）

## 安装

### 1. Python 后端依赖

```powershell
conda create -n supervision python=3.10 -y
conda activate supervision

cd examples/supervision-service
pip install -e .
# 或手动安装：
# pip install "fastapi[standard]>=0.115.0" supervision ultralytics "jsonargparse[signatures]" imageio-ffmpeg
```

> **RTX 50 系列（5070 Ti 等）**：需安装 CUDA 12.8 版 PyTorch，否则无法使用 GPU：
>
> ```powershell
> pip uninstall torch torchvision torchaudio -y
> pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir
> ```

### 2. 前端依赖

```powershell
cd examples/supervision-service/webapp
npm install
```

## 启动方式

### 方式一：开发模式（推荐）

前后端分开运行，前端支持热更新。

**终端 1 — 启动 API**

```powershell
cd examples/supervision-service
fastapi dev app/main.py
```

API 地址：http://127.0.0.1:8000  
API 文档：http://127.0.0.1:8000/docs

**终端 2 — 启动 Web 前端**

```powershell
cd examples/supervision-service/webapp
npm run dev
```

Web 控制台：http://127.0.0.1:5173

> 开发模式下，Vite 会将 `/api`、`/health`、`/docs` 代理到 `8000` 端口，无需额外配置 CORS。

### 方式二：生产模式（单端口）

先构建前端，再由 FastAPI 托管静态页面与 API。

```powershell
# 构建前端
cd examples/supervision-service/webapp
npm run build

# 启动服务
cd ..
conda activate supervision
fastapi run app/main.py
```

访问：http://127.0.0.1:8000（Web UI + API + `/docs` 同一端口）

## 项目结构

```
supervision-service/
├── app/                    # FastAPI 后端
│   ├── main.py             # 应用入口
│   ├── routers/            # API 路由
│   └── services/           # 视频处理逻辑
├── webapp/                 # Vue 3 前端
│   ├── src/
│   └── dist/               # 构建产物（npm run build 后生成）
├── uploads/                # 上传原始视频（持久保存）
├── outputs/                # 处理结果视频（自动创建）
├── data/                   # SQLite 数据库（supervision.db）
└── models/                 # YOLO 权重（首次运行自动下载）
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/api/info` | API 元信息 |
| POST | `/api/v1/videos/track` | 检测 + 跟踪，返回 MP4 |
| POST | `/api/v1/videos/speed-estimate` | 速度估算，返回 MP4 |
| GET | `/api/v1/records/uploads` | 上传记录列表 |
| GET | `/api/v1/records/uploads/{id}` | 单条上传记录 |
| GET | `/api/v1/records/uploads/{id}/file` | 下载原始上传文件 |
| GET | `/api/v1/records/jobs` | 解析任务记录列表 |
| GET | `/api/v1/records/jobs/{id}` | 单条解析任务 |
| GET | `/api/v1/records/jobs/{id}/file` | 下载解析结果视频 |

## 数据存储

上传的视频保存在 `uploads/` 目录，元数据与解析记录写入 SQLite：

- 数据库文件：`data/supervision.db`
- 表 `uploads`：原始文件名、存储路径、大小、上传时间
- 表 `processing_jobs`：任务类型（`track` / `speed`）、状态、参数、输出路径、错误信息

服务启动时会自动建表。处理接口响应头中包含：

- `X-Upload-Id`：上传记录 ID
- `X-Job-Id`：解析任务 ID

## 使用说明

### 检测跟踪

1. 打开 `/track`
2. 上传视频
3. 调整置信度 / IOU 阈值
4. 点击「开始处理」
5. 在线预览或下载结果

### 速度估算

1. 打开 `/speed`
2. 上传视频，在首帧上依次标记四个路面角点（远端左 → 远端右 → 近端右 → 近端左）
3. 填写路面真实宽度、长度（米）
4. 点击「开始测速」

> 标定精度直接影响 km/h 读数是否准确，详见 [`examples/speed_estimation`](../speed_estimation)。

## 常见问题

**前端显示 API 离线**  
确认后端已启动：`fastapi dev app/main.py`，且终端无报错。

**处理很慢**  
检查是否在使用 GPU：`python -c "import torch; print(torch.cuda.is_available())"`  
若为 `False`，需安装带 CUDA 的 PyTorch（见上方安装说明）。

**浏览器无法播放结果视频**  
后端会在处理完成后自动将视频转码为 H.264。若仍无法播放，可点击「下载视频」用本地播放器查看。修改代码后需**重启 FastAPI** 并**重新处理视频**。

**首次运行额外下载**  
- 跟踪模块：`models/yolov8s.pt`  
- 测速模块：`models/yolo11s.pt`

## 开发命令

```powershell
# 后端
fastapi dev app/main.py

# 前端
cd webapp
npm run dev          # 开发
npm run build        # 构建
npm run type-check   # 类型检查
```
