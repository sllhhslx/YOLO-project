# 饮料瓶检测系统

基于 [Ultralytics YOLO11](https://docs.ultralytics.com/) 的饮料瓶目标检测项目：包含 LabelMe 标注数据、YOLO 分割数据集、检测模型训练结果，以及一个开箱即用的 Flask Web 可视化演示。

系统可识别 **3 类饮料容器**：

| 类别 ID | 类别名 | 含义 |
| --- | --- | --- |
| 0 | `milk` | 牛奶 |
| 1 | `sparkling water` | 气泡水 |
| 2 | `water` | 水（矿泉水 / 纯净水） |

---

## ✨ 功能特性

- **目标检测**：定位图片 / 视频中的饮料瓶，输出类别与置信度
- **摄像头实时演示**：服务器端逐帧推理，浏览器直接查看 MJPEG 实时画面
- **分割数据集**：同时提供实例分割格式标注，可训练 YOLO11-Seg
- **开箱即用权重**：`project1/models/` 已附带训练好的检测权重
- **完整训练记录**：训练曲线、混淆矩阵、PR 曲线、逐轮指标均已上传

---

## ⚙️ 技术栈

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| Ultralytics | 8.4（YOLO11） |
| PyTorch | 2.8 + CUDA 12.8 |
| OpenCV | 4.13 |
| Flask | 3.1 |

Web 端依赖见 `project1/requirements`：

```
flask>=2.3.0
ultralytics>=8.0.0
opencv-python>=4.8.0
pillow>=9.0.0
numpy>=1.24.0
```

---

## 📁 目录结构

```
python3.12/code/
├── images_饮料/                # 原始数据集：284 张 JPG + 同名 LabelMe JSON
├── yolo_seg_dataset/          # 分割数据集（YOLO 格式）
│   ├── data.yaml
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
├── runs/detect/train4/        # 检测训练结果
│   ├── results.csv             # 逐轮指标
│   ├── results.png             # 损失与 mAP 曲线
│   ├── confusion_matrix.png    # 混淆矩阵
│   ├── Box{P,R,F1,PR}_curve.png
│   └── weights/{best,last}.pt
└── project1/                  # Flask Web 应用
    ├── app.py                  # 后端：检测 + 分类 + 视频流
    ├── templates/index.html    # 前端页面
    ├── models/
    │   ├── detect1.pt          # 饮料检测权重（3 类）
    │   ├── classify1.pt        # 分类权重（可替换）
    │   └── yolo11n-seg.pt      # 分割预训练权重
    └── requirements
```

---

## 📊 数据集

### 原始标注（`images_饮料/`）

共 **284 张**图片，每张图片配有同名 LabelMe JSON（多边形标注），合计 **1247** 个目标：

| 类别 | 标注数量 |
| --- | --- |
| `water` | 611 |
| `sparkling water` | 474 |
| `milk` | 162 |

### 分割数据集（`yolo_seg_dataset/`）

按训练集 / 验证集 / 测试集划分后转换为 YOLO 分割格式：

| 划分 | 图片数 |
| --- | --- |
| train | 261 |
| val | 102 |
| test | 55 |
| **合计** | **418** |

`data.yaml`：

```yaml
nc: 3
names: ["milk", "sparkling water", "water"]
```

标注文件为归一化多边形点集：

```
class_id x1 y1 x2 y2 ... xn yn
```

---

## 🧠 模型与训练结果

### 权重文件

| 文件 | 用途 |
| --- | --- |
| `project1/models/detect1.pt` | 饮料检测（milk / sparkling water / water） |
| `project1/models/classify1.pt` | 分类权重，可在 Web 端展示分类结果 |
| `project1/models/yolo11n-seg.pt` | YOLO11 分割预训练权重 |
| `runs/detect/train4/weights/best.pt` | 检测训练最优权重 |

### 检测训练指标（验证集，第 10 轮）

| Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- |
| 0.993 | 0.982 | 0.979 | 0.733 |

训练产物（`runs/detect/train4/`）包含逐轮指标 `results.csv`、训练曲线 `results.png`、混淆矩阵、各类 PR/F1 曲线与验证集对比图。

---

## 🚀 运行 Web 演示

```powershell
cd python3.12/code/project1
pip install -r requirements
python app.py
```

浏览器访问 **http://127.0.0.1:5000**。

### 页面

| 标签页 | 功能 |
| --- | --- |
| 📷 图片上传 | 拖拽或选择图片 → 执行检测 + 分类，返回标注后的结果图 |
| 🎥 摄像头实时 | 服务器读取摄像头并推流，实时显示检测框与类别 |

### 接口

| 路由 | 方法 | 说明 |
| --- | --- | --- |
| `/` | GET | 渲染主页面 |
| `/upload` | POST | 表单字段 `image`，返回 `image/jpeg` 处理结果 |
| `/video_feed` | GET | MJPEG 实时视频流 |

### 主要配置（`app.py` 的 `Config`）

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `DET_MODEL_PATH` | `models/detect1.pt` | 检测模型路径 |
| `CLS_MODEL_PATH` | `models/classify1.pt` | 分类模型路径 |
| `CAMERA_INDEX` | `0` | 摄像头索引 |
| `CAMERA_BACKEND` | `cv2.CAP_DSHOW` | Windows 采集后端 |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | 640 / 480 | 采集分辨率 |
| `FRAME_SKIP` | `3` | 每 4 帧检测一次，其余帧复用结果 |
| `CONFIDENCE_THRESH` | `0.5` | 检测框置信度阈值 |

> 模型加载路径基于绝对路径，移动到其他机器时需同步修改 `Config` 中的两个 `*_MODEL_PATH`。

---

## 📄 说明

- 数据集仅用于学习与研究。
- 项目基于 Ultralytics YOLO11（AGPL-3.0），使用前请遵守其开源协议。
