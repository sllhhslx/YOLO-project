import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import warnings
import logging
from typing import Tuple, Generator

warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ------------------- 配置参数（可集中修改） -------------------
class Config:
    DET_MODEL_PATH = "D:\\myvenv\\python3.12\\code\\project1\\models\\detect1.pt"
    CLS_MODEL_PATH = "D:\\myvenv\\python3.12\\code\\project1\\models\\classify1.pt"
    CAMERA_INDEX = 0                # 默认摄像头索引
    CAMERA_BACKEND = cv2.CAP_DSHOW  # Windows专用，Linux下可改为 cv2.CAP_ANY
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FRAME_SKIP = 3                  # 每 (FRAME_SKIP+1) 帧检测一次，数值越大越流畅但检测延迟增加
    CONFIDENCE_THRESH = 0.5         # 检测置信度阈值（可在模型预测时传入）
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- 全局模型加载（仅一次） -------------------
try:
    det_model = YOLO(Config.DET_MODEL_PATH)
    cls_model = YOLO(Config.CLS_MODEL_PATH)
    det_model.to(Config.DEVICE)
    cls_model.to(Config.DEVICE)
    logger.info(f"✅ 模型加载完成：检测模型 + 分类模型，设备: {Config.DEVICE}")
except Exception as e:
    logger.error(f"❌ 模型加载失败: {e}")
    raise

# ------------------- 辅助函数 -------------------
def draw_detections(image: np.ndarray, results, conf_thresh: float = Config.CONFIDENCE_THRESH) -> np.ndarray:
    """绘制检测框和标签，只显示置信度高于阈值的框"""
    if results[0].boxes is None:
        return image
    for box in results[0].boxes:
        conf = box.conf[0].item()
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0].item())
        label = f"{det_model.names[cls]} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def draw_classification(image: np.ndarray, results) -> np.ndarray:
    """在左上角显示分类结果"""
    probs = results[0].probs
    if probs is not None:
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        top1_label = cls_model.names[top1_idx]
        text = f"Class: {top1_label} ({top1_conf:.2f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        # 半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (tw + 15, th + 20), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        cv2.putText(image, text, (10, th + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image

def process_image_upload(img_array: np.ndarray) -> np.ndarray:
    """图片上传时的完整处理：检测 + 分类"""
    # 检测
    det_results = det_model(img_array, verbose=False)
    img = draw_detections(img_array.copy(), det_results)
    # 分类
    cls_results = cls_model(img_array, verbose=False)
    img = draw_classification(img, cls_results)
    return img

def process_frame_camera(img_array: np.ndarray) -> np.ndarray:
    """摄像头实时帧处理：仅检测（为流畅，不做分类）"""
    det_results = det_model(img_array, verbose=False)
    img = draw_detections(img_array.copy(), det_results)
    return img

# ------------------- Flask 路由 -------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """接收上传的图片，处理并返回结果图片"""
    if 'image' not in request.files:
        return "No image file", 400
    file = request.files['image']
    if file.filename == '':
        return "Empty filename", 400

    try:
        # 读取图片为 numpy 数组 (BGR)
        img_bytes = file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return "Invalid image file", 400

        processed_img = process_image_upload(img)
        _, buffer = cv2.imencode('.jpg', processed_img)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"图片处理出错: {e}")
        return "Internal server error", 500

def generate_video_frames() -> Generator[bytes, None, None]:
    """摄像头实时视频流生成器（仅检测，无分类）"""
    cap = None
    try:
        cap = cv2.VideoCapture(Config.CAMERA_INDEX, Config.CAMERA_BACKEND)
        if not cap.isOpened():
            logger.error("无法打开摄像头")
            # 返回错误提示帧
            error_img = np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(error_img, "Camera not found or inaccessible", (50, Config.FRAME_HEIGHT//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            while True:
                _, buffer = cv2.imencode('.jpg', error_img)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

        frame_skip = Config.FRAME_SKIP
        count = 0
        last_processed_frame = None

        while True:
            success, frame = cap.read()
            if not success:
                logger.warning("读取摄像头帧失败")
                break

            count += 1
            # 第一帧强制检测，或按间隔检测
            if last_processed_frame is None or frame_skip == 0 or count % (frame_skip + 1) == 0:
                processed = process_frame_camera(frame)
                last_processed_frame = processed
            else:
                processed = last_processed_frame  # 复用上一帧检测结果

            # 编码为 JPEG
            ret, buffer = cv2.imencode('.jpg', processed)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except Exception as e:
        logger.error(f"视频流生成异常: {e}")
    finally:
        if cap is not None:
            cap.release()
            logger.info("摄像头资源已释放")

@app.route('/video_feed')
def video_feed():
    """实时视频流 API"""
    return Response(generate_video_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------- 启动服务 -------------------
if __name__ == '__main__':
    # 生产环境关闭 debug，开启 threaded 多线程处理并发请求
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)