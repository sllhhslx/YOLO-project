# 忽略所有警告信息，避免警告信息干扰输出
import warnings

warnings.filterwarnings('ignore')

# 从ultralytics库导入YOLO模型类
from ultralytics import YOLO

# 主程序入口
if __name__ == '__main__':
    model = YOLO('./yolo11-cls.yaml')
    # 加载预训练权重文件yolo11s-cls.pt
    model.load('yolo11n-cls.pt')
    # 开始模型训练
    model.train(
        # 训练数据集路径
        data="D:\\myvenv\\python3.12\\code\\cls_dataset",
        # 任务类型：detect(检测), segment(分割), classify(分类), pose(姿态估计)
        task='classify',
        # 是否缓存数据集到内存，False表示不缓存
        cache=False,
        # 输入图像尺寸
        imgsz=640,
        # 训练总轮数
        epochs=30,
        # 是否使用单类别训练，False表示多类别
        single_cls=False,
        # 批次大小
        batch=4,
        # 使用的GPU设备ID，'0'表示使用第一个GPU
        device='0',
        # 优化器选择，这里使用SGD
        optimizer='SGD',
        # 是否启用自动混合精度训练，True表示启用
        amp=True,
    )