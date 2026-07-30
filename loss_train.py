import os
import warnings

# 过滤无关警告，让训练日志更简洁
warnings.filterwarnings("ignore")

# 设置仅使用第0块GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from ultralytics import YOLO

if __name__ == "__main__":
    # -------------------------- 模型加载（整合两段代码的核心逻辑） --------------------------
    # 方式1：从yaml构建模型 + 加载预训练权重（对应第一段代码）
    # model = YOLO("yolov8s.yaml")  # 从配置文件构建模型
    # model.load('yolov8s.pt')      # 加载预训练权重

    # 方式2：直接加载预训练模型（更简洁，对应第二段代码）
    model = YOLO('yolov8s')  # 等价于加载yolov8s.yaml + yolov8s.pt

    # -------------------------- 训练参数（整合+规范） --------------------------
    # 选择你要使用的IoU损失类型，取消对应注释即可
    # 1. 默认CIoU（无额外参数）
    train_params = {
        "data": "data.yaml",  # 数据集路径（第一段）
        "cache": False,
        "imgsz": 640,
        "epochs": 300,  # 总训练轮数（第二段）
        "single_cls": False,
        "batch": 32,  # 批次大小（第一段用8，避免显存溢出）
        "close_mosaic": 10,  # 训练10轮后关闭mosaic（第一段）
        "workers": 0,  # Linux系统用4（第一段），Windows改0
        "device": "0",
        "optimizer": "SGD",
        "resume": False,  # 不恢复断点（第二段）
        "amp": True,
        "project": "result/aircraft/yolov8-loss",  # 结果保存路径（第二段）
        "name": "CIoU",  # 实验名称，对应不同IoU类型
        "exist_ok": True  # 允许覆盖已有结果（最新版YOLOv8新增）
    }

    # 2. SIoU（取消下面注释，注释上面CIoU参数）
    # train_params = {
    #     "data": r"/home/ubuntu/work/ct/datasets/minidatasets/VisDrone.yaml",
    #     "cache": False,
    #     "imgsz": 640,
    #     "epochs": 300,
    #     "single_cls": False,
    #     "batch": 8,
    #     "close_mosaic": 10,
    #     "workers": 4,
    #     "device": "0",
    #     "optimizer": "SGD",
    #     "resume": False,
    #     "amp": True,
    #     "project": "result/aircraft/yolov8-loss",
    #     "name": "SIoU",
    #     "exist_ok": True,
    #     "iou_type": "Siou",
    #     "Inner_iou": False,
    #     "Focal": False,
    #     "Focaler": False
    # }

    # 3. EIoU（取消下面注释，注释上面参数）
    # train_params = {
    #     "data": r"/home/ubuntu/work/ct/datasets/minidatasets/VisDrone.yaml",
    #     "cache": False,
    #     "imgsz": 640,
    #     "epochs": 300,
    #     "single_cls": False,
    #     "batch": 8,
    #     "close_mosaic": 10,
    #     "workers": 4,
    #     "device": "0",
    #     "optimizer": "SGD",
    #     "resume": False,
    #     "amp": True,
    #     "project": "result/aircraft/yolov8-loss",
    #     "name": "EIoU",
    #     "exist_ok": True,
    #     "iou_type": "Eiou",
    #     "Inner_iou": False,
    #     "Focal": False,
    #     "Focaler": False
    # }

    # 4. AlphaIoU（取消下面注释，注释上面参数）
    # train_params = {
    #     "data": r"/home/ubuntu/work/ct/datasets/minidatasets/VisDrone.yaml",
    #     "cache": False,
    #     "imgsz": 640,
    #     "epochs": 300,
    #     "single_cls": False,
    #     "batch": 8,
    #     "close_mosaic": 10,
    #     "workers": 4,
    #     "device": "0",
    #     "optimizer": "SGD",
    #     "resume": False,
    #     "amp": True,
    #     "project": "result/aircraft/yolov8-loss",
    #     "name": "AlphaIoU",
    #     "exist_ok": True,
    #     "iou_type": "iou",
    #     "alpha": 3
    # }

    # 5. Wise-IoU（取消下面注释，注释上面参数）
    # train_params = {
    #     "data": r"/home/ubuntu/work/ct/datasets/minidatasets/VisDrone.yaml",
    #     "cache": False,
    #     "imgsz": 640,
    #     "epochs": 300,
    #     "single_cls": False,
    #     "batch": 8,
    #     "close_mosaic": 10,
    #     "workers": 4,
    #     "device": "0",
    #     "optimizer": "SGD",
    #     "resume": False,
    #     "amp": True,
    #     "project": "result/aircraft/yolov8-loss",
    #     "name": "WiseIoU",
    #     "exist_ok": True,
    #     "iou_type": "Wise-iou",
    #     "Inner_iou": False
    # }

    # -------------------------- 执行训练+验证+导出 --------------------------
    # 训练模型
    model.train(**train_params)

    # 验证模型（对应第一段代码）
    metrics = model.val()

    # # 导出ONNX模型（对应第一段代码）
    # path = model.export(format="onnx", dynamic=True)
    # print(f"模型已导出至：{path}")