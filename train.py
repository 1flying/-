# import warnings
# warnings.filterwarnings("ignore")
# from ultralytics import YOLO
#
# if __name__ == "__main__":
#     model = YOLO('result/aircraft/yolov8-baseline/exp7/weights/last.pt')
#
#     model.train(
#         data="data.yaml",
#         cache=False,
#         imgsz=640,
#         epochs=200,
#         single_cls=False,
#         batch=32,
#         close_mosaic=0,
#         workers=0,
#         device="0",
#         optimizer="SGD", # 使用SGD
#         resume=True, # 如果想继续训就设置last.pt的地址
#         amp=True, # 如果出现训练损失为Nan可以关闭map
#         project="result/aircraft/yolov8-baseline",
#         name="exp"
#     )

import warnings
warnings.filterwarnings("ignore")
from ultralytics import YOLO


if __name__ == "__main__":
    # 加载已训练100epoch的权重（作为预训练模型，而非断点恢复）
    model = YOLO('yolo26n.pt')

    model.train(
        data="data.yaml",
        cache=False,
        imgsz=640,
        epochs=100,  # 新的总训练轮数
        single_cls=False,
        batch=16,
        close_mosaic=0,
        workers=0,
        device="0",
        optimizer="SGD",
        resume=False,  # 核心修改：关闭断点恢复，改为基于预训练权重继续训练
        amp=True,
        project="result/aircraft",
        name="yolov13_baseline"
    )