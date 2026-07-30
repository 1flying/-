import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.yaml")  # build a new model from scratch
model.load('yolov8s.pt')

# Use the model
# 默认Ciou
model.train(data="data.yaml",
             imgsz=640, epochs=1, workers=0, batch=32, close_mosaic=0)

# # Siou
# model.train(data=r"/home/ubuntu/work/ct/datasets/minidatasets/VisDrone.yaml",
#              imgsz=640, epochs=1, workers=4, batch=8, close_mosaic=10,
#               iou_type="Siou", Inner_iou=False, Focal=False, Focaler=False)

# # Eiou
# model.train(data=r"/home/ubuntu/work/ct/datasets/minidatasets/VisDrone.yaml",
#              imgsz=640, epochs=1, workers=4, batch=8, close_mosaic=10,
#               iou_type="Eiou", Inner_iou=False, Focal=False, Focaler=False)

# # AlphaIoU,设置alpha即可, 仅支持iou/giou/diou/ciou
# model.train(data=r"/home/ubuntu/work/ct/datasets/minidatasets/VisDrone.yaml",
#              imgsz=640, epochs=1, workers=4, batch=8, close_mosaic=10,
#               iou_type="iou", alpha=3)

# # Wise-iou
# model.train(data=r"/home/ubuntu/work/ct/datasets/minidatasets/VisDrone.yaml",
#              imgsz=640, epochs=1, workers=4, batch=8, close_mosaic=10,
#               iou_type="Wise-iou", Inner_iou=False)

# # MPDiou
# model.train(data=r"/home/ubuntu/work/ct/datasets/minidatasets/VisDrone.yaml",
#              imgsz=640, epochs=1, workers=4, batch=8, close_mosaic=10,
#               iou_type="MPDiou", Inner_iou=False, Focaler=False)

# # Shape-iou
# model.train(data=r"/home/ubuntu/work/ct/datasets/minidatasets/VisDrone.yaml",
#              imgsz=640, epochs=1, workers=4, batch=8, close_mosaic=10,
#               iou_type="Shape-iou", Inner_iou=False, Focaler=False)

# # Powerful-iou
# model.train(data=r"/home/ubuntu/work/ct/datasets/minidatasets/VisDrone.yaml",
#              imgsz=640, epochs=1, workers=4, batch=8, close_mosaic=10,
#               iou_type="Powerful-iou")

# # Unified-iou
# model.train(data=r"/home/ubuntu/work/ct/datasets/minidatasets/VisDrone.yaml",
#              imgsz=640, epochs=1, workers=4, batch=8, close_mosaic=10,
#               iou_type="Unified-iou", Inner_iou=False, Focaler=False)

metrics = model.val()  # evaluate model performance on the validation set
path = model.export(format="onnx", dynamic=True)  # export the mode l to ONNX format

