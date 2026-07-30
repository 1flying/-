import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("../model/baseline2.0.pt")
    model.predict(
        source="image/image.jpg",
        project="runs/detect",
        name="baseline2.0",
        save=True,
        # visualize=True # visualize model features maps
    )
