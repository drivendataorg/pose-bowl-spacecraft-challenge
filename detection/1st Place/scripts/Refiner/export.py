import argparse

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export a YOLOv8 model to openvino")
    parser.add_argument("--model", type=str, default="yolov8n", help="The name of the model to export")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.export(format="openvino", imgsz=640)
