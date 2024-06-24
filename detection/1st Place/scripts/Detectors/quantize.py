import argparse

from ultralytics import YOLO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a YOLOv8 model")
    parser.add_argument("--model", type=str, default="yolov8n", help="The name of the model to quantize")
    args = parser.parse_args()
    model = YOLO(args.model)
    model.export(format="openvino", imgsz=1280, int8=True, data="na_all_cal.yaml")
