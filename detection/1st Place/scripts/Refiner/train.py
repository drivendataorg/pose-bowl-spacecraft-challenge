import argparse

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model")
    parser.add_argument("--model", type=str, default="yolov8m", help="The name of the model to train")
    parser.add_argument(
        "--conf",
        type=str,
        default="na_all_crop_syn3.yaml",
        help="the config file with dataset paths",
    )
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.conf,
        epochs=args.epochs,
        imgsz=640,
        seed=42,
        max_det=1,
        workers=8,
        momentum=0.737,
        weight_decay=0.0001,
        warmup_bias_lr=0.01,
        box=10,
        cls=3,
        mosaic=0.5,
        flipud=0.5,
        bgr=0.5,
        degrees=45,
    )
