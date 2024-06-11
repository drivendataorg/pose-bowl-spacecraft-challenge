import argparse

from ultralytics import YOLO

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model")

    # Add the model argument
    parser.add_argument("--model", type=str, default="yolov8n", help="The name of the model to train")

    # Parse the arguments
    args = parser.parse_args()

    # Import the YOLO library

    # Load the model with the specified name
    model = YOLO(args.model)

    # Train the model with the specified parameters
    model.train(
        data="na_all.yaml",
        epochs=100,
        imgsz=1280,
        seed=42,
        max_det=1,
        workers=8,
        momentum=0.737,
        weight_decay=0.0001,
        warmup_bias_lr=0.01,
        box=10,
        cls=3,
        mosaic=0.5,
    )
