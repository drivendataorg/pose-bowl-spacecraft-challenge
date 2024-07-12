from ultralytics import YOLO

yolov8s_model = YOLO('runs/detect/yolov8s_1280/weights/best.pt')
# Export the model to openvino
yolov8s_model.export(format='openvino', nms=True, imgsz=1280)


yolov8n_model = YOLO('runs/detect/yolov8n_1280/weights/best.pt')
# Export the model to openvino
yolov8n_model.export(format='openvino', nms=True, imgsz=1280)
