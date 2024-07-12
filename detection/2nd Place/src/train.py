from ultralytics import YOLO

# train model yolov8s 1280: public LB 0.9285 private LB 0.9226
# init yolov8s model and load a pretrained model from Ultralytics https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt
model1 = YOLO('yolov8s.pt')
results1 = model1.train(data='spacecraft_data.yaml', epochs=70, imgsz=1280, batch=32, workers=8, name="yolov8s_1280", lr0=0.01, flipud=0.5, fliplr=0.5, close_mosaic=15, single_cls=True)

# For speed bonus prize
# train model yolov8n 1280: public LB 0.9173 private LB 0.9098
# init yolov8n model and load a pretrained model from Ultralytics https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
model2 = YOLO('yolov8n.pt')
results2 = model2.train(data='spacecraft_data.yaml', epochs=70, imgsz=1280, batch=32, workers=8, name="yolov8n_1280", lr0=0.01, flipud=0.5, fliplr=0.5, close_mosaic=15, single_cls=True)
