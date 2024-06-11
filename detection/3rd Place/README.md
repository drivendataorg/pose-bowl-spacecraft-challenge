## Pose Bowl: Detection Track - 3rd place solution


### Summary

Solution consist of single Yolov8s pretrained model .The model is trained with img size 1280 for 30 epoch with batch size 32 on full data.The most important part of the solution was to export yolo model to openvino during inference which increases the inference speed 3x that allows me to train bigger model and img_size.For cross validation dataset was split into 10 folds non overlapping background_id and spacecraft_id in each fold which gives perfect correlation with lb. Due to large training time I only tweak loss weights hyperparameters (bbox, dfl) that improves cv and lb(+0.015).Smaller img_size tend to perform very poor possibly due to large no. of small objects in the data.No post processing of predictions was done.

### Hardware and time

- **Operating System**: UBUNTU 20.04
- **CPU**: AMD Ryzen Threadripper 3960X 24-Core 48 threads
- **GPU**: NVIDIA RTX A6000 48 GB
- **Training Time**: 4 hours on RTX A6000 and each epoch took around 8 minutes
- **Inference Time**: 1 hour 35 minutes(time varies depending on the time of submission)

### Run training
Run the jupyter notebook ‘yolov8s 1280 training’.

### Run Inference
Run the jupyter notebook ‘yolov8s 1280 inference’.The trained winning model is in assets directory
