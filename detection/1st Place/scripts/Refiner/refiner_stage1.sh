pip install ultralytics==8.2.2
WANDB_DISABLED=true LOCAL_RANK=-1 RANK=-1 python train.py --model=yolov8m --conf=na_all_crop_syn3.yaml --epochs=30
