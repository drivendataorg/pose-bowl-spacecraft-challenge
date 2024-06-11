pip install ultralytics==8.2.2
WANDB_DISABLED=true LOCAL_RANK=-1 RANK=-1 python export.py --model=PATH_TO_THE_TRAINED_MODEL ## Example './runs/detect/train1/weights/best.pt'