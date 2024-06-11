## for nano and medium
pip install ultralytics==8.1.34
## for small ultralytics==8.1.47
WANDB_DISABLED=true LOCAL_RANK=-1 RANK=-1 python quantize.py --model=PATH_TO_THE_TRAINED_MODEL ## Example './runs/detect/train1/weights/best.pt'