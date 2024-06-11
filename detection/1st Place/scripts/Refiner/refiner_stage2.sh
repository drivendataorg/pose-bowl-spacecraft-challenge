pip install ultralytics==8.2.2
WANDB_DISABLED=true LOCAL_RANK=-1 RANK=-1 python train.py --model=PATH_TO_THE_STAGE1_MODEL --conf=na_all_crop.yaml
## EXAMPLE runs/detect/train136/weights/last.pt pay attention that we are using last.pt not best.pt