# Pose-Bowl-Spacecraft-Detection

![Alt text](./images/nasa-spacecraft-header.jpg?raw=true "Optional Title")

2nd place solution for [Pose Bowl: Spacecraft Detection Challenge](https://www.drivendata.org/competitions/260/spacecraft-detection/leaderboard/)

### General
* **Competition Purpose:** identify the boundaries of generic spacecraft in photos.
* **Type:** Object detection
* **Host:** NASA
* **Platform:** Drivendata
* **Competition link:** https://www.drivendata.org/competitions/260/spacecraft-detection/
* **Placement:** 2nd (2/651)
* **User Name:** dungnb
* **Solution:** [SolutionDocumentation_SpacecraftDetection_DungNB.pdf](https://github.com/dungnb1333/Pose-Bowl-Spacecraft-Detection/blob/main/report/SolutionDocumentation_SpacecraftDetection_DungNB.pdf)

The key of solution is based on synthetic data generation and yolov8 model.

### System
* **Operating System:** Ubuntu 22.04
* **Nvidia Driver Version:** 545.23.08
* **Cuda:** Version 11.3
* **GPU:** 1xNvidia A100 40GB
* **RAM:** 64GB
* **Train duration:** yolov8s 1280 (~60 hours), yolov8n 1280 (~24hours)
* **Inference duration:** yolov8s 1280(1 hour 40 minutes), yolov8n 1280(46 minutes)

### Environment
```shell
conda create -n venv python=3.9.6
conda activate venv
conda install pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Data preparation
- Download [competition dataset](https://www.drivendata.org/competitions/260/spacecraft-detection/data/) and extract to folder [./dataset/drivendata](./dataset/drivendata)
- Download public dataset in [paper](https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/papers/Dung_A_Spacecraft_Dataset_for_Detection_Segmentation_and_Parts_Recognition_CVPRW_2021_paper.pdf) at [repo](https://github.com/Yurushia1998/SatelliteDataset) or [google drive](https://drive.google.com/drive/u/0/folders/1Q1wR9aBFCyeFEYa3wwyXNu9wk_fZdzUm) to folder [./dataset/satellite_external](./dataset/satellite_external)\
Note: This data has been confirmed at [discussion](https://community.drivendata.org/t/external-dataset-use-detection-track/10642)

- dataset structure should be [./dataset/dataset_structure.txt](./dataset/dataset_structure.txt)

And run following scripts

```shell
cd src
python prepare_drivendata_spacecraft.py
python prepare_external_data.py
python generate_synthetic_data.py
```

### Train model
Change line 1 in file [./src/spacecraft_data.yaml](https://github.com/dungnb1333/Pose-Bowl-Spacecraft-Detection/blob/main/src/spacecraft_data.yaml?plain=1#L1) to **absolute path** of folder [./dataset/](./dataset) in your system

And run following scripts
```shell
cd src
rm -rf runs
python train.py
```
### Export model to openvino
Change path of trained model in file [./src/export_to_openvino.py](src/export_to_openvino.py)
```shell
cd src
python export_to_openvino.py
```

### Inference
[Step by step to generate submission file](inference/README.md)

### Result
|              | Public LB | Private LB | Runtime |
| :----------- | :---- | :---- | :---- |
| [yolov8s 1280](https://github.com/dungnb1333/Pose-Bowl-Spacecraft-Detection/releases/download/V1.0/yolov8s_1280.zip) | 0.9285 | 0.9226 | 1 hour 40 minutes |
| [yolov8n 1280](https://github.com/dungnb1333/Pose-Bowl-Spacecraft-Detection/releases/download/V1.0/yolov8n_1280.zip) | 0.9173 | 0.9098 | 46 minutes |
