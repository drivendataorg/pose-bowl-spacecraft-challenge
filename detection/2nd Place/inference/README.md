### Inference
https://github.com/drivendataorg/spacecraft-pose-object-detection-runtime

- Downloads the latest official Docker image from the container registry Azure. You'll need an internet connection for this.
```shell
sudo make pull
```

- Copy yolov8s 1280 openvino model to folder [./inference/example_src/assets](https://github.com/dungnb1333/Pose-Bowl-Spacecraft-Detection/tree/main/inference/example_src/assets)

Example: 
```shell
cp -r ./src/runs/detect/yolov8s_1280/weights/best_openvino_model ./inference/example_src/assets/.
```

- Change path at line 38 in file [./inference/example_src/main.py](https://github.com/dungnb1333/Pose-Bowl-Spacecraft-Detection/blob/main/inference/example_src/main.py?plain=1#L38)
- Put your public or private data to folder [./inference/data](https://github.com/dungnb1333/Pose-Bowl-Spacecraft-Detection/tree/main/inference/data)

Data structure should be should be
```
├── images
│   ├── *.png
└── submission_format.csv
```
- make test-submission
```shell
cd inference
sudo make clean
sudo make pack-example
sudo make test-submission
```
- If everything worked as expected, you should see a new file has been generated at [./inference/submission/submission.csv](./inference/submission/submission.csv)