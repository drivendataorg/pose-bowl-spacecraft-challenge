[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

[<img src='https://s3.amazonaws.com/drivendata-public-assets/{competition-image}'>](https://space-inspection.drivendata.org/)

# Pose Bowl: Spacecraft Detection and Pose Estimation Challenge

## Goal of the Competition

{Competition summary, usually a paragraph or two from the home page or results blog post}

This challenge had two tracks:

1. **Detection Track**—develop object detection solutions that identify a bounding box around a spacecraft in an image
2. **Pose Estimation Track**—identify the relative position and orientation (pose) of the chaser spacecraft camera across sequences of images.

## What's in this Repository

This repository contains code from winning competitors in the [Pose Bowl: Spacecraft Detection and Pose Estimation Challenge](https://space-inspection.drivendata.org/) on DrivenData. Code for all winning solutions are open source under the MIT License.

Solution code for the two tracks can be found in the `detection/` and `pose-estimation/` subdirectories, respectively. Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

### Detection Track

Place | Team or User | Public Score | Private Score | Summary of Model
--- | --- | ---   | ---   | ---
1   |     |       |       | {Description from the 1st place's writeup}
2   |     |       |       | {Description from the 2nd place's writeup}
3   |     |       |       | {Description from the 3rd place's writeup}

### Pose Estimation Track

Place | Team or User | Public Score | Private Score | Summary of Model
--- | --- | ---   | ---   | ---
1   | dylanliu | 1.9312 | 1.9026 | Uses OpenCV's ORB and AKAZE feature detection with brute force matching between images and homography calculations to generate inter-image rotations as features. Ensemble of fine-tuned Chronos-T5 models uses OpenCV rotations and range values to estimate pose.
2   |     |       |       | {Description from the 2nd place's writeup}
3   |     |       |       | {Description from the 3rd place's writeup}

---

**Winners Blog Post: [{winners-blog-post-title}]({winners-blog-post-url})**

**Benchmark Blog Post: [{benchmark-blog-post-title}]({benchmark-blog-post-url})**
