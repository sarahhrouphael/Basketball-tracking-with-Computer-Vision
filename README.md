# 🏀 Basketball Tracking & Analytics – Computer Vision Project

## 📌 Overview

This project aims to build a full computer-vision pipeline capable of tracking players and the basketball during a game, analyzing team behavior, and generating useful visual analytics such as density maps and shot detection.  
The system focuses first on **single-camera static recordings**, and later extends to **multi-camera and broadcast-style footage**.

---

## 🎯 Objectives

- Detect players, the ball, and the basketball hoop in video frames.
- Track all players across time and assign consistent IDs.
- Identify team membership using visual appearance (e.g., jersey colors).
- Track the ball trajectory and detect important events (e.g., made baskets).
- Generate court-level analytics such as density maps and movement heatmaps.
- Prepare the pipeline for extension to multi-camera datasets.

Basketball-tracking-with-Computer-Vision/
|-- Failure/
|   |-- Court_segmentation.ipynb
|   `-- Team_Detection.ipynb
|
|-- models/
|   |-- Ball_detection/
|   |   `-- ball_hoop.pt
|   `-- Players_Detection/
|       `-- player.pt
|
|-- source/
|   `-- vid9.mp4
|
|-- final_output_video/
|   `-- tracked_with_court.mp4
|
|-- requirements.txt
|-- Ball_Possesion.py
|-- Team_classification.py
|-- Detection.ipynb
|-- PlayerTracking and Mapping.ipynb
|-- dataset_download.ipynb
|-- calib.png
|-- basketball_court.png
`-- README.md




### Environment Setup

Install all required dependencies using:
→ pip install -r requirements.txt

### Dataset & Assets Preparation

source/vid9.mp4
→ Input broadcast basketball video used for evaluation.
basketball_court.png
→ Top-down basketball court template used for projection.
calib.png
→ Frame used for manual calibration and homography estimation.


### Possible Future Improvements **
Potential enhancements include:

- Multi-camera fusion and 3D player localization  
- Pose estimation for advanced player behavior analysis  
- Automated identification of passes, rebounds, screens, and other actions  
- Real-time processing for live analytics

