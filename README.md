# IRPose16 Human Pose Dataset
## Overview
IRPose16 is a high-resolution infrared human-pose dataset created using low-cost reflective markers. It includes recordings from **10 participants**, totaling **100,000+ frames** with **16 keypoints** per frame. Movements are fully controlled, and annotations are generated using automated detection and temporal linking, followed by manual review. The dataset supports pose estimation, motion tracking, and low-visibility analysis.

---

## 1. Features
- Infrared recordings with reflective markers  
- 16 consistent anatomical keypoints  
- 10 participants with diverse Indian demographics  
- 100k+ annotated frames  
- No multi-camera calibration required  

---

## Folder Structure  
```
IRPose16-Human-Pose-Dataset/
├── custom_scripts/        # Custom data processing or generation scripts  
├── examples/              # Example usage / demo scripts  
├── Custom_Results/        
├── .gitignore  
└── README.md  
```

## Installation & Dependencies  

To set up the environment for using IRPose16:

### Step 1: Install [PyOrbbecSDK](https://github.com/orbbec/pyorbbecsdk/tree/main)

### Step 2: Setup IRPose16 Project Environment
```bash
git clone https://github.com/Alpha-github/IRPose16-Human-Pose-Dataset.git  
cd IRPose16-Human-Pose-Dataset  
```

## How to Reproduce / Usage

### Step 1: Camera Calibration and Data Capture

```bash
python custom_scripts/1_color_ir_recorder.py
```
### Step 2: Multimodal Homography

```bash
python custom_scripts/2_color_ir_homography.py
```
### Step 3: Sync and Save Multimodal Videos

```bash
python custom_scripts/3_sync_and_save_videos.py
```




