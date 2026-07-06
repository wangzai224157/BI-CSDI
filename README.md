
# BI-CSDI: Degradation-Aware Self-Supervised Blind Inpainting for Remote Sensing Images
## Overview

BI-CSDI is a degradation-aware blind inpainting framework for remote sensing images affected by cloud noise and stripe-shaped dead-pixel artifacts. The framework follows a segmentation-then-inpainting pipeline. It first predicts degradation-region masks and then restores corrupted regions under mask guidance.

This repository provides the implementation of the segmentation module, including:

Cloud-noise segmentation
Stripe-shaped dead-pixel segmentation
Two-stage training with self-supervised pretraining and supervised fine-tuning

## Requirements
torch==2.5.1
torchvision==0.20.1
numpy==2.2.0
opencv-python==4.10.0
Pillow==11.0.0
h5py==3.13.0
scikit-image==0.24.0
PyYAML==6.0.2
## Project Structure
├─ High-sensitivity stage 1/  # segmentation task  
│  ├─ main.py  
│  ├─ test.py  
│  └─ ...  
├─ High-sensitivity stage 2/  # segmentation task  
│  ├─ main.py  
│  ├─ test.py  
│  └─ ...  
├─ Decoupling-inpainting stage 1/  # inpainting task  
│  ├─ test.py  
│  ├─ train.py  
│  └─ ...  
└─ Decoupling-inpainting stage 2/  # inpainting task  
│  ├─ test.py  
│  ├─ test.py  
│  ├─ train.py  
│  └─ ...  
├─ Decoupling-inpainting stage 3/  # inpainting task  
│  ├─ test.py  
│  ├─ train.py  
│  └─ ...  


## Dataset and Model Checkpoints
SAFE dataset can be download at google drive. (https://drive.google.com/file/d/15nAMQKuvpnspykCEbYm-kRpfPMLtmHjh/view?usp=drive_link)

| Task               | Training Stage | Pre-trained Weights Link                                                                             |
|--------------------|----------------|------------------------------------------------------------------------------------------------------|
| Segmentation Task  | Stage 1        | (https://drive.google.com/file/d/1_Nhm7bbDx71invT6sBjQ01bGNHRsQhqE/view?usp=drive_link)|
|--------------------|----------------|------------------------------------------------------------------------------------------------------|
|  Inpainting Task   | Stage 1-3      | (https://drive.google.com/file/d/118SoOT70hdTVSj9CRkArrOgn-mQk-oCJ/view?usp=drive_link)|
|--------------------|----------------|------------------------------------------------------------------------------------------------------|


## Segmentation Training and Testing

This project provides a two-stage training strategy for degradation segmentation, including cloud-noise segmentation and stripe-shaped dead-pixel segmentation.

### Stage 1: Self-supervised Pretraining

In the first stage, the model is pretrained in a self-supervised manner to learn degradation-sensitive representations.

Cloud-noise segmentation pretraining
```bash
python train_c.py \
  --net HN \
  --loss L1 \
  --self_supervised True \
  --PN True \
  --lr  1e-5 \
  --alpha 1.0
```
Stripe-shaped dead-pixel segmentation pretraining

```bash
python train.py \
  --net HN \
  --loss L1 \
  --self_supervised True \
  --PN True \
  --lr  1e-5 \
  --alpha 1.0
```
###  Stage 2: Supervised Fine-tuning

In the second stage, the pretrained weights from Stage 1 are loaded and the model is fine-tuned using labeled masks.

Please replace /path/to/pretrain_checkpoint with the path to the pretrained checkpoint obtained in Stage 1.

Cloud-noise segmentation fine-tuning

```bash
python train_c.py \
  --net HN \
  --loss L1 \
  --PN True \
  --alpha 1.0 \
  -pth1 /path/to/pretrain_checkpoint \
  --cloud_dir /path/to/cloud_images \
  --lr  1e-4 \
  --mask_dir /path/to/cloud_masks
```
Stripe-shaped dead-pixel segmentation fine-tuning

```bash
python train.py \
  --net HN \
  --loss L1 \
  --PN True \
  --alpha 1.0 \
  --lr  1e-4 \
  -pth1 /path/to/pretrain_checkpoint
```
Testing

After training, the segmentation models can be evaluated using the following commands.

Test cloud-noise segmentation

```bash
python test_c.py \
  --alpha 1.0 \
  --loss L1 \
  --self_supervised False \
  --PN True \
  --display True
```
Test stripe-shaped dead-pixel segmentation

```bash
python test.py \
  --alpha 1.0 \
  --loss L1 \
  --self_supervised False \
  --PN True \
  --display True
```

## Inpainting Training and Testing

The inpainting task is trained in three stages. The same training script is used for all stages.

### Stage 1: Inpainting Pretraining

```bash
python train.py \
  --path /path/to/stage1_checkpoint
```

### Stage 2: Semantic Representation Learning

```bash
python train.py \
  --path /path/to/stage2_checkpoint
```

### Stage 3: Semantic-detail Decoupled Inpainting

```bash
python train.py \
  --path /path/to/stage3_checkpoint
```

### Testing

The testing command is the same for all three stages:

```bash
python test.py
```

