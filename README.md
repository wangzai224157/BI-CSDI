
# BI-CSDI  # blind inpainting
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



SAFE dataset can be download at google drive. (https://drive.google.com/file/d/15nAMQKuvpnspykCEbYm-kRpfPMLtmHjh/view?usp=drive_link)

| Task               | Training Stage | Pre-trained Weights Link                                                                             |
|--------------------|----------------|------------------------------------------------------------------------------------------------------|
| Segmentation Task  | Stage 1        | (https://drive.google.com/file/d/1QmugbOvStmxJY3oxQZLWDFEt2DVDYGcn/view?usp=drive_link)|
|--------------------|----------------|------------------------------------------------------------------------------------------------------|
|  Inpainting Task   | Stage 1-3      | (https://drive.google.com/file/d/118SoOT70hdTVSj9CRkArrOgn-mQk-oCJ/view?usp=drive_link)|
|--------------------|----------------|------------------------------------------------------------------------------------------------------|


#Segmentation Training and Testing

This project provides a two-stage training strategy for degradation segmentation, including cloud-noise segmentation and stripe-shaped dead-pixel segmentation.

##Stage 1: Self-supervised Pretraining

In the first stage, the model is pretrained in a self-supervised manner to learn degradation-sensitive representations.

Cloud-noise segmentation pretraining
python train_c.py \
  --net HN \
  --loss L1 \
  --self_supervised True \
  --PN True \
  --alpha 1.0
Stripe-shaped dead-pixel segmentation pretraining
python train.py \
  --net HN \
  --loss L1 \
  --self_supervised True \
  --PN True \
  --alpha 1.0
##Stage 2: Supervised Fine-tuning

In the second stage, the pretrained weights from Stage 1 are loaded and the model is fine-tuned using labeled masks.

Please replace /path/to/pretrain_checkpoint with the path to the pretrained checkpoint obtained in Stage 1.

Cloud-noise segmentation fine-tuning
python train_c.py \
  --net HN \
  --loss L1 \
  --PN True \
  --alpha 1.0 \
  -pth1 /path/to/pretrain_checkpoint \
  --cloud_dir /path/to/cloud_images \
  --mask_dir /path/to/cloud_masks
Stripe-shaped dead-pixel segmentation fine-tuning
python train.py \
  --net HN \
  --loss L1 \
  --PN True \
  --alpha 1.0 \
  -pth1 /path/to/pretrain_checkpoint
Testing

After training, the segmentation models can be evaluated using the following commands.

Test cloud-noise segmentation
python test_c.py \
  --alpha 1.0 \
  --loss L1 \
  --self_supervised False \
  --PN True \
  --display True
Test stripe-shaped dead-pixel segmentation
python test.py \
  --alpha 1.0 \
  --loss L1 \
  --self_supervised False \
  --PN True \
  --display True
