#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn

from hint.networks import HINT
from utils import *
from PIL import Image
import torchvision.transforms.functional as TF

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

config = get_config('configs/config.yaml')


parser = argparse.ArgumentParser(description="mask prediction test")

parser.add_argument('--config', type=str, default='configs/config.yaml')
parser.add_argument("--net", type=str, default="HN")
parser.add_argument("--display", type=str, default="True")
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--loss", type=str, default="L1")
parser.add_argument("--self_supervised", type=str, default="True")
parser.add_argument("--PN", type=str, default="True")

parser.add_argument(
    "--output_path",
    type=str,
    default="/mnt/d/zy/2.59/SWCNN-main_multi/result_mask_metric"
)

parser.add_argument(
    "--pth1",
    type=str,
    default="/mnt/d/zy/2.59/SWCNN-main_multi/runs/HNperL1n2nalpha1.0.pth"
)

parser.add_argument(
    "--cloud_dir",
    type=str,
    default="/mnt/d/zy/dataset/rice2test/cloud"
)

parser.add_argument(
    "--mask_dir",
    type=str,
    default="/mnt/d/zy/dataset/rice2test/mask"
)

parser.add_argument("--threshold", type=float, default=0.5)

opt = parser.parse_args()


def normalize(data):
    return data.astype(np.float32) / 255.0


def save_rgb_tensor_as_image(tensor, save_path):
    """
    tensor: [1,3,H,W], value range [0,1], RGB
    """
    img_np = tensor[0].detach().cpu().numpy()

    if img_np.shape[0] == 1:
        img_np = np.repeat(img_np, 3, axis=0)

    img_np = np.transpose(img_np[:3], (1, 2, 0))
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)


def save_mask_tensor_as_image(tensor, save_path):
    """
    tensor: [1,1,H,W] or [1,3,H,W], value range [0,1]
    保存单通道 mask。
    """
    mask_np = tensor[0].detach().cpu().numpy()

    if mask_np.shape[0] == 3:
        mask_np = mask_np.mean(axis=0)
    else:
        mask_np = mask_np[0]

    mask_np = (mask_np * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(save_path, mask_np)


def save_mask_numpy(mask_np, save_path):
    """
    mask_np: H,W, value range [0,1]
    """
    mask_save = (mask_np * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(save_path, mask_save)


def to_single_channel_mask(tensor):
    """
    输入:
        tensor: [1,C,H,W]
    输出:
        mask: [1,1,H,W]
    """
    if tensor.size(1) == 1:
        return tensor[:, :1, :, :]
    elif tensor.size(1) == 3:
        return tensor.mean(dim=1, keepdim=True)
    else:
        print(f"警告：输出通道数为 {tensor.size(1)}，默认取第一个通道作为 mask。")
        return tensor[:, :1, :, :]

def calculate_mask_metrics(pred_prob, gt_mask, threshold=0.5):
    """
    pred_prob: [1,1,H,W], 0~1
    gt_mask:   [1,1,H,W], 0~1

    返回:
        tp, tn, fp, fn
        IoU, F1, Precision, Recall, OA
    """
    pred_bin = (pred_prob > threshold).float()
    gt_bin = (gt_mask > 0.5).float()

    tp = (pred_bin * gt_bin).sum().item()
    tn = ((1 - pred_bin) * (1 - gt_bin)).sum().item()
    fp = (pred_bin * (1 - gt_bin)).sum().item()
    fn = ((1 - pred_bin) * gt_bin).sum().item()

    eps = 1e-8

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "oa": oa,
    }

def load_model(model, pth_path):
    print(f"Loading inference weight: {pth_path}")

    ckpt = torch.load(pth_path, map_location="cuda")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    missing, unexpected = model.load_state_dict(ckpt, strict=False)

    print(f"权重加载提示：缺失层 {len(missing)} 多余层 {len(unexpected)}")

    if len(missing) > 0:
        print("警告：部分网络层无对应权重，大概率 pth 路径不对或模型结构不一致。")
        print("前 10 个 missing keys:")
        for k in missing[:10]:
            print(k)

    if len(unexpected) > 0:
        print("前 10 个 unexpected keys:")
        for k in unexpected[:10]:
            print(k)

    return model


def read_cloud_as_tensor(cloud_path):
    """
    读取 cloud，转 RGB，归一化，转 [1,3,H,W]
    """
    img_bgr = cv2.imread(cloud_path, cv2.IMREAD_COLOR)
    assert img_bgr is not None, f"cloud 读取失败: {cloud_path}"

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img = normalize(img_rgb)
    img = np.expand_dims(img, axis=0)
    img = np.transpose(img, (0, 3, 1, 2))

    # 裁剪到 32 的倍数
    _, _, H, W = img.shape
    H_new = H // 32 * 32
    W_new = W // 32 * 32
    img = img[:, :, :H_new, :W_new]

    tensor = torch.from_numpy(img).float().cuda()

    return tensor, img_bgr, H_new, W_new


def read_mask_as_tensor(mask_path, H, W):
    """
    读取 GT mask，resize 到 cloud 裁剪后的尺寸，转 [1,1,H,W]
    """
    mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    assert mask_bgr is not None, f"mask 读取失败: {mask_path}"

    mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)

    # cv2.resize 参数是 (width, height)
    mask_gray = cv2.resize(mask_gray, (W, H), interpolation=cv2.INTER_NEAREST)

    mask_bin = (mask_gray > 127).astype(np.float32)

    mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0).unsqueeze(0).float().cuda()

    return mask_tensor, mask_bgr, mask_bin

def load_cloud_and_mask(index, cloud_list, mask_list, h=256, w=256):
    cloud_path = cloud_list[index % len(cloud_list)]
    mask_path = mask_list[index % len(mask_list)]

    cloud = Image.open(cloud_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    cloud = cloud.resize((w, h), Image.BILINEAR)
    mask = mask.resize((w, h), Image.NEAREST)

    cloud = TF.to_tensor(cloud)          # [3,H,W], 0~1
    mask = TF.to_tensor(mask)            # [1,H,W], 0~1

    # 只保留白色区域
    mask = (mask > 0.5).float()

    cloud = cloud.unsqueeze(0)           # [1,3,H,W]
    mask = mask.unsqueeze(0)             # [1,1,H,W]

    return cloud, mask
def water_test():
    os.makedirs(opt.output_path, exist_ok=True)

    print("Building model ...")

    if opt.net == "HN":
        net = HINT()
    else:
        raise RuntimeError("only HN support")

    model = nn.DataParallel(net, device_ids=[0]).cuda()
    model = load_model(model, opt.pth1)
    model.eval()

    cloud_files = sorted(
        glob.glob(os.path.join(opt.cloud_dir, "*.png"))
        + glob.glob(os.path.join(opt.cloud_dir, "*.jpg"))
        + glob.glob(os.path.join(opt.cloud_dir, "*.jpeg"))
        + glob.glob(os.path.join(opt.cloud_dir, "*.tif"))
    )

    mask_files = sorted(
        glob.glob(os.path.join(opt.mask_dir, "*.png"))
        + glob.glob(os.path.join(opt.mask_dir, "*.jpg"))
        + glob.glob(os.path.join(opt.mask_dir, "*.jpeg"))
        + glob.glob(os.path.join(opt.mask_dir, "*.tif"))
    )

    assert len(cloud_files) > 0, f"cloud_dir 中没有图像: {opt.cloud_dir}"
    assert len(mask_files) > 0, f"mask_dir 中没有图像: {opt.mask_dir}"
    assert len(cloud_files) == len(mask_files), "cloud 和 mask 数量不一致"

    print(f"测试样本数量: {len(cloud_files)}")

    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0

    total_tp = 0.0
    total_tn = 0.0
    total_fp = 0.0
    total_fn = 0.0

    with torch.no_grad():
        for f_index, (cloud_path, mask_path) in enumerate(zip(cloud_files, mask_files)):

            # =====================================================
            # 1. 读取 cloud 和 GT mask
            # =====================================================
            INoisy, cloud_bgr, H, W = read_cloud_as_tensor(cloud_path)
            GT_mask, mask_bgr, mask_bin_np = read_mask_as_tensor(mask_path, H, W)
            _, GT_mask_white = load_cloud_and_mask(
                index=f_index,
                cloud_list=cloud_files,
                mask_list=mask_files,
                h=H,
                w=W
            )

            GT_mask_white = GT_mask_white.cuda()

            # 用 load_cloud_and_mask 处理后的白色区域 mask 覆盖 GT_mask
            GT_mask = GT_mask_white

            # 为了后面 input_origin 保存的也是处理后的白色区域 mask
            mask_np = GT_mask[0, 0].detach().cpu().numpy()
            mask_np_uint8 = (mask_np * 255).clip(0, 255).astype(np.uint8)
            mask_bgr = cv2.cvtColor(mask_np_uint8, cv2.COLOR_GRAY2BGR)
            mask_bin_np = mask_np.astype(np.float32)

            # =====================================================
            # 2. 保存原始 mask 和 cloud
            # =====================================================
            cv2.imwrite(
                os.path.join(opt.output_path, f"input_origin_{f_index}.png"),
                mask_bgr
            )

            cv2.imwrite(
                os.path.join(opt.output_path, f"cloud_origin_{f_index}.png"),
                cloud_bgr
            )

            # 保存 resize 后的 GT mask
            save_mask_tensor_as_image(
                GT_mask,
                os.path.join(opt.output_path, f"gt_mask_resize_{f_index}.png")
            )

            # =====================================================
            # 3. 模型预测 mask
            # =====================================================
            Out = model(INoisy)
            Out = torch.clamp(Out, 0.0, 1.0)

            Out_mask = to_single_channel_mask(Out)
            Out_mask = torch.clamp(Out_mask, 0.0, 1.0)

            Out_bin = (Out_mask > opt.threshold).float()

            # =====================================================
            # 4. 打印范围
            # =====================================================
            print(f"\n===== {os.path.basename(cloud_path)} =====")
            print(f"INoisy shape: {tuple(INoisy.shape)}")
            print(f"GT_mask shape: {tuple(GT_mask.shape)}")
            print(f"Out shape: {tuple(Out.shape)}")
            print(f"Out_mask shape: {tuple(Out_mask.shape)}")

            print(f"INoisy range: {INoisy.min().item():.4f} ~ {INoisy.max().item():.4f}, mean={INoisy.mean().item():.4f}")
            print(f"GT_mask range: {GT_mask.min().item():.4f} ~ {GT_mask.max().item():.4f}, mean={GT_mask.mean().item():.4f}")
            print(f"Out_mask range: {Out_mask.min().item():.4f} ~ {Out_mask.max().item():.4f}, mean={Out_mask.mean().item():.4f}")
            print(f"Out_bin white ratio: {Out_bin.mean().item():.4f}")

            # =====================================================
            # 5. 计算 mask 和 mask 的 PSNR / SSIM / RMSE
            # =====================================================
            psnr_api = batch_PSNR(Out_mask, GT_mask, 1.0)
            ssim_api = batch_SSIM(Out_mask, GT_mask, 1.0)
            rmse_api = batch_RMSE(Out_mask, GT_mask, 1.0)

            # =====================================================
            # 6. 计算 IoU / F1 / Precision / Recall / OA
            # =====================================================
            metric_dict = calculate_mask_metrics(
                pred_prob=Out_mask,
                gt_mask=GT_mask,
                threshold=opt.threshold
            )

            total_psnr += psnr_api
            total_ssim += ssim_api
            total_rmse += rmse_api

            total_tp += metric_dict["tp"]
            total_tn += metric_dict["tn"]
            total_fp += metric_dict["fp"]
            total_fn += metric_dict["fn"]

            print(f"Mask PSNR: {psnr_api:.4f}")
            print(f"Mask SSIM: {ssim_api:.4f}")
            print(f"Mask RMSE: {rmse_api:.4f}")
            print(
                f"IoU: {metric_dict['iou']:.4f} | "
                f"F1: {metric_dict['f1']:.4f} | "
                f"Precision: {metric_dict['precision']:.4f} | "
                f"Recall: {metric_dict['recall']:.4f} | "
                f"OA: {metric_dict['oa']:.4f}"
            )

            # =====================================================
            # 7. 保存模型输入和预测结果
            # =====================================================
            if opt.display == "True":

                save_rgb_tensor_as_image(
                    INoisy,
                    os.path.join(opt.output_path, f"model_input_{f_index}.png")
                )

                save_mask_tensor_as_image(
                    Out_mask,
                    os.path.join(opt.output_path, f"model_out_mask_gray_{f_index}.png")
                )

                save_mask_tensor_as_image(
                    Out_bin,
                    os.path.join(opt.output_path, f"model_out_mask_bin_{f_index}.png")
                )

                # 保存叠加图：预测 mask 区域标红
                cloud_show = cloud_bgr[:H, :W].copy()
                pred_np = Out_bin[0, 0].detach().cpu().numpy().astype(np.uint8)

                red_layer = np.zeros_like(cloud_show)
                red_layer[:, :, 2] = 255  # BGR 中红色通道

                mask_bool = pred_np.astype(bool)
                alpha = 0.45

                cloud_show[mask_bool] = (
                    cloud_show[mask_bool].astype(np.float32) * (1 - alpha)
                    + red_layer[mask_bool].astype(np.float32) * alpha
                ).astype(np.uint8)

                cv2.imwrite(
                    os.path.join(opt.output_path, f"overlay_pred_red_{f_index}.png"),
                    cloud_show
                )

    # =====================================================
    # 8. 平均指标
    # =====================================================
    cnt = len(cloud_files)

    avg_psnr = total_psnr / cnt
    avg_ssim = total_ssim / cnt
    avg_rmse = total_rmse / cnt

    eps = 1e-8

    avg_precision = total_tp / (total_tp + total_fp + eps)
    avg_recall = total_tp / (total_tp + total_fn + eps)

    avg_f1 = (
        2 * avg_precision * avg_recall
        /
        (avg_precision + avg_recall + eps)
    )

    avg_iou = total_tp / (
        total_tp + total_fp + total_fn + eps
    )

    avg_oa = (
        total_tp + total_tn
    ) / (
        total_tp + total_tn + total_fp + total_fn + eps
    )

    print("\n========== 全部测试集 Global 指标：Mask vs Mask ==========")
    print(f"Mask PSNR      : {avg_psnr:.4f}")
    print(f"Mask SSIM      : {avg_ssim:.4f}")
    print(f"Mask RMSE      : {avg_rmse:.4f}")
    print(f"Mask IoU       : {avg_iou:.4f}")
    print(f"Mask F1        : {avg_f1:.4f}")
    print(f"Mask Precision : {avg_precision:.4f}")
    print(f"Mask Recall    : {avg_recall:.4f}")
    print(f"Mask OA        : {avg_oa:.4f}")
    print("=========================================================")

    print(f"\n结果保存到: {opt.output_path}")


if __name__ == "__main__":
    water_test()