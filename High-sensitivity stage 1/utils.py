import math
import string

import torch
import torch.nn as nn
import numpy as np
import cv2
# from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import random

from PIL import Image
import matplotlib.pyplot as plt


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def batch_SSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    Img = np.transpose(Img, (0, 2, 3, 1))
    Iclean = np.transpose(Iclean, (0, 2, 3, 1))
    # print(Iclean.shape)
    for i in range(Img.shape[0]):
        SSIM += compare_ssim(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range,
                             channel_axis=-1)
    return (SSIM / Img.shape[0])


def batch_RMSE(img, imclean, data_range):
    img = img * 255
    imclean = imclean * 255
    Img = img.data.cpu().numpy().astype(np.uint8)

    Iclean = imclean.data.cpu().numpy().astype(np.uint8)
    MSE = 0
    for i in range(Img.shape[0]):
        MSE += math.sqrt(compare_mse(Iclean[i, :, :, :], Img[i, :, :, :]))
    return (MSE / Img.shape[0])
def multi( img_train ):
    random_img = random.randint(1, 12)

def add_watermark_noise(img_train, occupancy=50, self_surpervision=False, same_random=0, alpha=0.3):
    # 加载水印,水印应该是随机加入
    random_img = random.randint(1, 2)
    # 对比实验的时候选取某个水印进行去除
    # random_img = 3  # "test"  # random.randint(1, 173)
    # Noise2Noise要确保类标和输入的水印为同一张
    if self_surpervision:
        random_img = same_random
    data_path = "/mnt/sda/zhouying/NewData/strip"
    watermark = Image.open(data_path + '/'+str(random_img) + ".png")
    watermark = watermark.convert("RGBA")
    w, h = watermark.size
    # 设置水印透明度
    alpha = random.uniform(0.7, 1)
    # 遍历水印的每个像素，调整透明度
    for i in range(w):
        for k in range(h):
            color = watermark.getpixel((i, k))
            if color[3] != 0:
                transparence = int(255 * alpha)
                color = color[:-1] + (transparence,)
            watermark.putpixel((i, k), color)
    # 将水印转换为numpy数组，并进行预处理
    watermark_np = np.array(watermark)
    watermark_np = watermark_np[:, :, 0:3]
    img_train = img_train.numpy()
    imgn_train = img_train
    # 数据归一化
    _, water_h, water_w = watermark_np.shape
    occupancy = np.random.uniform(0, occupancy)
    _, _, img_h, img_w = img_train.shape
    # 加载计算占有率的数组
    img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
    img_for_cnt = Image.fromarray(img_for_cnt)
    new_w, new_h = watermark.size
    img_train = np.ascontiguousarray(np.transpose(img_train, (0, 2, 3, 1)))
    imgn_train = np.ascontiguousarray(np.transpose(imgn_train, (0, 2, 3, 1)))
    
    # 设置最大循环次数，防止死循环
    max_attempts = 100
    attempt = 0
    
    for i in range(len(img_train)):
        tmp = Image.fromarray((img_train[i] * 255).astype(np.uint8))
        tmp = tmp.convert("RGBA")
        img_for_cnt = Image.fromarray(np.zeros((img_h, img_w, 3), np.uint8))
        
        while attempt < max_attempts:
            # 随机选取放缩比例和旋转角度
            angle = random.randint(-45, 45)
            scale = np.random.uniform(0.5, 1.0)  # 调整scale范围，增加水印大小
            # 旋转水印
            rotated_watermark = watermark.rotate(angle, expand=1)
            # 放缩水印
            water = rotated_watermark.resize((int(w * scale), int(h * scale)))
            # 创建透明层
            layer = Image.new("RGBA", tmp.size, (0, 0, 0, 0))
            # 随机选取粘贴位置，确保不超过图片尺寸
            max_x = img_w 
            max_y = img_h 
            
            # 水印太大，无法粘贴，重新调整scale
            scale = np.random.uniform(0.5, 1.0)
            water = rotated_watermark.resize((int(w * scale), int(h * scale)))

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            # 粘贴水印
            layer.paste(water, (x, y))
            tmp = Image.composite(layer, tmp, layer)
            # 计算水印覆盖区域
            img_for_cnt.paste(water, (x, y), water)
            img_for_cnt = img_for_cnt.convert("L")
            img_cnt = np.array(img_for_cnt)
            sum = (img_cnt > 0).sum()
            ratio = img_w * img_h * occupancy / 100
            if sum > ratio:
                img_rgb = np.array(tmp).astype(np.float64) / 255.
                img_train[i] = img_rgb[:, :, [0, 1, 2]]
                attempt = 0  # 重置尝试次数
                break
            else:
                attempt += 1
        else:
            # 超过最大尝试次数，无法满足条件，退出循环
            break
    img_train = np.transpose(img_train, (0, 3, 1, 2))
    return img_train




import random
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter


def generate_random_cloud_mask(img_h, img_w, occupancy=20):
    """
    生成较小面积的团簇状云 mask
    白色区域表示云层
    黑色区域表示无云
    """

    target_ratio = np.random.uniform(2, occupancy) / 100.0

    max_attempts = 100

    for _ in range(max_attempts):

        mask = Image.new("L", (img_w, img_h), 0)
        draw = ImageDraw.Draw(mask)

        # 云团数量减少
        cloud_num = random.randint(3, 10)

        for _ in range(cloud_num):

            cx = random.randint(0, img_w)
            cy = random.randint(0, img_h)

            # 云团半径减小
            rx = random.randint(
                max(3, img_w // 40),
                max(4, img_w // 12)
            )

            ry = random.randint(
                max(3, img_h // 40),
                max(4, img_h // 12)
            )

            x1 = cx - rx
            y1 = cy - ry
            x2 = cx + rx
            y2 = cy + ry

            draw.ellipse(
                [x1, y1, x2, y2],
                fill=random.randint(180, 255)
            )

        # 模糊半径减小
        blur_radius = random.uniform(4, 12)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        mask_np = np.array(mask).astype(np.float32)

        mask_np = mask_np / (mask_np.max() + 1e-8)

        # 阈值提高，保留更少云区域
        threshold = np.random.uniform(0.45, 0.75)
        binary_mask = (mask_np > threshold).astype(np.float32)

        current_ratio = binary_mask.mean()

        if current_ratio >= target_ratio:
            return mask_np, binary_mask

    return mask_np, binary_mask


def add_cloud_noise(
        img_train,
        occupancy=12,
        self_surpervision=False,
        same_random=0,
        alpha=0.6):
    """
    随机添加云噪声

    img_train: torch tensor, shape [B, C, H, W], range [0, 1]
    occupancy: 最大云覆盖率百分比
    alpha: 云层透明度
    """

    if isinstance(img_train, torch.Tensor):
        img_train = img_train.detach().cpu().numpy()

    img_train = img_train.copy()

    b, c, img_h, img_w = img_train.shape

    img_train = np.transpose(img_train, (0, 2, 3, 1))

    for i in range(b):

        image = img_train[i]

        # 生成随机云 mask
        soft_mask, binary_mask = generate_random_cloud_mask(
            img_h,
            img_w,
            occupancy=occupancy
        )

        # soft_mask: [H, W] -> [H, W, 1]
        soft_mask = soft_mask[:, :, np.newaxis]

        # 随机云颜色，接近白色或灰白色
        cloud_color = np.ones_like(image)
        cloud_intensity = np.random.uniform(0.75, 1.0)
        cloud_color = cloud_color * cloud_intensity

        # 随机透明度
        cur_alpha = np.random.uniform(0.4, alpha)

        # 云层叠加
        image_cloud = (
            image * (1 - soft_mask * cur_alpha)
            +
            cloud_color * (soft_mask * cur_alpha)
        )

        image_cloud = np.clip(image_cloud, 0, 1)

        img_train[i] = image_cloud

    img_train = np.transpose(img_train, (0, 3, 1, 2))

    return img_train


"""




def add_watermark_noise(img_train, occupancy=50, self_surpervision=False, same_random=0, alpha=0.3):
    # 加载水印,水印应该是随机加入
    random_img = random.randint(1, 2)
    # 对比实验的时候选取某个水印进行去除
    # random_img = 3  # "test"  # random.randint(1, 173)
    # Noise2Noise要确保类标和输入的水印为同一张
    if self_surpervision:
        random_img = same_random
    data_path = "/mnt/sda/zhouying/NewData/strip"
    watermark = Image.open(data_path + '/'+str(random_img) + ".png")
    watermark = watermark.convert("RGBA")
    w, h = watermark.size
    # 设置水印透明度
    alpha = random.uniform(0.7,1)
    #sum=0
    for i in range(w):
        for k in range(h):
            color = watermark.getpixel((i, k))
            if color[3] != 0:
                transparence = int(255 * alpha)
                # color = color[::-1]

                color = color[:-1] + (transparence,)
            watermark.putpixel((i, k), color)
    # watermark = watermark.convert("RGB")
    watermark_np = np.array(watermark)
    watermark_np = watermark_np[:, :, 0:3]
    img_train = img_train.numpy()
    # img_train = Image.fromarray(img_train)
    imgn_train = img_train
    # 数据归一化
    _, water_h, water_w = watermark_np.shape
    occupancy = np.random.uniform(0, occupancy)

    _, _, img_h, img_w = img_train.shape
    # 加载计算占有率的数组
    img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
    # 转成PIL
    img_for_cnt = Image.fromarray(img_for_cnt)
    new_w, new_h = watermark.size
    img_train = np.ascontiguousarray(np.transpose(img_train, (0, 2, 3, 1)))
    imgn_train = np.ascontiguousarray(np.transpose(imgn_train, (0, 2, 3, 1)))

    for i in range(len(img_train)):
        tmp = Image.fromarray((img_train[i] * 255).astype(np.uint8))
        tmp = tmp.convert("RGBA")
        img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
        # 转成PIL
        img_for_cnt = Image.fromarray(img_for_cnt)
        while True:
            # 随机选取放缩比例和旋转角度
            angle = random.randint(-45, 45)
            scale = np.random.uniform(0.3, 0.6)
            # 原本的是（0.5，1）
            # scale = 1.5
            # 旋转水印
            watermark = watermark.rotate(angle, expand=1)
            #  放缩水印
            water = watermark.resize((int(w * scale), int(h * scale)))
            # 将噪声转换为PIL
            layer = Image.new("RGBA", tmp.size, (0, 0, 0, 0))
            # 随机选取要粘贴的部位
            
            #下面是原版，但是
            #x = random.randint(0, img_w - int(w * scale))  # int(-w * scale)
            #y = random.randint(0, img_h - int(h * scale))  # int(-h * scale)
            x = random.randint(0, img_w - int(w * scale)+500)  # int(-w * scale)
            y = random.randint(0, img_h - int(h * scale)+500)  # int(-h * scale)
            #print(x,y)
                        

            layer.paste(water, (x, y))
            tmp = Image.composite(layer, tmp, layer)

            img_for_cnt.paste(water, (x, y), water)
            img_for_cnt = img_for_cnt.convert("L")
            img_cnt = np.array(img_for_cnt)
            sum = (img_cnt > 0).sum()
            ratio = img_w * img_h * occupancy / 150
            if sum > ratio:
                img_rgb = np.array(tmp).astype(np.float64) / 255.
                img_train[i] = img_rgb[:, :, [0, 1, 2]]
                break
    img_train = np.transpose(img_train, (0, 3, 1, 2))
    return img_train


    """



"""

            
            # Overlay the watermark

            # From here onward, the following code is the original version.
            layer.paste(water, (x, y))
            tmp = Image.composite(layer, tmp, layer)
            img_for_cnt.paste(water, (x, y), water)
            img_for_cnt = img_for_cnt.convert("L")
            #img_cnt = np.array(img_for_cnt)
            #sum = (img_cnt > 0).sum()
            sum1 =random.randint(100,800)
            sum = sum +sum1


            ratio = img_w * img_h * occupancy / 100
            if sum > ratio:
                img_rgb = np.array(tmp).astype(np.float64) / 255.
                img_train[i] = img_rgb[:, :, [0, 1, 2]]
                break
    img_train = np.transpose(img_train, (0, 3, 1, 2))
    return img_train
  """  

            




def add_watermark_noise_B(img_train, occupancy=50, self_surpervision=False, same_random=0, alpha=0.3):
    # 加载水印,水印应该是随机加入
    # random_img = random.randint(1, 13)
    # 对比实验的时候选取某个水印进行去除
    random_img = 3  # "test"  # random.randint(1, 173)
    # Noise2Noise要确保类标和输入的水印为同一张
    if self_surpervision:
        random_img = same_random
    data_path = "watermark/translucence/"
    watermark = Image.open(data_path + str(random_img) + ".png")
    watermark = watermark.convert("RGBA")
    w, h = watermark.size
    # 设置水印透明度
    #下面是原始版本
    alpha = 0.3 + random.randint(0, 70) * 0.01
    #alpha = random.uniform(0.7,1)
    for i in range(w):
        for k in range(h):
            color = watermark.getpixel((i, k))
            if color[3] != 0:
                transparence = int(255 * alpha)
                # color = color[::-1]
                color = color[:-1] + (transparence,)
            watermark.putpixel((i, k), color)
    # watermark = watermark.convert("RGB")
    watermark_np = np.array(watermark)
    watermark_np = watermark_np[:, :, 0:3]
    img_train = img_train.numpy()
    # img_train = Image.fromarray(img_train)
    imgn_train = img_train
    # 数据归一化
    _, water_h, water_w = watermark_np.shape
    occupancy = np.random.uniform(0, occupancy)

    _, _, img_h, img_w = img_train.shape
    # 加载计算占有率的数组
    img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
    # 转成PIL
    img_for_cnt = Image.fromarray(img_for_cnt)
    new_w, new_h = watermark.size
    img_train = np.ascontiguousarray(np.transpose(img_train, (0, 2, 3, 1)))
    imgn_train = np.ascontiguousarray(np.transpose(imgn_train, (0, 2, 3, 1)))

    for i in range(len(img_train)):
        tmp = Image.fromarray((img_train[i] * 255).astype(np.uint8))
        tmp = tmp.convert("RGBA")
        img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
        # 转成PIL
        img_for_cnt = Image.fromarray(img_for_cnt)
        while True:
            # 随机选取放缩比例和旋转角度
            angle = random.randint(-45, 45)
            scale = np.random.uniform(0.5, 1.0)
            # scale = 1.5
            # 旋转水印
            # img = watermark.rotate(angle, expand=1)
            #  放缩水印
            water = watermark.resize((int(w * scale), int(h * scale)))
            # 将噪声转换为PIL
            layer = Image.new("RGBA", tmp.size, (0, 0, 0, 0))
            # 随机选取要粘贴的部位
            x = random.randint(0, img_w - int(w * scale))  # int(-w * scale)
            y = random.randint(0, img_h - int(h * scale))  # int(-h * scale)
            # 合并水印文件
            layer.paste(water, (x, y))
            tmp = Image.composite(layer, tmp, layer)

            img_for_cnt.paste(water, (x, y), water)
            img_for_cnt = img_for_cnt.convert("L")
            img_cnt = np.array(img_for_cnt)
            sum = (img_cnt > 0).sum()
            ratio = img_w * img_h * occupancy / 60
            if sum > ratio:
                img_rgb = np.array(tmp).astype(np.float) / 255.
                img_train[i] = img_rgb[:, :, [0, 1, 2]]
                break
    img_train = np.transpose(img_train, (0, 3, 1, 2))
    return img_train



#  这个函数只用来测试
def add_watermark_noise_test(img_train, occupancy=50, img_id=19, scale_img=1.5, self_surpervision=False,
                                same_random=0, alpha=0.3):
    # 加载水印,水印应该是随机加入
    # random_img = random.randint(1, 13)
    # 对比实验的时候选取某个水印进行去除
    print(img_id)
    random_img = img_id  # "test"  # random.randint(1, 173)
    # Noise2Noise要确保类标和输入的水印为同一张
    if self_surpervision:
        random_img = same_random
    data_path = "/mnt/sda/zhouying/NewData/strip/"
    watermark = Image.open(data_path + str(random_img) + ".png")
    watermark = watermark.convert("RGBA")
    w, h = watermark.size
    # 设置水印透明度
    for i in range(w):
        for k in range(h):
            color = watermark.getpixel((i, k))
            if color[3] != 0:
                #transparence = int(255 * alpha)  # random.randint(100)#这个是透明的
                transparence = int(255 * 1)  #这是不透明的

                color = color[:-1] + (transparence,)
            watermark.putpixel((i, k), color)
    # watermark = watermark.convert("RGB")
    watermark_np = np.array(watermark)
    watermark_np = watermark_np[:, :, 0:3]
    img_train = img_train.numpy()
    # img_train = Image.fromarray(img_train)
    imgn_train = img_train
    # 数据归一化
    _, water_h, water_w = watermark_np.shape
    occupancy = np.random.uniform(0, occupancy)

    _, _, img_h, img_w = img_train.shape
    # 加载计算占有率的数组
    img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
    # 转成PIL
    img_for_cnt = Image.fromarray(img_for_cnt)
    new_w, new_h = watermark.size
    img_train = np.ascontiguousarray(np.transpose(img_train, (0, 2, 3, 1)))
    imgn_train = np.ascontiguousarray(np.transpose(imgn_train, (0, 2, 3, 1)))

    for i in range(len(img_train)):
        tmp = Image.fromarray((img_train[i] * 255).astype(np.uint8))
        tmp = tmp.convert("RGBA")
        img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
        # 转成PIL
        img_for_cnt = Image.fromarray(img_for_cnt)
        while True:
            # 随机选取放缩比例和旋转角度
            angle = random.randint(-45, 45)
            scale = np.random.uniform(0.1, 0.5)
            scale = scale_img
            # 旋转水印
            # img = watermark.rotate(angle, expand=1)
            #  放缩水印
            water = watermark.resize((int(w * scale), int(h * scale)))
            # 将噪声转换为PIL
            layer = Image.new("RGBA", tmp.size, (0, 0, 0, 0))
            # 随机选取要粘贴的部位
            #print(img_w,w)
            #下面这两句 原作者也没写好
            # x = random.randint(0, img_w - int(w * scale))  # int(-w * scale)
            #y = random.randint(0, img_h - int(h * scale))  # int(-h * scale)
            x = 1
            y = 1
            # 合并水印文件
            layer.paste(water, (x, y))
            tmp = Image.composite(layer, tmp, layer)

            img_for_cnt.paste(water, (x, y), water)
            img_for_cnt = img_for_cnt.convert("L")
            img_cnt = np.array(img_for_cnt)
            sum = (img_cnt > 0).sum()
            ratio = img_w * img_h * occupancy / 100
            if sum > ratio:
                img_rgb = np.array(tmp).astype(np.float64) / 255.
                img_train[i] = img_rgb[:, :, [0, 1, 2]]
                break
    img_train = np.transpose(img_train, (0, 3, 1, 2))
    
    return img_train


import torchvision.models as models
from models import VGG16


def load_froze_vgg16():
    # finetunning
    model_pretrain_vgg = models.vgg16(pretrained=True)

    # load VGG16
    net_vgg = VGG16()
    model_dict = net_vgg.state_dict()
    pretrained_dict = model_pretrain_vgg.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # load parameters
    net_vgg.load_state_dict(pretrained_dict)

    for child in net_vgg.children():
        for p in child.parameters():
            p.requires_grad = False
    device_ids = [0]

    model_vgg = nn.DataParallel(net_vgg, device_ids=device_ids).cuda()


    return model_vgg

def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))
import yaml


# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)
