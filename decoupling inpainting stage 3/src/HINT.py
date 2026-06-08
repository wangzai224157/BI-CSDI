import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR
from cv2 import circle
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import wandb
import lpips
import torchvision
import time  # 保留时间库

'''
This repo is modified basing on Edge-Connect
https://github.com/knazeri/edge-connect
'''

class HINT():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 2:
            model_name = 'inpaint'

        self.debug = False
        self.model_name = model_name

        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)
        self.transf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.cal_mae = nn.L1Loss(reduction='sum')

        # train mode
        if self.config.MODE == 1:
            if self.config.MODEL == 2:
                self.train_dataset = Dataset(config, config.TRAIN_INPAINT_IMAGE_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)

        # test mode
        if self.config.MODE == 2:
            if self.config.MODEL == 2:
                print('model == 2')
                self.test_dataset = Dataset(config, config.TEST_INPAINT_IMAGE_FLIST, config.TEST_MASK_FLIST,
                                            augment=False, training=False)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        if self.config.MODEL == 2:
            self.inpaint_model.load()

    def save(self):
        if self.config.MODEL == 2:
            self.inpaint_model.save()

    def train(self):
        wandb.watch(self.inpaint_model, self.psnr, log='all', log_freq=10)
        
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)
        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.inpaint_model.train()
                if model == 2:
                    images, masks = self.cuda(*items)
                    # inpaint model
                    outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss= self.inpaint_model.process(images,masks)
                    outputs_merged = (outputs_img * masks) + (images * (1-masks))
                    print("gen_style_loss:", gen_style_loss.item())
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()

                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))
                    logs.append(('gen_content_loss', gen_content_loss.item()))
                    logs.append(('gen_style_loss', gen_style_loss.item()))

                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])
                if iteration % 10 == 0:
                        wandb.log({'gen_loss': gen_loss, 'l1_loss': gen_l1_loss, 'style_loss': gen_style_loss,
                                   'perceptual loss': gen_content_loss, 'gen_gan_loss': gen_gan_loss,
                                   'dis_loss': dis_loss}, step=iteration)

                ###################### visialization
                if iteration % 40 == 0:
                    create_dir(self.results_path)
                    inputs = (images * (1 - masks))
                    images_joint = stitch_images(
                        self.postprocess(images),
                        self.postprocess(inputs),
                        self.postprocess(outputs_img),
                        self.postprocess(outputs_merged),
                        img_per_row=1
                    )

                    path_masked = os.path.join(self.results_path,self.model_name,'masked')
                    path_result = os.path.join(self.results_path, self.model_name,'result')
                    path_joint = os.path.join(self.results_path,self.model_name,'joint')
                    name = self.train_dataset.load_name(epoch-1)[:-4]+'.png'

                    create_dir(path_masked)
                    create_dir(path_result)
                    create_dir(path_joint)

                    masked_images = self.postprocess(images*(1-masks)+masks)[0]
                    images_result = self.postprocess(outputs_merged)[0]

                    print(os.path.join(path_joint,name[:-4]+'.png'))

                    images_joint.save(os.path.join(path_joint,name[:-4]+'.png'))
                    imsave(masked_images,os.path.join(path_masked,name))
                    imsave(images_result,os.path.join(path_result,name))

                    print(name + ' complete!')
                ##############

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()
        print('\nEnd training....')

    def test(self):
        self.inpaint_model.eval()
        model = self.config.MODEL
        create_dir(self.results_path)
        cal_mean_nme = self.cal_mean_nme()

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )
        
        psnr_list = []
        ssim_list = []
        l1_list = []
        lpips_list = []
        
        # ========== 核心新增1：推理时间统计初始化 ==========
        total_inference_time = 0.0  # 总推理耗时（秒）
        inference_count = 0         # 推理样本数
        single_inference_times = [] # 存储每个样本的推理耗时（毫秒）
        
        print('here')
        index = 0
        for items in test_loader:
            images, masks = self.cuda(*items)
            index += 1

            # inpaint model
            if model == 2:
                print(masks)
                
                inputs = (images * (1 - masks))
                with torch.no_grad():
                    # ========== 核心优化：精准计时（解决原计时误差问题） ==========
                    torch.cuda.synchronize()  # 同步GPU，避免异步执行导致的计时偏差
                    start_time = time.time()  # 开始时间（秒）
                    
                    outputs_img = self.inpaint_model(images, masks)
                    
                    torch.cuda.synchronize()  # 等待GPU推理完成
                    end_time = time.time()    # 结束时间（秒）
                    
                    # 计算耗时（毫秒）
                    single_time_ms = (end_time - start_time) * 1000
                    single_inference_times.append(single_time_ms)
                    total_inference_time += (end_time - start_time)
                    inference_count += 1  # 样本数+1
                    
                    # 打印单次耗时（保留2位小数，格式更规范）
                    print(f'Test sample {index} inference time: {single_time_ms:.2f} ms')

                outputs_merged = (outputs_img * masks) + (images * (1 - masks))
                
                psnr, ssim = self.metric(images, outputs_merged)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                
                if torch.cuda.is_available():
                    pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()).cuda(), self.transf(images[0].cpu()).cuda()).item()
                    lpips_list.append(pl)
                else:
                    pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()), self.transf(images[0].cpu())).item()
                    lpips_list.append(pl)                
                
                l1_loss = torch.nn.functional.l1_loss(outputs_merged, images, reduction='mean').item()
                l1_list.append(l1_loss)

                # ========== 优化打印格式：保留小数位数，更易读 ==========
                print("psnr:{:.4f}/{:.4f}  ssim:{:.4f}/{:.4f} l1:{:.6f}/{:.6f}  lpips:{:.6f}/{:.6f}  {}".format(
                    psnr, np.average(psnr_list),
                    ssim, np.average(ssim_list),
                    l1_loss, np.average(l1_list),
                    pl, np.average(lpips_list),
                    len(ssim_list)))

                images_joint = stitch_images(
                    self.postprocess(images),
                    self.postprocess(inputs),
                    self.postprocess(outputs_img),
                    self.postprocess(outputs_merged),
                    img_per_row=1
                )

                path_masked = os.path.join(self.results_path,self.model_name,'masked4060')
                path_result = os.path.join(self.results_path, self.model_name,'result4060')
                path_joint = os.path.join(self.results_path,self.model_name,'joint4060')
                
                # ===================== ✅ 【自动保存 GT（地面真值）】 =====================
                path_gt = os.path.join(self.results_path, self.model_name, 'gt')  # GT 目录
                create_dir(path_gt)
                # ========================================================================

                name = self.test_dataset.load_name(index-1)[:-4]+'.png'

                create_dir(path_masked)
                create_dir(path_result)
                create_dir(path_joint)

                masked_images = self.postprocess(images*(1-masks)+masks)[0]
                images_result = self.postprocess(outputs_merged)[0]
                
                # ===================== ✅ 【保存 GT 图片】 =====================
                gt_image = self.postprocess(images)[0]  # 取出原图
                
                # ==============================================================

                print(os.path.join(path_joint,name[:-4]+'.png'))

                images_joint.save(os.path.join(path_joint,name[:-4]+'.png'))
                imsave(masked_images,os.path.join(path_masked,name))
                imsave(images_result,os.path.join(path_result,name))

                imsave(gt_image, os.path.join(path_gt, name))  # 保存到 gt/

                print(name + ' complete!')

        # ========== 核心新增2：推理时间汇总统计 ==========
        print('\n========== Inference Time Summary ==========')
        if inference_count > 0:
            avg_time_ms = np.average(single_inference_times)  # 平均单次耗时（毫秒）
            std_time_ms = np.std(single_inference_times)      # 耗时标准差（反映稳定性）
            min_time_ms = np.min(single_inference_times)      # 最小单次耗时
            max_time_ms = np.max(single_inference_times)      # 最大单次耗时
            fps = inference_count / total_inference_time      # 每秒处理帧数（FPS）
            
            print(f'Total test samples: {inference_count}')
            print(f'Total inference time: {total_inference_time:.4f} seconds ({total_inference_time/60:.2f} minutes)')
            print(f'Average inference time per sample: {avg_time_ms:.2f} ms (±{std_time_ms:.2f} ms)')
            print(f'Min inference time: {min_time_ms:.2f} ms | Max inference time: {max_time_ms:.2f} ms')
            print(f'FPS (Frames Per Second): {fps:.2f}')
        else:
            print('No valid test samples for inference time statistics')
        print('============================================\n')

        #torch.onnx.export(model, images_joint, 'model.onnx')
        #wandb.save('model.onnx')
        print('\nEnd Testing')
        
        # ========== 优化指标打印格式：保留小数位数 ==========
        print('edge_psnr_ave:{:.4f} edge_ssim_ave:{:.4f} l1_ave:{:.6f} lpips:{:.6f}'.format(
            np.average(psnr_list),
            np.average(ssim_list),
            np.average(l1_list),
            np.average(lpips_list)))

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            print('load the generator:')
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))
            print('finish load')

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def metric(self, gt, pre):
        pre = pre.clamp_(0, 1) * 255.0
        pre = pre.permute(0, 2, 3, 1)
        pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

        gt = gt.clamp_(0, 1) * 255.0
        gt = gt.permute(0, 2, 3, 1)
        gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

        psnr = min(100, compare_psnr(gt, pre))
        ssim = compare_ssim(gt, pre, channel_axis=2, data_range=255)

        return psnr, ssim
    
    class cal_mean_nme():
        sum = 0
        amount = 0
        mean_nme = 0

        def __call__(self, nme):
            self.sum += nme
            self.amount += 1
            self.mean_nme = self.sum / self.amount
            return self.mean_nme

        def get_mean_nme(self):
            return self.mean_nme
