#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021-12-7 16:54
# @Author  : 26731
# @File    : train_tri.py
# @Software: PyCharm
import os
import argparse
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import prepare_data, Dataset, CloudMaskDataset
from utils import *
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from hint.networks import HINT,HINT1
from metric import *
from PIL import Image
import torchvision.transforms.functional as TF


parser = argparse.ArgumentParser(description="SWCNN")
config = get_config('configs/config.yaml')
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers(DnCNN)")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--alpha", type=float, default=1.0, help="The opacity of the watermark")
parser.add_argument("--outf", type=str, default=config['train_model_out_path_SWCNN'], help='path of model')
parser.add_argument("--net", type=str, default="HN", help='Network used in training')
parser.add_argument("--loss", type=str, default="L1", help='The loss function used for training')
parser.add_argument("--self_supervised", type=str, default="True", help='T stands for TRUE and F stands for FALSE')
parser.add_argument("--PN", type=str, default="True", help='Whether to use perception network')
parser.add_argument("--GPU_id", type=str, default="0", help='GPU_id')
parser.add_argument("--pth1", type=str, default="/mnt/d/zy/2.59/SWCNN-main_multi/pixel_pth/HNperL1n2nalpha1.0.pth", help='model1.path')
#parser.add_argument("--pth1", type=str, default="/mnt/sda/zhouying/2.13/SWCNN-main_multi/runs/HNperL1n2nalpha1.0.pth", help='model1.path')
parser.add_argument("--cloud_dir", type=str, default="/mnt/d/zy/dataset/rice2train/cloud")
parser.add_argument("--mask_dir", type=str, default="/mnt/d/zy/dataset/rice2train/mask")

opt = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_id

if opt.PN == "True":
    model_name_1 = "per"
else:
    model_name_1 = "woper"
if opt.loss == "L1":
    model_name_2 = "L1"
else:
    model_name_2 = "L2"
if opt.self_supervised == "True":
    model_name_3 = "n2n"
else:
    model_name_3 = "n2c"
tensorboard_name = opt.net + model_name_1 + model_name_2 + model_name_3 + "alpha" + str(opt.alpha)
model_name = tensorboard_name + ".pth"
def criterion(input, target, weight=0.1):
    return Loss(weight=weight)(input, target)


def main():
    # Load dataset
    print('Loading dataset ...\n')
    #dataset_train = Dataset(train=True, mode='color', data_path=config['train_data_path'])
    #dataset_val = Dataset(train=False, mode='color', data_path=config['train_data_path'])
    dataset_train = CloudMaskDataset(
        cloud_dir=opt.cloud_dir,
        mask_dir=opt.mask_dir,
        size=256
    )

    dataset_val = CloudMaskDataset(
        cloud_dir=opt.cloud_dir,
        mask_dir=opt.mask_dir,
        size=256
    )

    loader_train = DataLoader(
        dataset=dataset_train,
        num_workers=0,
        batch_size=opt.batchSize,
        shuffle=True
    )

    print("# of training samples:", len(dataset_train))
    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batchSize, shuffle=True)  # 4
    cloud_list = sorted([
        os.path.join(opt.cloud_dir, f)
        for f in os.listdir(opt.cloud_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
    ])

    mask_list = sorted([
        os.path.join(opt.mask_dir, f)
        for f in os.listdir(opt.mask_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
    ])

    assert len(cloud_list) == len(mask_list), "cloud 和 mask 数量不一致"
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # load network
    if opt.net == "HN":
        net1 = HINT1()
        net = HINT()
    else:
        assert False
    # TensorBoard was used to visually record the training results
    writer = SummaryWriter("runs/" + tensorboard_name)

    model_vgg = load_froze_vgg16()
    device_ids = [0]

    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    ckpt = torch.load(opt.pth1)
    model.load_state_dict(ckpt, strict=False)

    # load loss function
    if opt.loss == "L2":
        criterion = nn.MSELoss(size_average=False)
    else:
        criterion = nn.L1Loss(size_average=False)

    # Load the trained network and continue training
    # model.load_state_dict(torch.load(os.path.join(opt.outf, 'net_water_UNet_sec1_per0313n.pth')))
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    step = 0
    best_oIoU = -0.1


    for epoch in range(opt.epochs):

        if epoch < opt.milestone:
            current_lr = opt.lr 
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
   
            
            
        for i, data in enumerate(loader_train, 0):
            model.train()
            optimizer.zero_grad()

            imgn_train, mask = data
            imgn_train = imgn_train.cuda()
            mask = mask.cuda()

            mask_train = model(imgn_train)

            loss = criterion(mask_train, mask)
            step += 1

            if step % 10 == 0:
                with torch.no_grad():
                    mask_train_show = torch.clamp(mask_train, 0., 1.)
                    mask_show = torch.clamp(mask, 0., 1.)

                input_show = vutils.make_grid(
                    imgn_train.detach().cpu().clamp(0, 1),
                    normalize=False
                )

                gtmask_show = vutils.make_grid(
                    mask_show.detach().cpu(),
                    normalize=False
                )

                outmask_show = vutils.make_grid(
                    mask_train_show.detach().cpu(),
                    normalize=False
                )

                writer.add_scalar("loss", loss.item(), step)
                writer.add_image("input_cloud", input_show, step)
                writer.add_image("gt_mask", gtmask_show, step)
                writer.add_image("out_mask", outmask_show, step)


            """
            imgn_train, mask1 = add_watermark_noise(img_train, 40, True, random_img, alpha=opt.alpha)
            
            if opt.self_supervised == "True":
                imgn_train_2 ,mask= add_watermark_noise(img_train, 40, True, random_img, alpha=opt.alpha)
                mask =mask +mask1
                mask=mask.cuda()
            else:
                imgn_train_2 = img_train
            

            imgn_train = torch.Tensor(imgn_train)
            imgn_train_2 = torch.Tensor(imgn_train_2)
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            imgn_train_2 = Variable(imgn_train_2.cuda())

            """
            if opt.net == "FFDNet":
                noise_sigma = 0 / 255.
                noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(img_train.shape[0])]))
                noise_sigma = Variable(noise_sigma)
                noise_sigma = noise_sigma.cuda()
                out_train = model(imgn_train, noise_sigma)
            else:
                
                #imgn_train_m =torch.cat((imgn_train,imgn_train,imgn_train,imgn_train),dim=1)
                #mask_train = model(imgn_train_2)
                mask_train = model(imgn_train)   
                #imgn_train1 = torch.cat((imgn_train, mid_message,mid_message, mid_message ), dim=1)
            
               # out_train, mask_train = model(imgn_train1)




            #feature_out = model_vgg(out_train)
            feature_img = model_vgg(imgn_train)

            if opt.PN == "True":
                #loss = (1.0 * criterion(out_train, imgn_train_2) / imgn_train.size()[
                    #0] * 2) + (0.024 * criterion(feature_out, feature_img) / (feature_img.size()[0] / 2)
                    #+1.0* criterion(mask , mask_train)/ imgn_train.size()[0] * 2)

                loss = (1.0* criterion(mask , mask_train)/ imgn_train.size()[0] * 2)
            else:
                loss = (1.0 * criterion(out_train, img_train) / imgn_train.size()[
                    0] * 2) + (0.0 * criterion(feature_out, feature_img) / (feature_img.size()[0] / 2))
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            if step % 500 == 0:

                save_path = os.path.join(opt.outf, model_name)

                torch.save(
                    model.state_dict(),
                    save_path
                )

                print("Saved:", save_path)
 
            if opt.net == "FFDNet":
                out_train = torch.clamp(model(imgn_train, noise_sigma), 0., 1.)
            else:
                #imgn_train = torch.cat((imgn_train, imgn_train, imgn_train, imgn_train), dim=1)
                mask_train = torch.clamp(model(imgn_train), 0., 1.)
                
                mask = torch.clamp(mask, 0., 1.)

            #psnr_train = batch_PSNR(mask, mask_train[0], 1.)
            #print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                  #(epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
            
        ## the end of each epoch
        model.eval()
        # Save the trained network parameters
        
        # validate
        psnr_val = 0

        eval_seg_iou_list = [.5, .6, .7, .8, .9]
        seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
        seg_total = 0
        mean_IoU = []
        total_loss = 0
        total_its = 0
        acc_ious = 0

        # evaluation variables
        cum_I, cum_U = 0, 0


        with torch.no_grad():

            for k in range(len(dataset_val)):

                imgn_val, mask_val = dataset_val[k]

                imgn_val = imgn_val.unsqueeze(0).cuda()
                mask_val = mask_val.unsqueeze(0).cuda()

                eva_mask = model(imgn_val)
                eva_mask = torch.clamp(eva_mask, 0., 1.)

                iou, I, U = IoU(eva_mask, mask_val)

                loss = criterion(eva_mask, mask_val)

                total_loss += loss.item()
                acc_ious += iou
                mean_IoU.append(iou)
                cum_I += I
                cum_U += U

                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (iou >= eval_seg_iou)

                seg_total += 1

            iou = acc_ious / len(dataset_val) * 100

            writer.add_scalar("iou", iou, epoch + 1)

            print("\n[epoch %d] iou: %.4f" % (epoch + 1, iou))

            save_path = os.path.join(opt.outf, model_name)
            torch.save(model.state_dict(), save_path)
            print("Saved model to:", save_path)


    writer.close()


if __name__ == "__main__":
    # data preprocess.
    if opt.preprocess:
        prepare_data(data_path=config['train_data_path'], patch_size=256, stride=128, aug_times=1, mode='color')
    main()