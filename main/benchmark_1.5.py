#!/usr/bin/env python


import os
import random
import sys

import cv2
from os.path import join

import torchvision
from torch import nn
from tqdm import tqdm

sys.path.append("/opt/data/xiaobin/AIDN")

from base.baseTrainer import poly_learning_rate, reduce_tensor, save_checkpoint, load_state_dict
from base.utilities import get_parser, get_logger, main_process, AverageMeter
from models.RevealNet import RevealNet
from models.imp_subnet_DeepMIH import ImpMapBlock

cfg = get_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg.test_gpu)
import torch

print(torch.cuda.device_count())

from models import get_model
import numpy as np
from utils import util
import mmcv
from dataset.torch_bicubic import imresize
import math

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

## whether or not crop border of the generated images
crop = False

## whether or not adopt lpips as one of the evaluation metrics
benchmark_lpips = False
if benchmark_lpips:
    import lpips

    loss_fn_alex = lpips.LPIPS(net='alex').cuda()

## optionally record the quantitative results
import csv


def main():
    global cfg, logger
    logger = get_logger()
    logger.info(cfg)
    logger.info("=> creating model ...")
    model = get_model(cfg, logger)
    revealNet = RevealNet(input_nc=3, output_nc=3, cfg=cfg)
    revealNet_2 = RevealNet(input_nc=3, output_nc=3, cfg=cfg)
    imp_net = ImpMapBlock()


    model = model.cuda()
    revealNet = revealNet.cuda()
    revealNet_2 = revealNet_2.cuda()
    imp_net = imp_net.cuda()

    model.summary(logger, None)

    if os.path.isfile(cfg.model_path):
        logger.info("=> loading checkpoint '{}'".format(cfg.model_path))
        checkpoint = torch.load(cfg.model_path, map_location=lambda storage, loc: storage.cpu())
        load_state_dict(model, checkpoint['state_dict'])
        load_state_dict(revealNet, checkpoint['reveal'])
        load_state_dict(revealNet_2, checkpoint['reveal_2'])
        load_state_dict(imp_net, checkpoint['imp_net'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(cfg.model_path, checkpoint['epoch']))
    else:
        raise RuntimeError("=> no checkpoint flound at '{}'".format(cfg.model_path))

    # ####################### Data Loader ####################### #
    data_names = cfg.test_dataset.split('+') if '+' in cfg.test_dataset else [cfg.test_dataset]
    for data_name in data_names:
        cfg.data_name = data_name
        cfg.data_root = cfg.test_root
        if cfg.data_name in ['Set5', 'Set14', 'BSDS100', 'urban100', 'DIV2K', 'DIV2K_valid_HR_patch']:
            from dataset.div2k import DIV2K
            test_data = DIV2K(data_list=os.path.join(cfg.test_root, 'list', data_name + '_val.txt'), training=False,
                              cfg=cfg) if cfg.evaluate else None

        else:
            raise Exception('Dataset not supported yet'.format(cfg.data_name))
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.test_batch_size,
                                                  shuffle=False, num_workers=cfg.workers, pin_memory=True,
                                                  drop_last=False)
        test_scales = cfg.test_scale.split('+') if '+' in str(cfg.test_scale) else [cfg.test_scale]
        for scale in test_scales:
            logger.info("\n=> Dataset '{}' (x{})\n".format(data_name, scale))
            test(model, revealNet, revealNet_2, imp_net, test_loader, scale=float(scale), save=cfg.save,
                 data_name=cfg.data_name)


def test(model, revealNet, revealNet_2, imp_net, test_data_loader, scale=1.5, save=False, data_name=None):
    psnr_meter_sr_1_2, psnr_meter_rev_1_4 = AverageMeter(), AverageMeter()
    ssim_meter_sr_1_2, ssim_meter_rev_1_4 = AverageMeter(), AverageMeter()

    psnr_meter_sr, psnr_meter_rev_1_2 = AverageMeter(), AverageMeter()
    ssim_meter_sr, ssim_meter_rev_1_2 = AverageMeter(), AverageMeter()
    filepath = os.path.join(cfg.save_folder, cfg.data_name + 'x' + str(scale) + '.txt')
    with torch.no_grad():
        model.eval()
        with open(filepath, 'w') as f:
            for i, batch in enumerate(tqdm(test_data_loader)):
                hr, sec = batch['img_gt'], batch['img_sec']
                sec_2 = batch['img_sec_2']
                hr, sec_gt = hr.cuda(), sec.cuda()
                sec_gt2 = sec_2.cuda()

                lr_1_4 = imresize(hr, scale=1.0 / (scale * scale)).detach()
                lr_1_2 = imresize(hr, scale=1.0 / scale).detach()

                sec = imresize(sec_gt, scale=1.0 / (scale * scale)).detach()
                sec_imp = imresize(sec_gt, scale=1.0 / scale).detach()
                sec_2 = imresize(sec_gt2, scale=1.0 / scale).detach()

                imp_map = imp_net(lr_1_2, sec_imp, sec_2)
                restored_hr, restored_hr2 = model(lr_1_4, sec, sec_2, imp_map, scale)
                recovered = revealNet(restored_hr, scale)
                recovered_2 = revealNet_2(restored_hr2, scale)

                restored_hr = torch.clamp(restored_hr, 0, 1)
                restored_hr2 = torch.clamp(restored_hr2, 0, 1)
                recovered = torch.clamp(recovered, 0, 1)
                recovered_2 = torch.clamp(recovered_2, 0, 1)

                _, _, w, h = restored_hr.shape
                dist = nn.functional.interpolate(restored_hr2, [w, h], mode="bilinear")

                rev_dist = revealNet(dist, scale)

                # =======================================
                torchvision.utils.save_image(hr, "/opt/data/xiaobin/AIDN/assets/hr/hr" + str(i) + ".jpg")
                torchvision.utils.save_image(lr_1_4, "/opt/data/xiaobin/AIDN/assets/lr/lr" + str(i) + ".jpg")
                torchvision.utils.save_image(lr_1_2, "/opt/data/xiaobin/AIDN/assets/lr2/lr" + str(i) + ".jpg")

                torchvision.utils.save_image(sec, "/opt/data/xiaobin/AIDN/assets/sec/sec" + str(i) + ".jpg")
                torchvision.utils.save_image(sec_2, "/opt/data/xiaobin/AIDN/assets/sec2/sec" + str(i) + ".jpg")
                torchvision.utils.save_image(recovered,
                                             "/opt/data/xiaobin/AIDN/assets/rec/rec" + str(i) + ".jpg")
                torchvision.utils.save_image(recovered_2,
                                             "/opt/data/xiaobin/AIDN/assets/rec2/rec" + str(i) + ".jpg")

                torchvision.utils.save_image(restored_hr, "/opt/data/xiaobin/AIDN/assets/stego/stego" + str(i) + ".jpg")
                torchvision.utils.save_image(restored_hr2,
                                             "/opt/data/xiaobin/AIDN/assets/stego2/stego" + str(i) + ".jpg")

                res_stego1 = (lr_1_2 - restored_hr) * 10
                res_stego2 = (hr - restored_hr2) * 10
                res_rec = (sec - recovered) * 10
                res_rec2 = (sec_2 - recovered_2) * 10
                torchvision.utils.save_image(res_stego1,
                                             "/opt/data/xiaobin/AIDN/assets/res-stego/res" + str(i) + ".jpg")
                torchvision.utils.save_image(res_stego2,
                                             "/opt/data/xiaobin/AIDN/assets/res-stego2/res" + str(i) + ".jpg")
                torchvision.utils.save_image(res_rec, "/opt/data/xiaobin/AIDN/assets/res-rec/res" + str(i) + ".jpg")
                torchvision.utils.save_image(res_rec2, "/opt/data/xiaobin/AIDN/assets/res-rec2/res" + str(i) + ".jpg")

                # =======================================

                ########################## CALCULATE METRIC
                restored_hr = util.tensor2img(restored_hr)
                restored_hr2 = util.tensor2img(restored_hr2)
                recovered = util.tensor2img(recovered)
                recovered_2 = util.tensor2img(recovered_2)
                sec = util.tensor2img(sec)
                sec_2 = util.tensor2img(sec_2)
                hr = util.tensor2img(hr)
                lr_1_4 = util.tensor2img(lr_1_4)
                lr_1_2 = util.tensor2img(lr_1_2)

                for j in range(imp_map.shape[1]):
                    min_val = imp_map[:, j, :].min()
                    max_val = imp_map[:, j, :].max()
                    imp_map[:, j, :] = (imp_map[:, j, :] - min_val) / (max_val - min_val)

                torchvision.utils.save_image(imp_map, "/opt/data/xiaobin/AIDN/assets/imp/imp_maps" + str(i) + ".jpg")
                imp_map = util.tensor2img(imp_map)
                rev_dist = util.tensor2img(rev_dist)

                # if crop:
                #     crop_border = math.ceil(scale)
                #     restored_hr = restored_hr[crop_border:-crop_border, crop_border:-crop_border, :]
                #     restored_hr2 = restored_hr2[crop_border:-crop_border, crop_border:-crop_border, :]
                #     recovered = recovered[crop_border:-crop_border, crop_border:-crop_border, :]
                #     recovered_2 = recovered_2[crop_border:-crop_border, crop_border:-crop_border, :]
                #     sec = sec[crop_border:-crop_border, crop_border:-crop_border, :]
                #     sec_2 = sec_2[crop_border:-crop_border, crop_border:-crop_border, :]
                #     hr = hr[crop_border:-crop_border, crop_border:-crop_border, :]
                #     lr_1_4 = lr_1_4[crop_border:-crop_border, crop_border:-crop_border, :]
                #     lr_1_2 = lr_1_2[crop_border:-crop_border, crop_border:-crop_border, :]

                psnr_meter_sr_1_2.update(util.calculate_psnr(restored_hr * 255, lr_1_2 * 255))
                psnr_meter_sr.update(util.calculate_psnr(restored_hr2 * 255, hr * 255))
                ssim_meter_sr_1_2.update(util.calculate_ssim(restored_hr * 255, lr_1_2 * 255))
                ssim_meter_sr.update(util.calculate_ssim(restored_hr2 * 255, hr * 255))

                psnr_meter_rev_1_4.update(util.calculate_psnr(recovered * 255, sec * 255))
                psnr_meter_rev_1_2.update(util.calculate_psnr(recovered_2 * 255, sec_2 * 255))
                ssim_meter_rev_1_4.update(util.calculate_ssim(recovered * 255, sec * 255))
                ssim_meter_rev_1_2.update(util.calculate_ssim(recovered_2 * 255, sec_2 * 255))

                if benchmark_lpips:
                    lpips_a = loss_fn_alex(
                        torch.from_numpy(restored_hr * 2 - 1.0).cuda()[None].permute(0, 3, 1, 2).float(),
                        torch.from_numpy(hr * 2 - 1.0).cuda()[None].permute(0, 3, 1, 2).float())
                    lpips_a = lpips_a.cpu().detach().item()
                else:
                    lpips_a = 0

                if save:
                    # f.write(test_data_loader.dataset.imgs[i].split('/')[-1].split('.')[0]+' '+'{:.2f}'.format(psnr_y_hr_)+' '+ '{:.4f}'.format(ssim_y_hr_) + ' ' +'{:.4f}'.format(lpips_a) +'_'+'{:.2f}'.format(psnr_y_lr_)+' '+ '{:.4f}'.format(ssim_y_lr_) +'\n')
                    for x, last_fix in zip([hr, sec, sec_2, lr_1_4, lr_1_2,
                                            restored_hr, restored_hr2, recovered, recovered_2, imp_map, rev_dist],
                                           ["_hr.png", "_sec.png", "_sec_2.png", "_lr_1_4.png", "_lr_1_2.png",
                                            "_stego_hr.png", "_stego_hr2.png", "_recovered.png",
                                            "_recovered_2.png", "_imp_map.png", "_rev_dist.png"]):
                        # print(111)
                        mmcv.imwrite(mmcv.rgb2bgr(np.uint8(x*255)),
                                     join(cfg.save_folder, cfg.data_name + '_x%.1f' % scale,
                                          test_data_loader.dataset.imgs[i].split('/')[-1].split('.')[0] + last_fix))
    logger.info('========> SR: \n'
                'PSNR_SR_1_2: {psnr_meter_sr_1_2.avg:.2f}\n'
                'SSIM_SR_1_2: {ssim_meter_sr_1_2.avg:.4f}\n'
                'PSNR_SR: {psnr_meter_sr.avg:.2f}\n'
                'SSIM_SR: {ssim_meter_sr.avg:.4f}\n'
                '========> Recover: \n'
                'PSNR_Rev_1_4: {psnr_meter_rev_1_4.avg:.2f}\n'
                'SSIM_Rev_1_4: {ssim_meter_rev_1_4.avg:.4f}\n'
                'PSNR_Rev_1_2: {psnr_meter_rev_1_2.avg:.2f}\n'
                'SSIM_Rev_1_2: {ssim_meter_rev_1_2.avg:.4f}\n'
                .format(psnr_meter_sr_1_2=psnr_meter_sr_1_2, ssim_meter_sr_1_2=ssim_meter_sr_1_2,
                        psnr_meter_sr=psnr_meter_sr, ssim_meter_sr=ssim_meter_sr,
                        psnr_meter_rev_1_4=psnr_meter_rev_1_4, ssim_meter_rev_1_4=ssim_meter_rev_1_4,
                        psnr_meter_rev_1_2=psnr_meter_rev_1_2, ssim_meter_rev_1_2=ssim_meter_rev_1_2
                        ))


if __name__ == '__main__':
    main()
