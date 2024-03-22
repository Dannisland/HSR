#!/usr/bin/env python


import os
import sys

import cv2
from os.path import join

from torch import nn
from tqdm import tqdm

sys.path.append("/opt/data/xiaobin/AIDN")

from base.utilities import get_parser, get_logger, AverageMeter
from models.RevealNet import RevealNet

cfg = get_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg.test_gpu)
import torch

print(torch.cuda.device_count())

from models import get_model
import numpy as np
from utils import util
from base.baseTrainer import load_state_dict
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
    revealNet_3 = RevealNet(input_nc=3, output_nc=3, cfg=cfg)

    model = model.cuda()
    revealNet = revealNet.cuda()
    revealNet_2 = revealNet_2.cuda()
    revealNet_3 = revealNet_3.cuda()
    model.summary(logger, None)

    if os.path.isfile(cfg.model_path):
        logger.info("=> loading checkpoint '{}'".format(cfg.model_path))
        checkpoint = torch.load(cfg.model_path, map_location=lambda storage, loc: storage.cpu())
        load_state_dict(model, checkpoint['state_dict'])
        load_state_dict(revealNet, checkpoint['reveal'])
        load_state_dict(revealNet_2, checkpoint['reveal_2'])
        load_state_dict(revealNet_3, checkpoint['reveal_3'])
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
            test(model, revealNet, revealNet_2, revealNet_3, test_loader, scale=float(scale), save=cfg.save,
                 data_name=cfg.data_name)


def test(model, revealNet, revealNet_2, revealNet_3, test_data_loader, scale=4, save=False, data_name=None):
    psnr_meter_sr_1_4, psnr_meter_rev_1_8 = AverageMeter(), AverageMeter()
    ssim_meter_sr_1_4, ssim_meter_rev_1_8 = AverageMeter(), AverageMeter()

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
                sec_2, sec_3 = batch['img_sec_2'], batch['img_sec_3']
                hr, sec = hr.cuda(), sec.cuda()
                sec_2, sec_3 = sec_2.cuda(), sec_3.cuda()

                lr_1_8 = imresize(hr, scale=1.0 / (scale * scale * scale)).detach()
                lr_1_4 = imresize(hr, scale=1.0 / (scale * scale)).detach()
                lr_1_2 = imresize(hr, scale=1.0 / scale).detach()

                sec = imresize(sec, scale=1.0 / (scale * scale * scale)).detach()
                sec_2 = imresize(sec_2, scale=1.0 / (scale * scale)).detach()
                sec_3 = imresize(sec_3, scale=1.0 / scale).detach()

                restored_hr, restored_hr2, restored_hr3 = model(lr_1_8, sec, sec_2, sec_3, scale)
                recovered = revealNet(restored_hr, scale)
                recovered_2 = revealNet_2(restored_hr2, scale)
                recovered_3 = revealNet_3(restored_hr3, scale)

                restored_hr = torch.clamp(restored_hr, 0, 1)
                restored_hr2 = torch.clamp(restored_hr2, 0, 1)
                restored_hr3 = torch.clamp(restored_hr3, 0, 1)
                recovered = torch.clamp(recovered, 0, 1)
                recovered_2 = torch.clamp(recovered_2, 0, 1)
                recovered_3 = torch.clamp(recovered_3, 0, 1)

                ########################## CALCULATE METRIC
                restored_hr = util.tensor2img(restored_hr)
                restored_hr2 = util.tensor2img(restored_hr2)
                restored_hr3 = util.tensor2img(restored_hr3)
                recovered = util.tensor2img(recovered)
                recovered_2 = util.tensor2img(recovered_2)
                recovered_3 = util.tensor2img(recovered_3)
                sec = util.tensor2img(sec)
                sec_2 = util.tensor2img(sec_2)
                sec_3 = util.tensor2img(sec_3)
                hr = util.tensor2img(hr)
                lr_1_8 = util.tensor2img(lr_1_8)
                lr_1_4 = util.tensor2img(lr_1_4)
                lr_1_2 = util.tensor2img(lr_1_2)

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

                psnr_meter_sr_1_4.update(util.calculate_psnr(restored_hr * 255, lr_1_4 * 255))
                psnr_meter_sr_1_2.update(util.calculate_psnr(restored_hr2 * 255, lr_1_2 * 255))
                psnr_meter_sr.update(util.calculate_psnr(restored_hr3 * 255, hr * 255))
                ssim_meter_sr_1_4.update(util.calculate_ssim(restored_hr * 255, lr_1_4 * 255))
                ssim_meter_sr_1_2.update(util.calculate_ssim(restored_hr2 * 255, lr_1_2 * 255))
                ssim_meter_sr.update(util.calculate_ssim(restored_hr3 * 255, hr * 255))

                psnr_meter_rev_1_8.update(util.calculate_psnr(recovered * 255, sec * 255))
                psnr_meter_rev_1_4.update(util.calculate_psnr(recovered_2 * 255, sec_2 * 255))
                psnr_meter_rev_1_2.update(util.calculate_psnr(recovered_3 * 255, sec_3 * 255))
                ssim_meter_rev_1_8.update(util.calculate_ssim(recovered * 255, sec * 255))
                ssim_meter_rev_1_4.update(util.calculate_ssim(recovered_2 * 255, sec_2 * 255))
                ssim_meter_rev_1_2.update(util.calculate_ssim(recovered_3 * 255, sec_3 * 255))

                if benchmark_lpips:
                    lpips_a = loss_fn_alex(
                        torch.from_numpy(restored_hr * 2 - 1.0).cuda()[None].permute(0, 3, 1, 2).float(),
                        torch.from_numpy(hr * 2 - 1.0).cuda()[None].permute(0, 3, 1, 2).float())
                    lpips_a = lpips_a.cpu().detach().item()
                else:
                    lpips_a = 0

                if save:
                    # f.write(test_data_loader.dataset.imgs[i].split('/')[-1].split('.')[0]+' '+'{:.2f}'.format(psnr_y_hr_)+' '+ '{:.4f}'.format(ssim_y_hr_) + ' ' +'{:.4f}'.format(lpips_a) +'_'+'{:.2f}'.format(psnr_y_lr_)+' '+ '{:.4f}'.format(ssim_y_lr_) +'\n')
                    for x, last_fix in zip([hr, sec, sec_2, sec_3, lr_1_8, lr_1_4, lr_1_2,
                                            restored_hr, restored_hr2, restored_hr2,
                                            recovered, recovered_2, recovered_3],
                                           ["_hr.png", "_sec.png", "_sec_2.png", "_sec_3.png",
                                            "_lr_1_8.png", "_lr_1_4.png", "_lr_1_2.png",
                                            "_restored_hr.png", "_restored_hr2.png", "_restored_hr3.png",
                                            "_recovered.png", "_recovered_2.png", "_recovered_3.png"]):
                        # print(111)
                        mmcv.imwrite(mmcv.rgb2bgr(np.uint8(x * 255)),
                                     join(cfg.save_folder, cfg.data_name + '_x%.1f' % scale,
                                          test_data_loader.dataset.imgs[i].split('/')[-1].split('.')[0] + last_fix))
    logger.info('========> SR: \n'
                'PSNR_SR_1_4: {psnr_meter_sr_1_4.avg:.2f}\n'
                'SSIM_SR_1_4: {ssim_meter_sr_1_4.avg:.4f}\n'
                'PSNR_SR_1_2: {psnr_meter_sr_1_2.avg:.2f}\n'
                'SSIM_SR_1_2: {ssim_meter_sr_1_2.avg:.4f}\n'
                'PSNR_SR: {psnr_meter_sr.avg:.2f}\n'
                'SSIM_SR: {ssim_meter_sr.avg:.4f}\n'
                '========> Recover: \n'
                'PSNR_Rev_1_8: {psnr_meter_rev_1_8.avg:.2f}\n'
                'SSIM_Rev_1_8: {ssim_meter_rev_1_8.avg:.4f}\n'
                'PSNR_Rev_1_4: {psnr_meter_rev_1_4.avg:.2f}\n'
                'SSIM_Rev_1_4: {ssim_meter_rev_1_4.avg:.4f}\n'
                'PSNR_Rev_1_2: {psnr_meter_rev_1_2.avg:.2f}\n'
                'SSIM_Rev_1_2: {ssim_meter_rev_1_2.avg:.4f}\n'
                .format(psnr_meter_sr_1_4=psnr_meter_sr_1_4, ssim_meter_sr_1_4=ssim_meter_sr_1_4,
                        psnr_meter_sr_1_2=psnr_meter_sr_1_2, ssim_meter_sr_1_2=ssim_meter_sr_1_2,
                        psnr_meter_sr=psnr_meter_sr, ssim_meter_sr=ssim_meter_sr,
                        psnr_meter_rev_1_8=psnr_meter_rev_1_8, ssim_meter_rev_1_8=ssim_meter_rev_1_8,
                        psnr_meter_rev_1_4=psnr_meter_rev_1_4, ssim_meter_rev_1_4=ssim_meter_rev_1_4,
                        psnr_meter_rev_1_2=psnr_meter_rev_1_2, ssim_meter_rev_1_2=ssim_meter_rev_1_2,
                        ))


if __name__ == '__main__':
    main()
