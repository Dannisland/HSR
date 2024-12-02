#!/usr/bin/env python
import itertools
import os
import sys
import time
import random

import kornia
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
import cv2

# todo 注意U4/U2的路径位置
sys.path.append("/opt/data/xiaobin/Project/AIDN")
from base.baseTrainer import poly_learning_rate, reduce_tensor, save_checkpoint_loop1, load_state_dict, \
    save_checkpoint_loop2
from base.utilities import get_parser, get_logger, main_process, AverageMeter

################## 注意此处的 revealnet选取 ###################
# todo revealnet选取
from models.RevealNet_tradition_downsampling import RevealNet
from models import get_model_inv
from metrics.loss import *
from metrics import psnr, ssim
from dataset.torch_bicubic import imresize
from torch.optim.lr_scheduler import StepLR
from random import choices
import models.modules.Unet_common as common
from models.hinet import *

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

population = [i / 10.0 for i in range(11, 41)]
weights = [i ** 2 for i in population]

weights_np = np.array(weights)
weights_np_sum = np.sum(weights_np)
weights = [i / weights_np_sum for i in weights]

dwt = common.DWT()
iwt = common.IWT()


def main():
    args = get_parser()
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    cudnn.benchmark = True

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        # cudnn.benchmark = False
        # cudnn.deterministic = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.train_gpu = args.train_gpu[0]
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def worker_init_fn(worker_id):
    manual_seed = 131
    random.seed(manual_seed + worker_id)
    np.random.seed(manual_seed + worker_id)
    torch.manual_seed(manual_seed + worker_id)
    torch.cuda.manual_seed(manual_seed + worker_id)
    torch.cuda.manual_seed_all(manual_seed + worker_id)


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def main_worker(gpu, ngpus_per_node, args):
    cfg = args
    cfg.gpu = gpu
    best_metric = 1e10
    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url, world_size=cfg.world_size,
                                rank=cfg.rank)
    # ####################### Model ####################### #
    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(cfg.save_path)
    inv_model = Model()
    inv_model2 = Model()
    init_model(inv_model)
    init_model(inv_model2)
    inv_crm = get_model_inv(cfg, logger)
    inv_crm2 = get_model_inv(cfg, logger)

    # revealNet = RevealNet(input_nc=3, output_nc=3, cfg=cfg)
    # revealNet_2 = RevealNet(input_nc=3, output_nc=3, cfg=cfg)

    if cfg.sync_bn:
        logger.info("using DDP synced BN")
        inv_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(inv_model)
        inv_crm = torch.nn.SyncBatchNorm.convert_sync_batchnorm(inv_crm)
        inv_model2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(inv_model2)
        inv_crm2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(inv_crm2)
        # revealNet_2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(revealNet_2)
    if main_process(cfg):
        logger.info(cfg)
        logger.info("=> creating model ...")
        inv_crm.summary(logger, writer)
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
        cfg.batch_size_val = int(cfg.batch_size_val / ngpus_per_node)
        cfg.workers = int(cfg.workers / ngpus_per_node)
        inv_model = torch.nn.parallel.DistributedDataParallel(inv_model.cuda(gpu), device_ids=[gpu])
        inv_crm = torch.nn.parallel.DistributedDataParallel(inv_crm.cuda(gpu), device_ids=[gpu])
        inv_model2 = torch.nn.parallel.DistributedDataParallel(inv_model2.cuda(gpu), device_ids=[gpu])
        inv_crm2 = torch.nn.parallel.DistributedDataParallel(inv_crm2.cuda(gpu), device_ids=[gpu])
        # revealNet_2 = torch.nn.parallel.DistributedDataParallel(revealNet_2.cuda(gpu), device_ids=[gpu])
    else:
        torch.cuda.set_device(gpu)
        inv_model = inv_model.cuda()
        inv_crm = inv_crm.cuda()
        inv_model2 = inv_model2.cuda()
        inv_crm2 = inv_crm2.cuda()
        # revealNet_2 = revealNet_2.cuda()
        # model = torch.nn.DataParallel(model.cuda(), device_ids=gpu)

    # inv_model_list = [inv_model]
    # inv_crm_list = [inv_crm]

    # ####################### Loss ####################### #
    loss_fn_lr = nn.MSELoss()
    loss_fn_hr = nn.L1Loss()
    loss = [loss_fn_lr, loss_fn_hr]

    # ####################### Optimizer ####################### #
    if cfg.use_sgd:
        optimizer = torch.optim.SGD(
            itertools.chain(inv_model.parameters(), inv_crm.parameters(), inv_model2.parameters(),
                            inv_crm2.parameters()), lr=cfg.base_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(
            itertools.chain(inv_model.parameters(), inv_crm.parameters(), inv_model2.parameters(),
                            inv_crm2.parameters()),
            lr=2e-05, eps=cfg.eps, weight_decay=cfg.weight_decay, betas=(0.5, 0.999))

    if cfg.weight:
        if os.path.isfile(cfg.weight):
            if main_process(cfg):
                logger.info("=> loading weight '{}'".format(cfg.weight))
            checkpoint = torch.load(cfg.weight, map_location=torch.device('cpu'))

            load_state_dict(inv_model, checkpoint['state_dict'], strict=False)
            load_state_dict(inv_crm, checkpoint['reveal'], strict=False)
            load_state_dict(inv_model2, checkpoint['state_dict2'], strict=False)
            load_state_dict(inv_crm2, checkpoint['reveal2'], strict=False)
            # load_state_dict(revealNet_2, checkpoint['reveal_2'], strict=False)

            if main_process(cfg):
                logger.info("=> loaded weight '{}'".format(cfg.weight))
        else:
            if main_process(cfg):
                logger.info("=> no weight found at '{}'".format(cfg.weight))
    if cfg.StepLR:
        scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    else:
        scheduler = None
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            if main_process(cfg):
                logger.info("=> loading checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
            load_state_dict(inv_model, checkpoint['state_dict'])
            load_state_dict(inv_crm, checkpoint['reveal'])
            load_state_dict(inv_model2, checkpoint['state_dict2'])
            load_state_dict(inv_crm2, checkpoint['reveal2'])

            # load_state_dict(revealNet_2, checkpoint['reveal_2'])
            cfg.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_metric = checkpoint['best_metric']
            if cfg.StepLR:
                scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma, last_epoch=cfg.start_epoch - 1)
                scheduler.load_state_dict(checkpoint['scheduler'])

            if main_process(cfg):
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(cfg.resume, checkpoint['epoch']))
        else:
            if main_process(cfg):
                logger.info("=> no checkpoint found at '{}'".format(cfg.resume))

    # ####################### Data Loader ####################### #
    if cfg.data_name == 'DIV2K':
        from dataset.div2k import DIV2K
        train_data = DIV2K(data_list=os.path.join(cfg.data_root, 'list/train.txt'), training=True,
                           cfg=cfg)
        val_data = DIV2K(data_list=os.path.join(cfg.data_root, 'list/val.txt'), training=False,
                         cfg=cfg) if cfg.evaluate else None

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if cfg.distributed else None
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=cfg.workers, pin_memory=True,
                                                   sampler=train_sampler,
                                                   worker_init_fn=worker_init_fn)
        if cfg.evaluate:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data) if cfg.distributed else None
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.batch_size_val,
                                                     shuffle=False, num_workers=cfg.workers, pin_memory=True,
                                                     drop_last=False,
                                                     worker_init_fn=worker_init_fn, sampler=val_sampler)
    else:
        raise Exception('Dataset not supported yet'.format(cfg.data_name))

    # ####################### Train ####################### #
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
            if cfg.evaluate:
                val_sampler.set_epoch(epoch)

        loss_train, hr_loss, _ = train(train_loader, inv_model, inv_crm, inv_model2, inv_crm2, loss, optimizer, epoch,
                                       cfg)
        epoch_log = epoch + 1
        # # Adaptive LR
        if cfg.StepLR:
            scheduler.step()
        if main_process(cfg):
            logger.info('TRAIN Epoch: {} '
                        'loss_train: {} '
                        'loss_hr: {} '
                        .format(epoch_log, loss_train, hr_loss)
                        )
            for m, s in zip([loss_train, hr_loss],
                            ["train/loss", "train/loss_hr"]):
                writer.add_scalar(s, m, epoch_log)

        is_best = False
        if cfg.evaluate and (epoch_log % cfg.eval_freq == 0):
            loss_val, hr_loss, _, PSNR, SSIM = \
                validate(val_loader, inv_model, inv_crm, inv_model2, inv_crm2, loss, epoch, cfg)
            if main_process(cfg):
                logger.info('VAL Epoch: {} '
                            'loss_val: {:.6} '
                            'loss_hr: {:.6} '
                            'PSNR: {:.2},{:.2}'
                            'SSIM: {:.4},{:.4}'
                            .format(epoch_log, loss_val, hr_loss, *PSNR, *SSIM)
                            )
                for m, s in zip([loss_val, hr_loss, *PSNR, *SSIM],
                                ["val/loss", "val/loss_hr",
                                 "val/PSNR_lr", "val/PSNR_hr",
                                 "val/SSIM_lr", "val/SSIM_hr"]):
                    writer.add_scalar(s, m, epoch_log)

            # remember best iou and save checkpoint
            is_best = hr_loss < best_metric
            best_metric = min(best_metric, hr_loss)
        if (epoch_log % cfg.save_freq == 0) and main_process(cfg):
            save_checkpoint_loop2(inv_model,
                                  inv_crm,
                                  inv_model2,
                                  inv_crm2,
                                  # revealNet_2,
                                  other_state={
                                      'epoch': epoch_log,
                                      'state_dict': inv_model.state_dict(),
                                      'reveal': inv_crm.state_dict(),
                                      'state_dict2': inv_model2.state_dict(),
                                      'reveal2': inv_crm2.state_dict(),
                                      # 'reveal_2': revealNet_2.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      'best_metric': best_metric},
                                  sav_path=os.path.join(cfg.save_path, 'model'),
                                  is_best=is_best
                                  )


def train(train_loader, inv_model, inv_crm, inv_model2, inv_crm2, loss_fn, optimizer, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_hr_meter = AverageMeter()
    loss_hr_meter2 = AverageMeter()
    loss_sec_meter = AverageMeter()
    loss_sec_meter2 = AverageMeter()

    inv_model.train()
    inv_crm.train()
    inv_model2.train()
    inv_crm2.train()
    # revealNet_2.train()

    end = time.time()
    max_iter = cfg.epochs * len(train_loader)
    for i, batch in enumerate(train_loader):
        # pdb.set_trace()
        if cfg.fixed_scale:  # if training with fixed_scale
            scale = cfg.scale
        else:
            scale = 1.5

        current_iter = epoch * len(train_loader) + i + 1
        data_time.update(time.time() - end)
        hr, sec = batch['img_gt'], batch['img_sec']
        sec_2 = batch['img_sec_2']

        hr = hr.cuda(cfg.gpu, non_blocking=True)
        sec = sec.cuda(cfg.gpu, non_blocking=True)  # size = hr/scale
        sec_2 = sec_2.cuda(cfg.gpu, non_blocking=True)  # size = hr / (scale*2)

        #################
        #    scale:     #
        #################
        lr = imresize(hr, scale=1.0 / (scale * scale)).detach()
        lr_2 = imresize(hr, scale=1.0 / scale).detach()

        sec = imresize(sec, scale=1.0 / (scale * scale)).detach()
        sec_2 = imresize(sec_2, scale=1.0 / scale).detach()

        #################
        #   forward-1:  #
        #################
        cover_input = dwt(lr)
        secret_input = dwt(sec)
        input_img = torch.cat((cover_input, secret_input), 1).cuda(cfg.gpu, non_blocking=True)
        output = inv_model(input_img)
        output_steg = output.narrow(1, 0, 12)
        output_z = output.narrow(1, 12, output.shape[1] - 12)

        output_img = iwt(output_steg)
        output_stego = inv_crm(output_img, scale)

        #################
        #   forward-2:  #
        #################
        cover_input_2 = dwt(output_stego)
        secret_input_2 = dwt(sec_2)
        input_img_2 = torch.cat((cover_input_2, secret_input_2), 1).cuda(cfg.gpu, non_blocking=True)
        output_2 = inv_model2(input_img_2)
        output_steg_2 = output_2.narrow(1, 0, 12)
        output_z_2 = output_2.narrow(1, 12, output_2.shape[1] - 12)

        output_img_2 = iwt(output_steg_2)
        output_stego_2 = inv_crm2(output_img_2, scale)

        #################
        #   backward-1: #
        #################
        output_crm = inv_crm(output_stego, scale, rev=True)
        output_crm_rev = dwt(output_crm)
        # output_crm_rev = dwt(output_img)

        output_z_guass = gauss_noise(output_z.shape)
        output_rev = torch.cat((output_crm_rev, output_z_guass), 1)
        output_image = inv_model(output_rev, rev=True)

        recovered_fre = output_image.narrow(1, 12, output_image.shape[1] - 12)
        recovered = iwt(recovered_fre)

        #################
        #   backward-2: #
        #################
        output_crm_2 = inv_crm2(output_stego_2, scale, rev=True)
        output_crm_rev_2 = dwt(output_crm_2)
        # output_crm_rev_2 = dwt(output_img_2)

        output_z_guass_2 = gauss_noise(output_z_2.shape)
        output_rev_2 = torch.cat((output_crm_rev_2, output_z_guass_2), 1)
        output_image_2 = inv_model2(output_rev_2, rev=True)

        recovered_fre_2 = output_image_2.narrow(1, 12, output_image_2.shape[1] - 12)
        recovered_2 = iwt(recovered_fre_2)

        ######################
        #   backward2-dist:  #
        ######################
        _, _, w, h = output_stego.shape
        dist = nn.functional.interpolate(output_stego_2, [w, h], mode="bicubic")

        output_crm_dist = inv_crm(dist, scale, rev=True)
        output_crm_rev_dist = dwt(output_crm_dist)
        # output_crm_rev = dwt(output_img)

        output_z_guass_dist = gauss_noise(output_z.shape)
        output_rev_dist = torch.cat((output_crm_rev_dist, output_z_guass_dist), 1)
        output_image_dist = inv_model(output_rev_dist, rev=True)

        recovered_fre_dist = output_image_dist.narrow(1, 12, output_image_dist.shape[1] - 12)
        recovered_dist = iwt(recovered_fre_dist)

        #################
        #     Loss:     #
        #################
        # loss_lr = loss_fn[0](encoded_lr, lr)  # 0: MSE 1:L1

        loss_hr = loss_fn[0](lr_2, output_stego)  # MSE cover -- stego
        loss_sec = loss_fn[0](sec, recovered) * 2  # MSE secret -- recover
        loss_hr_2 = loss_fn[0](hr, output_stego_2)  # MSE cover2 -- stego2
        loss_sec_2 = loss_fn[0](sec_2, recovered_2) * 2  # MSE secret2 -- recover2

        loss_hr_dist = loss_fn[0](lr_2, dist)  # MSE cover -- dist
        loss_sec_dist = loss_fn[0](sec, recovered_dist)  # MSE secret -- sec_dist

        steg_low = output_steg.narrow(1, 0, 3)  # 低频 stego
        cover_low = cover_input.narrow(1, 0, 3)  # 低频 cover
        recover_low = recovered_fre.narrow(1, 0, 3)  # 低频 recover
        secret_low = secret_input.narrow(1, 0, 3)  # 低频 secret

        recover_low_dist = recovered_fre_dist.narrow(1, 0, 3)  # 低频 dist_recover

        steg_low_2 = output_steg_2.narrow(1, 0, 3)  # 低频 stego2
        cover_low_2 = cover_input_2.narrow(1, 0, 3)  # 低频 cover
        recover_low_2 = recovered_fre_2.narrow(1, 0, 3)  # 低频 recover2
        secret_low_2 = secret_input_2.narrow(1, 0, 3)  # 低频 secret2

        loss_fre_cover = loss_fn[0](steg_low, cover_low)  # 低频Loss cover -- stego
        loss_fre_secret = loss_fn[0](recover_low, secret_low)  # 低频Loss secret -- recover
        loss_fre_cover_2 = loss_fn[0](steg_low_2, cover_low_2)  # 低频Loss cover2 -- stego2
        loss_fre_secret_2 = loss_fn[0](recover_low_2, secret_low_2)  # 低频Loss secret2 -- recover2

        loss_fre_secret_dist = loss_fn[0](recover_low_dist, secret_low) * 0.5  # 低频Loss secret -- dist_recover

        loss_fre_inv = loss_fn[0](output_steg, output_crm_rev) * 0.5  # 频率系数 Loss inv输出 -- inv输入
        loss_fre_inv_2 = loss_fn[0](output_steg_2, output_crm_rev_2) * 0.5  # 频率系数 Loss inv2输出 -- inv2输入

        loss_fre_inv_dist = loss_fn[0](output_steg, output_crm_rev_dist) * 0.5  # 频率系数 Loss dist_inv2输出 -- inv输入

        loss = loss_hr + loss_sec + loss_fre_cover + loss_fre_secret + loss_fre_inv + \
               loss_hr_2 + loss_sec_2 + loss_fre_cover_2 + loss_fre_secret_2 + loss_fre_inv_2 + \
               loss_hr_dist + loss_sec_dist + loss_fre_secret_dist + loss_fre_inv_dist

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        for m, x in zip([loss_meter, loss_hr_meter, loss_hr_meter2, loss_sec_meter, loss_sec_meter2],
                        [loss, loss_hr, loss_hr_2, loss_sec, loss_sec_2]):
            m.update(x.item(), lr.shape[0])
        # Adjust lr
        if cfg.poly_lr:
            current_lr = poly_learning_rate(cfg.base_lr, current_iter, max_iter, power=cfg.power)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        with (torch.no_grad()):
            batch_enc_psnr = abs(kornia.losses.psnr_loss(lr_2, output_stego, 1))
            batch_dec_psnr = abs(kornia.losses.psnr_loss(sec, recovered, 1))
            batch_enc_ssim = 1 - abs(
                kornia.losses.ssim_loss(lr_2, output_stego, window_size=5, reduction="mean"))
            batch_dec_ssim = 1 - abs(kornia.losses.ssim_loss(recovered, sec, window_size=5, reduction="mean"))

            batch_enc_psnr_2 = abs(kornia.losses.psnr_loss(hr, output_stego_2, 1))
            batch_dec_psnr_2 = abs(kornia.losses.psnr_loss(sec_2, recovered_2, 1))
            batch_enc_ssim_2 = 1 - abs(
                kornia.losses.ssim_loss(hr, output_stego_2, window_size=5, reduction="mean"))
            batch_dec_ssim_2 = 1 - abs(kornia.losses.ssim_loss(recovered_2, sec_2, window_size=5, reduction="mean"))

            batch_dist_psnr = abs(kornia.losses.psnr_loss(dist, output_stego, 1))
            batch_redist_psnr = abs(kornia.losses.psnr_loss(recovered_dist, sec, 1))

            data_result_info = ('\nSR_Step1 == psnr_enc:{}, psnr_dec:{}, ssim_enc:{}, ssim_dec:{} \n'
                                'SR_Step2 == psnr_enc2:{}, psnr_dec2:{}, ssim_enc2:{}, ssim_dec2:{} \n'
                                'SR2 TO Dist1 == psnr_dist:{}, psnr_redist:{} \n'
                                ).format(batch_enc_psnr, batch_dec_psnr, batch_enc_ssim, batch_dec_ssim,
                                         batch_enc_psnr_2, batch_dec_psnr_2, batch_enc_ssim_2, batch_dec_ssim_2,
                                         batch_dist_psnr, batch_redist_psnr)

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % cfg.print_freq == 0 and main_process(cfg):
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain: {remain_time} '
                        'Loss: {loss_meter.val:.4f} '
                        'Loss_hr: {loss_hr_meter.val:.4f} '
                        'Loss_sec: {loss_sec_meter.val:.4f} '
                        'Loss_hr2: {loss_hr_meter2.val:.4f} '
                        'Loss_sec2: {loss_sec_meter2.val:.4f} '
                        'data_info: {data_result_info} '
                        .format(epoch + 1, cfg.epochs, i + 1, len(train_loader),
                                batch_time=batch_time, data_time=data_time,
                                remain_time=remain_time,
                                loss_meter=loss_meter,
                                loss_hr_meter=loss_hr_meter,
                                loss_sec_meter=loss_sec_meter,
                                loss_hr_meter2=loss_hr_meter2,
                                loss_sec_meter2=loss_sec_meter2,
                                data_result_info=data_result_info
                                ))
            for m, s in zip([loss_meter, loss_hr_meter, loss_sec_meter, loss_hr_meter2, loss_sec_meter2],
                            ["train_batch/loss", "train_batch/loss_hr", "train_batch/loss_sec", "train_batch/loss_hr2",
                             "train_batch/loss_sec2"]):
                writer.add_scalar(s, m.val, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)
            writer.add_histogram('train_batch/scale', scale, current_iter)
    return loss_meter.avg, loss_hr_meter.avg, loss_sec_meter.avg


def validate(val_loader, inv_model, inv_crm, inv_model2, inv_crm2, loss_fn, epoch, cfg):
    loss_meter = AverageMeter()
    loss_hr_meter = AverageMeter()
    loss_hr_meter2 = AverageMeter()
    loss_sec_meter = AverageMeter()
    loss_sec_meter2 = AverageMeter()
    psnr_meter, ssim_meter = [AverageMeter() for _ in range(6)], [AverageMeter() for _ in range(4)]

    psnr_calculator, ssim_calculator = psnr.PSNR(), ssim.SSIM()

    inv_model.eval()
    inv_crm.eval()
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            scale = cfg.scale  # 4
            hr, sec = batch['img_gt'], batch['img_sec']
            sec_2 = batch['img_sec_2']

            hr = hr.cuda(cfg.gpu, non_blocking=True)
            sec = sec.cuda(cfg.gpu, non_blocking=True)
            sec_2 = sec_2.cuda(cfg.gpu, non_blocking=True)  # size = hr / (scale*2)

            #################
            #    scale:     #
            #################
            lr = imresize(hr, scale=1.0 / (scale * scale)).detach()
            lr_2 = imresize(hr, scale=1.0 / scale).detach()

            sec = imresize(sec, scale=1.0 / (scale * scale)).detach()
            sec_2 = imresize(sec_2, scale=1.0 / scale).detach()

            #################
            #    forward:   #
            #################

            cover_input = dwt(lr)
            secret_input = dwt(sec)
            input_img = torch.cat((cover_input, secret_input), 1).cuda(cfg.gpu, non_blocking=True)
            output = inv_model(input_img)
            output_steg = output.narrow(1, 0, 12)
            output_z = output.narrow(1, 12, output.shape[1] - 12)

            output_img = iwt(output_steg)
            output_stego = inv_crm(output_img, scale)

            #################
            #   forward-2:  #
            #################
            cover_input_2 = dwt(output_stego)
            secret_input_2 = dwt(sec_2)
            input_img_2 = torch.cat((cover_input_2, secret_input_2), 1).cuda(cfg.gpu, non_blocking=True)
            output_2 = inv_model2(input_img_2)
            output_steg_2 = output_2.narrow(1, 0, 12)
            output_z_2 = output_2.narrow(1, 12, output_2.shape[1] - 12)

            output_img_2 = iwt(output_steg_2)
            output_stego_2 = inv_crm2(output_img_2, scale)

            #################
            #   backward:   #
            #################

            output_crm = inv_crm(output_stego, scale, rev=True)
            output_crm_rev = dwt(output_crm)

            output_z_guass = gauss_noise(output_z.shape)
            output_rev = torch.cat((output_crm_rev, output_z_guass), 1)
            output_image = inv_model(output_rev, rev=True)

            recovered_fre = output_image.narrow(1, 12, output_image.shape[1] - 12)
            recovered = iwt(recovered_fre)

            #################
            #   backward-2: #
            #################
            output_crm_2 = inv_crm2(output_stego_2, scale, rev=True)
            output_crm_rev_2 = dwt(output_crm_2)
            # output_crm_rev_2 = dwt(output_img_2)

            output_z_guass_2 = gauss_noise(output_z_2.shape)
            output_rev_2 = torch.cat((output_crm_rev_2, output_z_guass_2), 1)
            output_image_2 = inv_model2(output_rev_2, rev=True)

            recovered_fre_2 = output_image_2.narrow(1, 12, output_image_2.shape[1] - 12)
            recovered_2 = iwt(recovered_fre_2)

            ######################
            #   backward2-dist:  #
            ######################
            _, _, w, h = output_stego.shape
            dist = nn.functional.interpolate(output_stego_2, [w, h], mode="bicubic")

            output_crm_dist = inv_crm(dist, scale, rev=True)
            output_crm_rev_dist = dwt(output_crm_dist)
            # output_crm_rev = dwt(output_img)

            output_z_guass_dist = gauss_noise(output_z.shape)
            output_rev_dist = torch.cat((output_crm_rev_dist, output_z_guass_dist), 1)
            output_image_dist = inv_model(output_rev_dist, rev=True)

            recovered_fre_dist = output_image_dist.narrow(1, 12, output_image_dist.shape[1] - 12)
            recovered_dist = iwt(recovered_fre_dist)

            #################
            #     Loss:     #
            #################
            # loss_lr = loss_fn[0](encoded_lr, lr)  # 0: MSE 1:L1

            loss_hr = loss_fn[0](lr_2, output_stego)  # MSE cover -- stego
            loss_sec = loss_fn[0](sec, recovered) * 2  # MSE secret -- recover
            loss_hr_2 = loss_fn[0](hr, output_stego_2)  # MSE cover2 -- stego2
            loss_sec_2 = loss_fn[0](sec_2, recovered_2) * 2  # MSE secret2 -- recover2

            loss_hr_dist = loss_fn[1](lr_2, dist) * 0.5  # MSE cover -- dist
            loss_sec_dist = loss_fn[1](sec, recovered_dist)  # MSE secret -- sec_dist

            steg_low = output_steg.narrow(1, 0, 3)  # 低频 stego
            cover_low = cover_input.narrow(1, 0, 3)  # 低频 cover
            recover_low = recovered_fre.narrow(1, 0, 3)  # 低频 recover
            secret_low = secret_input.narrow(1, 0, 3)  # 低频 secret

            recover_low_dist = recovered_fre_dist.narrow(1, 0, 3)  # 低频 dist_recover

            steg_low_2 = output_steg_2.narrow(1, 0, 3)  # 低频 stego2
            cover_low_2 = cover_input_2.narrow(1, 0, 3)  # 低频 cover
            recover_low_2 = recovered_fre_2.narrow(1, 0, 3)  # 低频 recover2
            secret_low_2 = secret_input_2.narrow(1, 0, 3)  # 低频 secret2

            loss_fre_cover = loss_fn[0](steg_low, cover_low)  # 低频Loss cover -- stego
            loss_fre_secret = loss_fn[0](recover_low, secret_low)  # 低频Loss secret -- recover
            loss_fre_cover_2 = loss_fn[0](steg_low_2, cover_low_2)  # 低频Loss cover2 -- stego2
            loss_fre_secret_2 = loss_fn[0](recover_low_2, secret_low_2)  # 低频Loss secret2 -- recover2

            loss_fre_secret_dist = loss_fn[0](recover_low_dist, secret_low)  # 低频Loss secret -- dist_recover

            loss_fre_inv = loss_fn[0](output_steg, output_crm_rev)  # 频率系数 Loss inv输出 -- inv输入
            loss_fre_inv_2 = loss_fn[0](output_steg_2, output_crm_rev_2)  # 频率系数 Loss inv2输出 -- inv2输入

            loss_fre_inv_dist = loss_fn[0](output_steg, output_crm_rev_dist)  # 频率系数 Loss dist_inv2输出 -- inv输入

            loss = loss_hr + loss_sec + loss_fre_cover + loss_fre_secret + loss_fre_inv + \
                   loss_hr_2 + loss_sec_2 + loss_fre_cover_2 + loss_fre_secret_2 + loss_fre_inv_2 + \
                   loss_hr_dist + loss_sec_dist + loss_fre_secret_dist + loss_fre_inv_dist

            psnr_lr, psnr_hr = \
                psnr_calculator(lr_2, output_stego), psnr_calculator(sec, recovered)
            ssim_lr, ssim_hr = \
                ssim_calculator(lr_2, output_stego), ssim_calculator(sec, recovered)

            psnr_lr_2, psnr_hr_2 = \
                psnr_calculator(recovered_2, sec_2), psnr_calculator(output_stego_2, hr)
            ssim_lr_2, ssim_hr_2 = \
                ssim_calculator(recovered_2, sec_2), ssim_calculator(output_stego_2, hr)

            psnr_dist, psnr_redist = \
                psnr_calculator(dist, output_stego), psnr_calculator(recovered_dist, sec)

            if cfg.distributed:
                loss = reduce_tensor(loss, cfg)
                loss_hr = reduce_tensor(loss_hr, cfg)
                loss_sec = reduce_tensor(loss_sec, cfg)

                psnr_lr = reduce_tensor(psnr_lr, cfg)
                psnr_hr = reduce_tensor(psnr_hr, cfg)
                ssim_lr = reduce_tensor(ssim_lr, cfg)
                ssim_hr = reduce_tensor(ssim_hr, cfg)

                psnr_lr_2 = reduce_tensor(psnr_lr_2, cfg)
                psnr_hr_2 = reduce_tensor(psnr_hr_2, cfg)
                ssim_lr_2 = reduce_tensor(ssim_lr_2, cfg)
                ssim_hr_2 = reduce_tensor(ssim_hr_2, cfg)

                psnr_dist = reduce_tensor(psnr_dist, cfg)
                psnr_redist = reduce_tensor(psnr_redist, cfg)

            for m, x in zip([loss_meter, loss_hr_meter, loss_sec_meter, *psnr_meter, *ssim_meter],
                            [loss, loss_hr, loss_sec, psnr_lr, psnr_hr, psnr_lr_2, psnr_hr_2, psnr_dist, psnr_redist,
                             ssim_lr, ssim_hr, ssim_lr_2, ssim_hr_2]):
                m.update(x.item(), hr.shape[0])

            # Visualize after validation
        if main_process(cfg):
            sample_lr = torchvision.utils.make_grid(recovered.clamp(0.0, 1.0))
            sample_hr = torchvision.utils.make_grid(hr.clamp(0.0, 1.0))
            writer.add_image('sample_results/res_lr', sample_lr, epoch + 1)
            writer.add_image('sample_results/res_hr', sample_hr, epoch + 1)

    return loss_meter.avg, loss_hr_meter.avg, loss_sec_meter.avg, [m.avg for m in psnr_meter], [m.avg for m in
                                                                                                ssim_meter]


if __name__ == '__main__':
    main()
