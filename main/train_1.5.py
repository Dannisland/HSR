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
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
import cv2

sys.path.append("/opt/data/xiaobin/AIDN")
from base.baseTrainer import poly_learning_rate, reduce_tensor, save_checkpoint, load_state_dict
from base.utilities import get_parser, get_logger, main_process, AverageMeter
from models.RevealNet import RevealNet
from models import get_model
from metrics.loss import *
from metrics import psnr, ssim
from dataset.torch_bicubic import imresize
from torch.optim.lr_scheduler import StepLR
from random import choices

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

population = [i / 10.0 for i in range(11, 41)]
weights = [i ** 2 for i in population]

weights_np = np.array(weights)
weights_np_sum = np.sum(weights_np)
weights = [i / weights_np_sum for i in weights]


def main():
    args = get_parser()
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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
    model = get_model(cfg, logger)
    revealNet = RevealNet(input_nc=3, output_nc=3, cfg=cfg)
    revealNet_2 = RevealNet(input_nc=3, output_nc=3, cfg=cfg)

    if cfg.sync_bn:
        logger.info("using DDP synced BN")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        revealNet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(revealNet)
        revealNet_2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(revealNet_2)
    if main_process(cfg):
        logger.info(cfg)
        logger.info("=> creating model ...")
        model.summary(logger, writer)
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
        cfg.batch_size_val = int(cfg.batch_size_val / ngpus_per_node)
        cfg.workers = int(cfg.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(gpu), device_ids=[gpu])
        revealNet = torch.nn.parallel.DistributedDataParallel(revealNet.cuda(gpu), device_ids=[gpu])
        revealNet_2 = torch.nn.parallel.DistributedDataParallel(revealNet_2.cuda(gpu), device_ids=[gpu])
    else:
        torch.cuda.set_device(gpu)
        model = model.cuda()
        revealNet = revealNet.cuda()
        revealNet_2 = revealNet_2.cuda()
        # model = torch.nn.DataParallel(model.cuda(), device_ids=gpu)
    # ####################### Loss ####################### #
    loss_fn_lr = nn.MSELoss()
    loss_fn_hr = nn.L1Loss()
    loss = [loss_fn_lr, loss_fn_hr]

    # ####################### Optimizer ####################### #
    if cfg.use_sgd:
        optimizer = torch.optim.SGD(
            itertools.chain(model.parameters(), revealNet.parameters(), revealNet_2.parameters()), lr=cfg.base_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(
            itertools.chain(model.parameters(), revealNet.parameters(), revealNet_2.parameters()), lr=cfg.base_lr)

    if cfg.weight:
        if os.path.isfile(cfg.weight):
            if main_process(cfg):
                logger.info("=> loading weight '{}'".format(cfg.weight))
            checkpoint = torch.load(cfg.weight, map_location=torch.device('cpu'))

            load_state_dict(model, checkpoint['state_dict'], strict=False)
            load_state_dict(revealNet, checkpoint['reveal'], strict=False)
            load_state_dict(revealNet_2, checkpoint['reveal_2'], strict=False)

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
            load_state_dict(model, checkpoint['state_dict'])
            load_state_dict(revealNet, checkpoint['reveal'])
            load_state_dict(revealNet_2, checkpoint['reveal_2'])
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

        loss_train, hr_loss, _ = train(train_loader, model, revealNet, revealNet_2, loss, optimizer, epoch, cfg)
        epoch_log = epoch + 1
        # Adaptive LR
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
                validate(val_loader, model, revealNet, revealNet_2, loss, epoch, cfg)
            if main_process(cfg):
                logger.info('VAL Epoch: {} '
                            'loss_val: {:.6} '
                            'loss_hr: {:.6} '
                            'PSNR: {:.2},{:.2},{:.2},{:.2} '
                            'SSIM: {:.4},{:.4},{:.4},{:.4}'
                            .format(epoch_log, loss_val, hr_loss, *PSNR, *SSIM)
                            )
                for m, s in zip([loss_val, hr_loss, *PSNR, *SSIM],
                                ["val/loss", "val/loss_hr", "val/PSNR_lr", "val/PSNR_hr", "val/SSIM_lr",
                                 "val/SSIM_hr", "val/PSNR_lr_2", "val/PSNR_hr_2", "val/SSIM_lr_2",
                                 "val/SSIM_hr_2"]):
                    writer.add_scalar(s, m, epoch_log)

            # remember best iou and save checkpoint
            is_best = hr_loss < best_metric
            best_metric = min(best_metric, hr_loss)
        if (epoch_log % cfg.save_freq == 0) and main_process(cfg):
            save_checkpoint(model,
                            revealNet,
                            revealNet_2,
                            other_state={
                                'epoch': epoch_log,
                                'state_dict': model.state_dict(),
                                'reveal': revealNet.state_dict(),
                                'reveal_2': revealNet_2.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'best_metric': best_metric},
                            sav_path=os.path.join(cfg.save_path, 'model'),
                            is_best=is_best
                            )


def train(train_loader, model, revealNet, revealNet_2, loss_fn, optimizer, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_hr_meter = AverageMeter()
    loss_hr_meter2 = AverageMeter()
    loss_sec_meter = AverageMeter()
    loss_sec_meter2 = AverageMeter()
    model.train()
    revealNet.train()
    revealNet_2.train()

    end = time.time()
    max_iter = cfg.epochs * len(train_loader)
    for i, batch in enumerate(train_loader):
        # pdb.set_trace()
        if cfg.fixed_scale:  # if training with fixed_scale
            scale = cfg.scale
        else:
            if epoch == 0:
                scale = 1.5 #random.randint(2, cfg.scale)
            else:
                scale = 1.5
                # if cfg.balanceS:
                #     scale = choices(population, weights)[0]
                # else:
                #     scale = random.randint(11, cfg.scale * 10) / 10.0

        current_iter = epoch * len(train_loader) + i + 1
        data_time.update(time.time() - end)
        hr, sec = batch['img_gt'], batch['img_sec']
        sec_2 = batch['img_sec_2']

        hr = hr.cuda(cfg.gpu, non_blocking=True)
        sec = sec.cuda(cfg.gpu, non_blocking=True)  # size = hr/scale
        sec_2 = sec_2.cuda(cfg.gpu, non_blocking=True)  # size = hr / (scale*2)

        lr_1_4 = imresize(hr, scale=1.0 / (scale * scale)).detach()
        lr_1_2 = imresize(hr, scale=1.0 / scale).detach()

        sec = imresize(sec, scale=1.0 / (scale * scale)).detach()
        sec_2 = imresize(sec_2, scale=1.0 / scale).detach()

        restored_hr, restored_hr2 = model(lr_1_4, sec, sec_2, scale)
        recovered = revealNet(restored_hr, scale)
        recovered_2 = revealNet_2(restored_hr2, scale)

        # LOSS
        # loss_lr = loss_fn[0](encoded_lr, lr)  # 0: MSE 1:L1
        loss_hr = loss_fn[1](restored_hr, lr_1_2)
        loss_hr_2 = loss_fn[1](restored_hr2, hr)
        loss_sec = loss_fn[1](sec, recovered)
        loss_sec_2 = loss_fn[1](sec_2, recovered_2)

        loss = loss_hr + loss_sec + loss_hr_2 + loss_sec_2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        for m, x in zip([loss_meter, loss_hr_meter, loss_hr_meter2, loss_sec_meter, loss_sec_meter2],
                        [loss, loss_hr, loss_hr_2, loss_sec, loss_sec_2]):
            m.update(x.item(), lr_1_4.shape[0])
        # Adjust lr
        if cfg.poly_lr:
            current_lr = poly_learning_rate(cfg.base_lr, current_iter, max_iter, power=cfg.power)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        with torch.no_grad():
            batch_enc_psnr = abs(kornia.losses.psnr_loss(restored_hr, lr_1_2, 1))
            batch_dec_psnr = abs(kornia.losses.psnr_loss(recovered, sec, 1))
            batch_enc_ssim = abs(kornia.losses.ssim_loss(restored_hr.detach(), lr_1_2, window_size=5, reduction="mean"))
            batch_dec_ssim = abs(kornia.losses.ssim_loss(recovered.detach(), sec, window_size=5, reduction="mean"))

            batch_enc_psnr_2 = abs(kornia.losses.psnr_loss(restored_hr2, hr, 1))
            batch_dec_psnr_2 = abs(kornia.losses.psnr_loss(recovered_2, sec_2, 1))
            batch_enc_ssim_2 = abs(kornia.losses.ssim_loss(restored_hr2.detach(), hr, window_size=5, reduction="mean"))
            batch_dec_ssim_2 = abs(
                kornia.losses.ssim_loss(recovered_2.detach(), sec_2, window_size=5, reduction="mean"))
            data_result_info = ('1/4 SR == psnr_enc:{}, psnr_dec:{}, ssim_enc:{}, ssim_dec:{} '
                                '1/2 SR == psnr_enc2:{}, psnr_dec2:{}, ssim_enc2:{}, ssim_dec2:{}'
                                ).format(batch_enc_psnr, batch_dec_psnr, batch_enc_ssim, batch_dec_ssim,
                                         batch_enc_psnr_2, batch_dec_psnr_2, batch_enc_ssim_2, batch_dec_ssim_2)

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


def validate(val_loader, model, revealNet, revealNet_2, loss_fn, epoch, cfg):
    loss_meter = AverageMeter()
    loss_hr_meter = AverageMeter()
    loss_sec_meter = AverageMeter()
    psnr_meter, ssim_meter = [AverageMeter() for _ in range(4)], [AverageMeter() for _ in range(4)]

    psnr_calculator, ssim_calculator = psnr.PSNR(), ssim.SSIM()

    model.eval()
    revealNet.eval()
    revealNet_2.eval()
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            scale = cfg.scale  # 4
            hr, sec = batch['img_gt'], batch['img_sec']
            sec_2 = batch['img_sec_2']

            hr = hr.cuda(cfg.gpu, non_blocking=True)
            sec = sec.cuda(cfg.gpu, non_blocking=True)
            sec_2 = sec_2.cuda(cfg.gpu, non_blocking=True)  # size = hr / (scale*2)

            lr_1_4 = imresize(hr, scale=1.0 / (scale*2)).detach()
            lr_1_2 = imresize(hr, scale=1.0 / scale).detach()
            restored_hr, restored_hr2 = model(lr_1_4, sec, sec_2, scale)
            recovered = revealNet(restored_hr, scale)
            recovered_2 = revealNet_2(restored_hr2, scale)

            # LOSS
            loss_hr = loss_fn[1](restored_hr, lr_1_2)
            loss_hr_2 = loss_fn[1](restored_hr2, hr)
            loss_sec = loss_fn[1](sec, recovered)
            loss_sec_2 = loss_fn[1](sec_2, recovered_2)

            loss = loss_hr + loss_sec + loss_hr_2 + loss_sec_2

            psnr_lr, psnr_hr = \
                psnr_calculator(recovered, sec), psnr_calculator(restored_hr, lr_1_2)
            ssim_lr, ssim_hr = \
                ssim_calculator(recovered, sec), ssim_calculator(restored_hr, lr_1_2)

            psnr_lr_2, psnr_hr_2 = \
                psnr_calculator(recovered_2, sec_2), psnr_calculator(restored_hr2, hr)
            ssim_lr_2, ssim_hr_2 = \
                ssim_calculator(recovered_2, sec_2), ssim_calculator(restored_hr2, hr)

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

            for m, x in zip([loss_meter, loss_hr_meter, loss_sec_meter, *psnr_meter, *ssim_meter],
                            [loss, loss_hr, loss_sec, psnr_lr, psnr_hr, psnr_lr_2, psnr_hr_2,
                             ssim_lr, ssim_hr, ssim_lr_2, ssim_hr_2]):
                m.update(x.item(), hr.shape[0])

            # Visualize after validation
        if main_process(cfg):
            sample_lr = torchvision.utils.make_grid(recovered.clamp(0.0, 1.0))
            sample_hr = torchvision.utils.make_grid(restored_hr.clamp(0.0, 1.0))
            writer.add_image('sample_results/res_lr', sample_lr, epoch + 1)
            writer.add_image('sample_results/res_hr', sample_hr, epoch + 1)

    return loss_meter.avg, loss_hr_meter.avg, loss_sec_meter.avg, [m.avg for m in psnr_meter], [m.avg for m in ssim_meter]


if __name__ == '__main__':
    main()
