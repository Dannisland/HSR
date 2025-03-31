#!/usr/bin/env python
import shutil

import torch
from os.path import join
import torch.distributed as dist
from .utilities import check_makedirs
from collections import OrderedDict
from torch.nn.parallel import DataParallel, DistributedDataParallel


def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    return lr


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def save_checkpoint(model, model2, model3, model4, other_state: dict, sav_path, is_best=False,
#                     filename='AIDN_last.pth.tar'):
#     if isinstance(model, (DistributedDataParallel, DataParallel)):
#         weight = model.module.state_dict()
#     elif isinstance(model, torch.nn.Module):
#         weight = model.state_dict()
#     else:
#         raise ValueError('model must be nn.Module or nn.DataParallel!')
#
#     if isinstance(model2, (DistributedDataParallel, DataParallel)):
#         weight2 = model2.module.state_dict()
#     elif isinstance(model2, torch.nn.Module):
#         weight2 = model2.state_dict()
#     else:
#         raise ValueError('model must be nn.Module or nn.DataParallel!')
#
#     if isinstance(model3, (DistributedDataParallel, DataParallel)):
#         weight3 = model3.module.state_dict()
#     elif isinstance(model3, torch.nn.Module):
#         weight3 = model3.state_dict()
#     else:
#         raise ValueError('model must be nn.Module or nn.DataParallel!')
#
#     if isinstance(model4, (DistributedDataParallel, DataParallel)):
#         weight4 = model4.module.state_dict()
#     elif isinstance(model4, torch.nn.Module):
#         weight4 = model4.state_dict()
#     else:
#         raise ValueError('model must be nn.Module or nn.DataParallel!')
#
#     check_makedirs(sav_path)
#
#     other_state['state_dict'] = weight
#     other_state['reveal'] = weight2
#     other_state['reveal_2'] = weight3
#     other_state['reveal_3'] = weight4
#
#     filename = join(sav_path, filename)
#     torch.save(other_state, filename)
#     if is_best:
#         shutil.copyfile(filename, join(sav_path, 'AIDN.pth.tar'))


def save_checkpoint(model, model2, model3, other_state: dict, sav_path, is_best=False, filename='AIDN_last.pth.tar'):
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        weight = model.module.state_dict()
    elif isinstance(model, torch.nn.Module):
        weight = model.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')

    if isinstance(model2, (DistributedDataParallel, DataParallel)):
        weight2 = model2.module.state_dict()
    elif isinstance(model2, torch.nn.Module):
        weight2 = model2.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')

    if isinstance(model3, (DistributedDataParallel, DataParallel)):
        weight3 = model3.module.state_dict()
    elif isinstance(model3, torch.nn.Module):
        weight3 = model3.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')


    check_makedirs(sav_path)

    other_state['state_dict'] = weight
    other_state['reveal'] = weight2
    other_state['reveal_2'] = weight3

    filename = join(sav_path, filename)
    torch.save(other_state, filename)
    if is_best:
        shutil.copyfile(filename, join(sav_path, 'AIDN.pth.tar'))

def save_checkpoint_inv_msg(model, other_state: dict, sav_path, is_best=False, filename='AIDN_last.pth.tar'):
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        weight = model.module.state_dict()
    elif isinstance(model, torch.nn.Module):
        weight = model.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')

    check_makedirs(sav_path)

    other_state['state_dict'] = weight

    filename = join(sav_path, filename)
    torch.save(other_state, filename)
    if is_best:
        shutil.copyfile(filename, join(sav_path, 'AIDN.pth.tar'))

def save_checkpoint_loop1(model, model2, other_state: dict, sav_path, is_best=False, filename='AIDN_last.pth.tar'):
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        weight = model.module.state_dict()
    elif isinstance(model, torch.nn.Module):
        weight = model.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')

    if isinstance(model2, (DistributedDataParallel, DataParallel)):
        weight2 = model2.module.state_dict()
    elif isinstance(model2, torch.nn.Module):
        weight2 = model2.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')


    check_makedirs(sav_path)

    other_state['state_dict'] = weight
    other_state['reveal'] = weight2

    filename = join(sav_path, filename)
    torch.save(other_state, filename)
    if is_best:
        shutil.copyfile(filename, join(sav_path, 'AIDN.pth.tar'))

def save_checkpoint_loop2(model, model2, model3, model4, other_state: dict, sav_path, is_best=False, filename='AIDN_last.pth.tar'):
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        weight = model.module.state_dict()
    elif isinstance(model, torch.nn.Module):
        weight = model.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')

    if isinstance(model2, (DistributedDataParallel, DataParallel)):
        weight2 = model2.module.state_dict()
    elif isinstance(model2, torch.nn.Module):
        weight2 = model2.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')

    if isinstance(model3, (DistributedDataParallel, DataParallel)):
        weight3 = model3.module.state_dict()
    elif isinstance(model3, torch.nn.Module):
        weight3 = model3.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')

    if isinstance(model4, (DistributedDataParallel, DataParallel)):
        weight4 = model4.module.state_dict()
    elif isinstance(model4, torch.nn.Module):
        weight4 = model4.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')

    check_makedirs(sav_path)

    other_state['state_dict'] = weight
    other_state['reveal'] = weight2
    other_state['state_dict2'] = weight3
    other_state['reveal2'] = weight4

    filename = join(sav_path, filename)
    torch.save(other_state, filename)
    if is_best:
        shutil.copyfile(filename, join(sav_path, 'AIDN.pth.tar'))

def save_checkpoint_loop3(model, model2, model3, model4, other_state: dict, sav_path, is_best=False, filename='AIDN_last.pth.tar'):
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        weight = model.module.state_dict()
    elif isinstance(model, torch.nn.Module):
        weight = model.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')

    if isinstance(model2, (DistributedDataParallel, DataParallel)):
        weight2 = model2.module.state_dict()
    elif isinstance(model2, torch.nn.Module):
        weight2 = model2.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')

    if isinstance(model3, (DistributedDataParallel, DataParallel)):
        weight3 = model3.module.state_dict()
    elif isinstance(model3, torch.nn.Module):
        weight3 = model3.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')

    if isinstance(model4, (DistributedDataParallel, DataParallel)):
        weight4 = model4.module.state_dict()
    elif isinstance(model4, torch.nn.Module):
        weight4 = model4.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')


    check_makedirs(sav_path)

    other_state['state_dict'] = weight
    other_state['reveal'] = weight2
    other_state['reveal_2'] = weight3
    other_state['reveal_3'] = weight4

    filename = join(sav_path, filename)
    torch.save(other_state, filename)
    if is_best:
        shutil.copyfile(filename, join(sav_path, 'AIDN.pth.tar'))


def save_checkpoint_loop2_imp(model, model2, model3, model4, model5, other_state: dict, sav_path, is_best=False, filename='AIDN_last.pth.tar'):
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        weight = model.module.state_dict()
    elif isinstance(model, torch.nn.Module):
        weight = model.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')

    if isinstance(model2, (DistributedDataParallel, DataParallel)):
        weight2 = model2.module.state_dict()
    elif isinstance(model2, torch.nn.Module):
        weight2 = model2.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')

    if isinstance(model3, (DistributedDataParallel, DataParallel)):
        weight3 = model3.module.state_dict()
    elif isinstance(model3, torch.nn.Module):
        weight3 = model3.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')

    if isinstance(model4, (DistributedDataParallel, DataParallel)):
        weight4 = model4.module.state_dict()
    elif isinstance(model4, torch.nn.Module):
        weight4 = model4.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')

    if isinstance(model5, (DistributedDataParallel, DataParallel)):
        weight5 = model5.module.state_dict()
    elif isinstance(model5, torch.nn.Module):
        weight5 = model5.state_dict()
    else:
        raise ValueError('model must be nn.Module or nn.DataParallel!')

    check_makedirs(sav_path)

    other_state['state_dict'] = weight
    other_state['reveal'] = weight2
    other_state['state_dict2'] = weight3
    other_state['reveal2'] = weight4
    other_state['imp_model'] = weight5

    filename = join(sav_path, filename)
    torch.save(other_state, filename)
    if is_best:
        shutil.copyfile(filename, join(sav_path, 'AIDN.pth.tar'))

def load_state_dict(model, state_dict, strict=False):
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        model.module.load_state_dict(state_dict, strict=strict)
    else:
        model.load_state_dict(state_dict, strict=strict)


def state_dict_remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:]  # remove 'module.' of dataparallel
        name = k.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict


def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt
