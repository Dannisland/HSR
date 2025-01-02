import time
import kornia
import numpy as np
import torch
import utils
from options import *
from model.Combine import combine_model
from tensorboardX import SummaryWriter

def train(model: combine_model,
          device: torch.device,
          train_options: TrainingOptions
          ):
    # 加载数据集,用tip2018
    train_data, val_data = utils.get_data_loaders(train_options)

    writer_path = utils.create_folder_for_run(train_options.experiment_name)
    writer = SummaryWriter(writer_path)

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):

        print(time.strftime('%Y-%m-%d %H:%M:%S'), "epoch", epoch)
        step = 0
        s_loss=0
        en_de_loss=0
        bitwise_error = 0
        SUM_PSNR = 0
        for clean_image,noise_image in train_data:
            # 加载图片与消息
            noise_image = noise_image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (noise_image.shape[0], 30))).to(device)
            losses, (image1,image2) = model.train_on_batch([noise_image, message])

            s_loss += losses['s_loss']
            en_de_loss += losses['en_de_loss']
            bitwise_error+=losses['bitwise_error']
            SUM_PSNR = SUM_PSNR - kornia.losses.psnr_loss(image1,image2, 1).item()

            step += 1
            if(step==1):
                utils.save_images([image1, image2],
                                  epoch, train_options.experiment_name,True,train_options.nrow)

        s_loss = s_loss / step
        en_de_loss = en_de_loss / step
        bitwise_error=bitwise_error/step
        SUM_PSNR = SUM_PSNR / step

        writer.add_scalar("train_s_loss",s_loss,global_step=epoch)
        writer.add_scalar("train_en_de_loss", en_de_loss, global_step=epoch)
        writer.add_scalar("train_bitwise_error", bitwise_error, global_step=epoch)
        writer.add_scalar("train_SUM_PSNR", SUM_PSNR, global_step=epoch)
        utils.save_checkpoint(model,train_options.experiment_name,epoch)
        print("train_bitwise_error", bitwise_error)
        print("train_SUM_PSNR", SUM_PSNR)


        step = 0
        s_loss=0
        en_de_loss=0
        bitwise_error = 0
        SUM_PSNR=0
        for clean_image,noise_image in val_data:
            # 加载图片与消息
            noise_image = noise_image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (noise_image.shape[0], 30))).to(device)
            losses, (image1, image2) = model.validate_on_batch([noise_image, message])

            s_loss += losses['s_loss']
            en_de_loss += losses['en_de_loss']
            bitwise_error += losses['bitwise_error']
            SUM_PSNR = SUM_PSNR - kornia.losses.psnr_loss(image1, image2, 1).item()

            step += 1
            if (step == 1):
                utils.save_images([image1, image2],
                                  epoch, train_options.experiment_name, False, train_options.nrow)

        s_loss = s_loss / step
        en_de_loss = en_de_loss / step
        bitwise_error = bitwise_error / step
        SUM_PSNR = SUM_PSNR / step

        writer.add_scalar("val_s_loss",s_loss,global_step=epoch)
        writer.add_scalar("val_en_de_loss", en_de_loss, global_step=epoch)
        writer.add_scalar("val_bitwise_error", bitwise_error, global_step=epoch)
        writer.add_scalar("val_SUM_PSNR", SUM_PSNR, global_step=epoch)

        print("val_bitwise_error",bitwise_error)
        print("val_SUM_PSNR",SUM_PSNR)

    writer.close()