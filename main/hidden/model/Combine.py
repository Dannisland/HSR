import numpy as np
import torch
import torch.nn as nn
from model.decoder import Decoder
from model.Steganalyzer import Steganalyzer
from model.encoder import Encoder

class combine_model:
    def __init__(self,device: torch.device):

        super(combine_model, self).__init__()

        self.encoder1=Encoder().to(device)
        self.s = Steganalyzer().to(device)
        self.decoder=Decoder().to(device)

        params=list(self.encoder1.parameters())+list(self.decoder.parameters())
        self.optimizer_enc_dec=torch.optim.Adam(params)
        self.optimizer_s = torch.optim.Adam(self.s.parameters())

        self.device = device

        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

    def train_on_batch(self, batch: list):

        clean_image,message = batch
        self.encoder1.train()
        self.decoder.train()
        self.s.train()

        with torch.enable_grad():
            # ---------------- Train the Steganalyzer -----------------------------
            self.optimizer_s.zero_grad()

            encoded_image=self.encoder1(clean_image,message)
            # 隐写分析鉴别器s
            D_output_f=self.s(encoded_image.detach())
            D_loss_f=self.bce_with_logits_loss(D_output_f,torch.zeros(D_output_f.size()).to(self.device))

            D_output_r = self.s(clean_image.detach())
            D_loss_r = self.bce_with_logits_loss(D_output_r, torch.ones(D_output_r.size()).to(self.device))

            s_loss = D_loss_r+ D_loss_f
            s_loss.backward()
            self.optimizer_s.step()

            # -----------------------  Train the encoder_decoder   -----------------------------
            self.optimizer_enc_dec.zero_grad()

            g_adv_out=self.s(encoded_image)
            g_loss_adv = self.bce_with_logits_loss(g_adv_out, torch.ones(g_adv_out.size()).to(self.device))

            decoded_message = self.decoder(encoded_image)
            g_loss_dec = self.mse_loss(decoded_message, message)
            x_loss=self.mse_loss(clean_image,encoded_image)

            en_de_loss = 1e-3*g_loss_adv+1*g_loss_dec+0.7*x_loss
            en_de_loss.backward()
            self.optimizer_enc_dec.step()


        # decoded_messages转到cpu numpy.取整.预定在（0,1）之间
        decoded_rounded = decoded_message.detach().cpu().numpy().round().clip(0, 1)
        # 计算平均错误率
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())) / (
                clean_image.shape[0] * message.shape[1])

        losses = {
            's_loss': s_loss.item(),
            'en_de_loss': en_de_loss.item(),
            'bitwise_error': bitwise_avg_err,
        }

        return losses, (clean_image,encoded_image)

    def validate_on_batch(self, batch: list):
        # 对由图像和消息组成的单批数据运行验证

        clean_image,message = batch
        # test1&2
        # self.encoder.eval()
        # test3&4
        self.encoder1.eval()
        self.decoder.eval()
        self.s.eval()

        with torch.no_grad():

            encoded_image=self.encoder1(clean_image,message)

            # 隐写分析鉴别器s
            D_output_f=self.s(encoded_image.detach())
            D_loss_f=self.bce_with_logits_loss(D_output_f,torch.zeros(D_output_f.size()).to(self.device))

            D_output_r = self.s(clean_image.detach())
            D_loss_r = self.bce_with_logits_loss(D_output_r, torch.ones(D_output_r.size()).to(self.device))

            s_loss = D_loss_r+ D_loss_f


            g_adv_out=self.s(encoded_image)
            g_loss_adv = self.bce_with_logits_loss(g_adv_out, torch.ones(g_adv_out.size()).to(self.device))

            decoded_message = self.decoder(encoded_image)
            g_loss_dec = self.mse_loss(decoded_message, message)
            x_loss = self.mse_loss(clean_image, encoded_image)

            en_de_loss = 1e-3*g_loss_adv+1*g_loss_dec+0.7*x_loss

        # decoded_messages转到cpu numpy.取整.预定在（0,1）之间
        decoded_rounded = decoded_message.detach().cpu().numpy().round().clip(0, 1)
        # 计算平均错误率
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())) / (
                clean_image.shape[0] * message.shape[1])

        losses = {
            's_loss': s_loss.item(),
            'en_de_loss': en_de_loss.item(),
            'bitwise_error': bitwise_avg_err,
        }
        return losses, (clean_image,encoded_image)

