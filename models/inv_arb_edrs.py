import torch
import torchvision

from base.base_model import BaseModel
from models.arbedrs import EDRS
from models.arbedrs_inv import EDRS as EDRS_INV
from models.lib.quantization import Quantization, Quantization_RS

class InvArbEDRS(BaseModel):
    def __init__(self, cfg=None):
        super(InvArbEDRS, self).__init__()
        self.cfg = cfg
        # cfg.rescale = 'down'
        # self.down_net = EDRS(cfg)
        # if cfg.quantization and cfg.quantization_type == 'naive':
        #     self.quantizer = Quantization()
        # elif cfg.quantization and cfg.quantization_type == 'round_soft':
        #     self.quantizer = Quantization_RS()
        # else:
        #     self.quantizer = None

        if cfg.jpeg:
            if cfg.jpeg_type == 'DiffJPEG':
                from models.lib.jpg_module_DiffJPEG import JPGQuantizeFun
                self.jpeg = JPGQuantizeFun(quality=90)
            else:
                raise NotImplementedError(
                    'JPEG Compression Simulator {' + cfg.jpeg_type + '} has not been implemented!')
        cfg.rescale = 'up'
        self.up_net = EDRS(cfg, ifsec=True)

        self.up_net_2 = EDRS(cfg, ifsec=True)

        # self.up_net_3 = EDRS(cfg, ifsec=True)

    def forward(self, x, sec, sec_2, scale, precalculated_lr=None):
        B, C, H2, W2 = sec_2.shape

        lr_processed = torch.cat([x, sec], dim=1)
        sr_1 = self.up_net(lr_processed, scale, H2, H2)

        lr_processed_2 = torch.cat([sr_1, sec_2], dim=1)
        sr_2 = self.up_net_2(lr_processed_2, scale, int(H2 * scale), int(H2 * scale))

        return sr_1, sr_2

class InvArbEDRS_Inv(BaseModel):
    def __init__(self, cfg=None):
        super(InvArbEDRS_Inv, self).__init__()
        self.cfg = cfg
        cfg.rescale = 'down'
        self.down_net = EDRS_INV(cfg, ifsec=True)

        cfg.rescale = 'up'
        self.up_net = EDRS_INV(cfg, ifsec=True)

        # self.up_net_2 = EDRS_INV(cfg, ifsec=True)

        # self.up_net_3 = EDRS_INV(cfg, ifsec=True)

    def forward(self, input, scale, rev=False):
        B, C, H, W = input.shape
        if H == 86:
            H, W = 85.5, 85.5
        if not rev:
            output = self.up_net(input, scale, int(H*scale), int(W*scale))
        else:
            output = self.down_net(input, scale)

        return output



class InvArbEDRS_Backup(BaseModel):
    def __init__(self, cfg=None):
        super(InvArbEDRS_Backup, self).__init__()
        self.cfg = cfg
        cfg.rescale = 'down'
        self.down_net = EDRS(cfg)
        if cfg.quantization and cfg.quantization_type == 'naive':
            self.quantizer = Quantization()
        elif cfg.quantization and cfg.quantization_type == 'round_soft':
            self.quantizer = Quantization_RS()
        else:
            self.quantizer = None

        if cfg.jpeg:
            if cfg.jpeg_type == 'DiffJPEG':
                from models.lib.jpg_module_DiffJPEG import JPGQuantizeFun
                self.jpeg = JPGQuantizeFun(quality=90)
            else:
                raise NotImplementedError(
                    'JPEG Compression Simulator {' + cfg.jpeg_type + '} has not been implemented!')
        cfg.rescale = 'up'
        self.up_net = EDRS(cfg)

    def forward(self, x, scale, precalculated_lr=None):
        B, C, H, W = x.shape
        lr = self.down_net(x, 1.0 / scale)

        if precalculated_lr is None:  # directly use the LR images downscaled by the encoder
            if self.quantizer is not None:
                lr_processed = self.quantizer(lr)
            else:
                lr_processed = lr

            lr_processed = self.jpeg(lr_processed) if self.cfg.jpeg else lr_processed
            sr = self.up_net(lr_processed, scale, H, W)
        else:  # use the provided LR image for upscaling
            sr = self.up_net(precalculated_lr, scale, H, W)

        return lr, sr
