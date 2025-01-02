import os

import numpy as np
import torch
from torchvision.utils import save_image
import utils
from model.decoder import Decoder
from model.encoder import Encoder
from options import TrainingOptions
import torchvision.transforms as T
from torchvision.transforms.functional import rotate


# 01分布的高斯噪声
def Gauss_noise(inputs, noise_factor=0.15):
    noisy = inputs + torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0., 1.)
    return noisy


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# 高斯模糊
blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=0.9)
# 加载预训练模型
checkpoint = torch.load('checkpoints/hidden-no-noise--epoch-200.pyt', map_location=device)
en_de = checkpoint['enc-dec-model']
# 分离参数
# dict转list
lst_key = []
lst_value = []
noise_key = []
noise_value = []
for key, value in en_de.items():
    if key.find("noiser") == -1:
        lst_key.append(key[8:])
        lst_value.append(value)
    else:
        noise_key.append(key[7:])
        noise_value.append(value)
# list转dict
# 并对encode与decode参数进行切割
en_dict = dict(zip(lst_key[0:37], lst_value[0:37]))
de_dict = dict(zip(lst_key[37:], lst_value[37:]))
encoder = Encoder().to(device)
encoder.load_state_dict(en_dict)
decoder = Decoder().to(device)
decoder.load_state_dict(de_dict)

# 加载数据集
train_options = TrainingOptions(
    batch_size=1,
    # train_folder='/workshop/tiewei/datasets/tip2018',
    # validation_folder='/workshop/tiewei/datasets/tip2018/testData',
    train_folder='/opt/data/xiaobin/Project/AIDN/Data/Set5',
    validation_folder='/opt/data/xiaobin/Project/AIDN/Data/Set5',
    train_data_len=2000,
    val_data_len=5000
)
train_data, val_data = utils.get_data_loaders(train_options)

print("start")

bit_err1 = 0
bit_err2 = 0
bit_err3 = 0
bit_err4 = 0
bit_err5 = 0
step = 0
for clean_image, noise_image in val_data:
    # for clean_image, noise_image in train_data:
    step = step + 1
    # 验证模式
    encoder.eval()
    decoder.eval()

    # 加载图片与消息
    clean_image = clean_image.to(device)
    # message = torch.Tensor(np.random.choice([0, 1], (clean_image.shape[0], 30))).to(device)
    message = torch.Tensor([[1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                             1., 0., 1., 1., 0., 1., 1.]]).to(device)

    # 输入干净图
    # encoded_image1 = encoder(clean_image, message)
    # encoded_image1.clamp(0, 1)
    encoded_image1 = clean_image
    # 没有噪声
    decoded_message1 = decoder(encoded_image1)
    # 添加噪声
    # encoded_image1=blurrer(encoded_image1)

    encoded_image2 = rotate(encoded_image1, 30, expand=True)
    encoded_image2 = rotate(encoded_image2, -30)
    encoded_image2 = T.CenterCrop(128)(encoded_image2)

    encoded_image3 = rotate(encoded_image1, 45, expand=True)
    encoded_image3 = rotate(encoded_image3, -45)
    encoded_image3 = T.CenterCrop(128)(encoded_image3)

    # encoded_image2 = T.Resize(112)(encoded_image1)
    # encoded_image2 = T.Resize(128)(encoded_image2)

    # encoded_image3 = T.Resize(96)(encoded_image1)
    # encoded_image3 = T.Resize(128)(encoded_image3)

    # encoded_image4 = T.Resize(80)(encoded_image1)
    # encoded_image4 = T.Resize(128)(encoded_image4)

    # encoded_image5 = T.Resize(64)(encoded_image1)
    # encoded_image5 = T.Resize(128)(encoded_image5)

    # encoded_image2=Gauss_noise(encoded_image1,0.001)
    # encoded_image3=Gauss_noise(encoded_image1,0.002)
    # encoded_image4=Gauss_noise(encoded_image1,0.005)

    decoded_message2 = decoder(encoded_image2)
    decoded_message3 = decoder(encoded_image3)
    # decoded_message4 = decoder(encoded_image4)
    # decoded_message5 = decoder(encoded_image5)

    # decoded_messages转到cpu numpy.取整.预定在（0,1）之间
    decoded_rounded = decoded_message1.detach().cpu().numpy().round().clip(0, 1)
    # 计算平均错误率
    bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())) / (
            clean_image.shape[0] * message.shape[1])
    bit_err1 = bit_err1 + bitwise_avg_err

    decoded_rounded = decoded_message2.detach().cpu().numpy().round().clip(0, 1)
    # 计算平均错误率
    bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())) / (
            clean_image.shape[0] * message.shape[1])
    bit_err2 = bit_err2 + bitwise_avg_err

    # decoded_messages转到cpu numpy.取整.预定在（0,1）之间
    decoded_rounded = decoded_message3.detach().cpu().numpy().round().clip(0, 1)
    # 计算平均错误率
    bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())) / (
            clean_image.shape[0] * message.shape[1])
    bit_err3 = bit_err3 + bitwise_avg_err

    # save_image(encoded_image1, 'hidden' + str(step) + '.png')

bit_err1 = bit_err1 / step
bit_err2 = bit_err2 / step
bit_err3 = bit_err3 / step

print("bit_err1", bit_err1)
print("bit_err2", bit_err2)
print("bit_err3", bit_err3)
