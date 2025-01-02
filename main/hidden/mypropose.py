import numpy as np
import torch
from torchvision.utils import save_image

import utils
from model.decoder import Decoder
from model.encoder import Encoder
from model.generator import MoireCNN
from options import TrainingOptions
import torchvision.transforms as T

# 01分布的高斯噪声
def Gauss_noise(inputs, noise_factor=0.15):
    noisy = inputs + torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0., 1.)
    return noisy

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#高斯模糊
blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=1)
# 声明模型
encoder1 = Encoder().to(device)
encoder2 = Encoder().to(device)
encoder3 = Encoder().to(device)
encoder4 = Encoder().to(device)
encoder5 = Encoder().to(device)
generator = MoireCNN(input_channel=6).to(device)
decoder=Decoder().to(device)
# 加载模型
checkpoint = torch.load('model/moire_checkpoint_16.pth', map_location=device)
# 加载预训练的生成器网络参数
generator.load_state_dict((checkpoint['Generator']))
# 加载模型
checkpoint = torch.load('checkpoints/bit30-test2_2--epoch-100.pth', map_location=device)
encoder1.load_state_dict((checkpoint['encoder1']))
encoder2.load_state_dict((checkpoint['encoder2']))
encoder3.load_state_dict((checkpoint['encoder3']))
encoder4.load_state_dict((checkpoint['encoder4']))
encoder5.load_state_dict((checkpoint['encoder5']))
decoder.load_state_dict((checkpoint['decoder']))
# 加载数据集
train_options = TrainingOptions(
    batch_size=5,
    # train_folder='/workshop/tiewei/datasets/tip2018',
    # validation_folder='/workshop/tiewei/datasets/tip2018/testData',
    train_folder='/opt/data/tiewei/datasets/tip2018',
    validation_folder='/opt/data/tiewei/datasets/tip2018/testData',
    train_data_len=2000, val_data_len=5000)
train_data,val_data = utils.get_data_loaders(train_options)


# 开始计算
# clean_image
bit_err1=0
# clean_image,noise
bit_err2=0
step = 0
for clean_image, noise_image in val_data:
    step += 1
    # 验证模式
    encoder1.eval()
    encoder2.eval()
    encoder3.eval()
    encoder4.eval()
    encoder5.eval()
    decoder.eval()
    generator.eval()
    # 加载图片与消息
    clean_image = clean_image.to(device)
    message = torch.Tensor(np.random.choice([0, 1], (clean_image.shape[0], 30))).to(device)

    init_noise = torch.normal(0, 1, clean_image.shape).to(device)
    init_clean = torch.cat([init_noise, clean_image], dim=1)
    x1, x2, x3, x4, x5 = generator(init_clean)

    x1_encoded = encoder1(x1, message)
    x2_encoded = encoder2(x2, message)
    x3_encoded = encoder3(x3, message)
    x4_encoded = encoder4(x4, message)
    x5_encoded = encoder5(x5, message)

    x_encoded = x1_encoded + x2_encoded + x3_encoded + x4_encoded + x5_encoded

    image_encoded = clean_image + x_encoded
    image_encoded = image_encoded.clamp(0, 1)

    # 没有噪声
    decoded_message1 = decoder(image_encoded)
    # 添加噪声
    # encoded_image1=blurrer(image_encoded)
    encoded_image1=Gauss_noise(image_encoded)

    decoded_message2 = decoder(encoded_image1)
    # decoded_messages转到cpu numpy.取整.预定在（0,1）之间
    decoded_rounded = decoded_message1.detach().cpu().numpy().round().clip(0, 1)
    # 计算平均错误率
    bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())) / (
        clean_image.shape[0] * message.shape[1])
    bit_err1=bit_err1+bitwise_avg_err

    decoded_rounded = decoded_message2.detach().cpu().numpy().round().clip(0, 1)
    # 计算平均错误率
    bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())) / (
        clean_image.shape[0] * message.shape[1])
    bit_err2 = bit_err2 + bitwise_avg_err

bit_err1=bit_err1/step
bit_err2=bit_err2/step
save_image(encoded_image1,'mypropose.png')
print("bit_err1",bit_err1)
print("bit_err2",bit_err2)


