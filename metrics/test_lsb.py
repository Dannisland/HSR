import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# LSB隐藏方法
def hide_image(cover_img, secret_img, start_bit=0):
    """
    将秘密图像隐藏到载体图像中
    :param cover_img: 载体图像
    :param secret_img: 秘密图像（灰度图）
    :param start_bit: 从第几位开始隐藏（0表示最低位，1表示次低位，依此类推）
    :return: 含密图像
    """
    stego_img = cover_img.copy()
    for i in range(secret_img.shape[0]):
        for j in range(secret_img.shape[1]):
            for k in range(3):  # 对RGB三个通道进行操作
                # 将秘密图像的1位隐藏到载体图像的指定位置
                stego_img[i, j, k] = (cover_img[i, j, k] & ~(1 << start_bit)) | ((secret_img[i, j] >> (7 - start_bit)) & 0x01) << start_bit
    return stego_img

# LSB提取方法
def extract_image(stego_img, secret_img_shape, start_bit=0):
    """
    从含密图像中提取秘密图像
    :param stego_img: 含密图像
    :param secret_img_shape: 秘密图像的形状
    :param start_bit: 从第几位开始提取（0表示最低位，1表示次低位，依此类推）
    :return: 提取的秘密图像（灰度图）
    """
    extracted_img = np.zeros(secret_img_shape, dtype=np.uint8)
    for i in range(secret_img_shape[0]):
        for j in range(secret_img_shape[1]):
            for k in range(3):  # 对RGB三个通道进行操作
                # 从含密图像的指定位置提取1位
                extracted_img[i, j] = (extracted_img[i, j] << 1) | ((stego_img[i, j, k] >> start_bit) & 0x01)
    return extracted_img

# 读取载体图像和秘密图像
cover_img = cv2.imread('/opt/data/xiaobin/AIDN/assets/lr/lr0.jpg')
secret_img1 = cv2.cvtColor(cv2.imread('/opt/data/xiaobin/AIDN/assets/sec/sec0.jpg'), cv2.COLOR_BGR2GRAY)
secret_img2 = cv2.cvtColor(cv2.imread('/opt/data/xiaobin/AIDN/assets/sec/sec1.jpg'), cv2.COLOR_BGR2GRAY)

if cover_img.shape[0] != secret_img1.shape[0] or cover_img.shape[1] != secret_img1.shape[1] or \
   cover_img.shape[0] != secret_img2.shape[0] or cover_img.shape[1] != secret_img2.shape[1]:
    raise ValueError("载体图像和秘密图像大小必须相同")

# 第一步：隐藏第一张秘密图像到载体图像的最低有效位（LSB），生成含密图像1
stego_img1 = hide_image(cover_img, secret_img1, start_bit=0)

# 第二步：隐藏第二张秘密图像到含密图像1的次低有效位，生成含密图像2
stego_img2 = hide_image(stego_img1, secret_img2, start_bit=1)

# 保存含密图像
cv2.imwrite('/opt/data/xiaobin/AIDN/metrics/lsb-result/stego_image1.jpg', stego_img1)
cv2.imwrite('/opt/data/xiaobin/AIDN/metrics/lsb-result/stego_image2.jpg', stego_img2)

# 提取秘密图像
extracted_secret_img1 = extract_image(stego_img2, secret_img1.shape, start_bit=0)
extracted_secret_img2 = extract_image(stego_img2, secret_img2.shape, start_bit=1)

# 保存提取的秘密图像
cv2.imwrite('/opt/data/xiaobin/AIDN/metrics/lsb-result/extracted_secret_image1.jpg', extracted_secret_img1)
cv2.imwrite('/opt/data/xiaobin/AIDN/metrics/lsb-result/extracted_secret_image2.jpg', extracted_secret_img2)

# 计算PSNR和SSIM
# 含密图像1与载体图像的PSNR和SSIM
psnr_stego1_cover = psnr(cover_img, stego_img1)
ssim_stego1_cover = ssim(cover_img, stego_img1, channel_axis=2)

# 含密图像2与载体图像的PSNR和SSIM
psnr_stego2_cover = psnr(cover_img, stego_img2)
ssim_stego2_cover = ssim(cover_img, stego_img2, channel_axis=2)

# 提取的秘密图像1与原始秘密图像1的PSNR和SSIM
psnr_extracted1_secret1 = psnr(secret_img1, extracted_secret_img1)
ssim_extracted1_secret1 = ssim(secret_img1, extracted_secret_img1)

# 提取的秘密图像2与原始秘密图像2的PSNR和SSIM
psnr_extracted2_secret2 = psnr(secret_img2, extracted_secret_img2)
ssim_extracted2_secret2 = ssim(secret_img2, extracted_secret_img2)

# 输出结果
print("含密图像1与载体图像的PSNR:", psnr_stego1_cover)
print("含密图像1与载体图像的SSIM:", ssim_stego1_cover)
print("含密图像2与载体图像的PSNR:", psnr_stego2_cover)
print("含密图像2与载体图像的SSIM:", ssim_stego2_cover)
print("提取的秘密图像1与原始秘密图像1的PSNR:", psnr_extracted1_secret1)
print("提取的秘密图像1与原始秘密图像1的SSIM:", ssim_extracted1_secret1)
print("提取的秘密图像2与原始秘密图像2的PSNR:", psnr_extracted2_secret2)
print("提取的秘密图像2与原始秘密图像2的SSIM:", ssim_extracted2_secret2)