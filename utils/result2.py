import cv2
import numpy as np

def draw_box(image, box, color=(0, 0, 255), thickness=2):
    # 在图像上绘制框
    x, y, w, h = box
    return cv2.rectangle(image.copy(), (x, y), (x + w, y + h), color, thickness)

def crop_image(image, box):
    # 根据给定的框坐标裁剪图像
    x, y, w, h = box
    return image[y:y+h, x:x+w]

def resize_images(image1, image2):
    # 将两张图片调整到相同的尺寸（较小的那个）
    h1, w1 = 128, 128
    h2, w2 = 128, 129
    if h1 != h2 or w1 != w2:
        min_h, min_w = min(h1, h2), min(w1, w2)
        image1 = cv2.resize(image1, (min_w, min_h))
        image2 = cv2.resize(image2, (min_w, min_h))
    return image1, image2

def create_difference_image(image1, image2):
    # 创建残差图
    diff_image = cv2.absdiff(image1, image2)
    diff_image = cv2.convertScaleAbs(diff_image, alpha=5)
    return diff_image

def save_image(image, output_path):
    # 保存图像
    cv2.imwrite(output_path, image)

# 指定小正方形区域的位置 (x, y, width, height)
box_coordinates = (50, 70, 50, 50)  # 示例坐标，请根据实际情况调整

# 第一张图像路径
# image_path_1 = '/opt/data/xiaobin/AIDN/utils/test/sec3.jpg'
image_path_1 = '/opt/data/xiaobin/AIDN/utils/test-udh/sec.jpg'
# 第二张图像路径
# image_path_2 = '/opt/data/xiaobin/AIDN/utils/test/rec3.jpg'
image_path_2 = '/opt/data/xiaobin/AIDN/utils/test-udh/rec.jpg'

# 读取原始图像
original_image_1 = cv2.imread(image_path_1)
original_image_2 = cv2.imread(image_path_2)

# 调整图像尺寸至相同
resized_image_1, resized_image_2 = resize_images(original_image_1, original_image_2)

# 绘制红框并保存带框图像
boxed_image_1 = draw_box(resized_image_1, box_coordinates)
boxed_image_2 = draw_box(resized_image_2, box_coordinates)
# save_image(boxed_image_1, '/opt/data/xiaobin/AIDN/utils/result/output_marked_image1_resized.jpg')
# save_image(boxed_image_2, '/opt/data/xiaobin/AIDN/utils/result/output_marked_image2_resized.jpg')

# 裁剪出框内的图像
cropped_image_1 = crop_image(resized_image_1, box_coordinates)
cropped_image_2 = crop_image(resized_image_2, box_coordinates)
# save_image(cropped_image_1, '/opt/data/xiaobin/AIDN/utils/result/output_res1.jpg')
# save_image(cropped_image_2, '/opt/data/xiaobin/AIDN/utils/result/output_res2.jpg')

# 确保裁剪后的图像尺寸相同
# cropped_image_1, cropped_image_2 = resize_images(cropped_image_1, cropped_image_2)

# 创建残差图并保存
# difference_image = create_difference_image(cropped_image_1, cropped_image_2)
difference_image = create_difference_image(original_image_1, original_image_2)
save_image(difference_image, '/opt/data/xiaobin/AIDN/utils/result/output_difference_image.jpg')


from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
def calculate_psnr_ssim(imageA, imageB):
    # 确保输入图像为灰度图
    if len(imageA.shape) == 3:
        imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # 计算PSNR
    psnr_value = psnr(imageA, imageB)

    # 计算SSIM，并返回SSIM图像（用于可视化差异）
    ssim_value, _ = ssim(imageA, imageB, full=True)

    return psnr_value, ssim_value

psnr_value, ssim_value = calculate_psnr_ssim(resized_image_1, resized_image_2)
# psnr_value, ssim_value = calculate_psnr_ssim(original_image_1, original_image_2)

print("Images have been processed and saved.")
print(f"PSNR: {psnr_value:.3f}")
print(f"SSIM: {ssim_value:.5f}")