import os
import numpy as np
from PIL import Image
import random
import gzip

def load_mnist(folder, img_file_name, label_file_name):
    """
    从指定的文件夹加载 MNIST 数据集。
    
    参数:
    folder (str): 存储 MNIST 数据文件的文件夹路径。
    img_file_name (str): 图像文件名（gzip 压缩格式）。
    label_file_name (str): 标签文件名（gzip 压缩格式）。
    
    返回:
    tuple: 包含图像数据 (x_set) 和标签数据 (y_set) 的元组。
    """
    # 读取标签文件并解压缩，读取数据并跳过前 8 字节（元数据）
    with gzip.open(os.path.join(folder, label_file_name), 'rb') as lbpath:
        y_set = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    
    # 读取图像文件并解压缩，读取数据并跳过前 16 字节（元数据），然后将数据重塑为 (样本数, 28, 28)
    with gzip.open(os.path.join(folder, img_file_name), 'rb') as imgpath:
        x_set = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_set), 28, 28)
    
    return x_set, y_set

def invert_image(image):
    """
    对图像进行反转处理，将像素值从 0-255 反转为 255-0。
    
    参数:
    image (numpy.ndarray): 输入的图像数据（28x28）。
    
    返回:
    numpy.ndarray: 反转后的图像数据。
    """
    return 255 - image

def rotate_image(image, max_angle=90):
    """
    旋转图像到随机角度，角度范围是 -max_angle 到 +max_angle。
    
    参数:
    image (numpy.ndarray): 输入的图像数据（28x28）。
    max_angle (float): 最大旋转角度。
    
    返回:
    PIL.Image: 旋转后的图像。
    """
    angle = random.uniform(-max_angle, max_angle)  # 随机生成旋转角度
    return Image.fromarray(image).rotate(angle, fillcolor=255)  # 使用填充色255（白色）进行旋转

def save_image(image, save_path, file_name):
    """
    将图像保存为 JPEG 格式。
    
    参数:
    image (numpy.ndarray): 输入的图像数据（28x28）。
    save_path (str): 保存图像的路径。
    file_name (str): 图像文件名。
    """
    img = Image.fromarray(image).convert('L')  # 将图像数据转换为 PIL 图像对象，并确保是灰度模式
    img.save(os.path.join(save_path, f'{file_name}.jpg'))  # 保存图像为 JPEG 文件

def augment_and_save(x, y, save_path, augmentations):
    """
    对数据进行增强处理并保存图像和标签。
    
    参数:
    x (numpy.ndarray): 输入的图像数据。
    y (numpy.ndarray): 图像对应的标签。
    save_path (str): 保存增强后图像和标签的路径。
    augmentations (list of functions): 图像增强函数列表。
    """
    os.makedirs(save_path, exist_ok=True)  # 创建保存目录，如果目录不存在的话

    for i, image in enumerate(x):
        augmented_image = image
        for augmentation in augmentations:
            augmented_image = augmentation(augmented_image)  # 应用每个增强函数
        
        save_image(np.array(augmented_image), save_path, i)  # 将增强后的图像保存为 JPEG 文件

    # 保存labels.txt到图片的上一级目录
    parent_path = os.path.dirname(save_path)
    labels_file_path = os.path.join(parent_path, 'labels.txt')
    with open(labels_file_path, 'w') as f:
        for label in y:
            f.write(str(label) + '\n')

# 定义数据文件路径
root_path = './data/mnist/MNIST/raw'
img_file_path = 'train-images-idx3-ubyte.gz'
label_file_path = 'train-labels-idx1-ubyte.gz'

# 加载 MNIST 数据集
raw_x, raw_y = load_mnist(root_path, img_file_path, label_file_path)

# 定义保存路径
save_root_path = './data/process'

# 应用不同的数据增强方法并保存
augment_and_save(raw_x, raw_y, os.path.join(save_root_path, 'invert'), [invert_image])
augment_and_save(raw_x, raw_y, os.path.join(save_root_path, 'rotate90'), [lambda img: rotate_image(img, 90)])
augment_and_save(raw_x, raw_y, os.path.join(save_root_path, 'invert_and_rotate'), [invert_image, lambda img: rotate_image(img, 90)])
