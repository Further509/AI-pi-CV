import os
import shutil
import random

# 定义源图像和标签的路径
Images_path = 'ImageSets'  
Labels_path = 'Label'

# 定义训练集和验证集的保存路径
train_labels = 'labels/train'
val_labels = 'labels/val'

train_images = 'images/train'
val_images = 'images/val'

# 定义验证集划分的比例或数量
radio = 0.1  
nums = 100  
is_radio = True  

# 如果保存路径不存在，则创建文件夹
if not os.path.exists(train_images):
    os.mkdir(train_images)
if not os.path.exists(val_images):
    os.mkdir(val_images)

if not os.path.exists(train_labels):
    os.mkdir(train_labels)
if not os.path.exists(val_labels):
    os.mkdir(val_labels)

# 获取源图像路径下的所有文件名
Imgs = os.listdir(Images_path)

# 根据is_radio的值决定是按比例还是按数量划分验证集
if is_radio:
    val_nums = int(len(Imgs) * radio)
else:
    val_nums = nums

# 随机选择文件名列表中的文件，作为验证集
val_Imgs = random.sample(Imgs, val_nums)

# 遍历验证集图像列表，移动图像和对应的标签文件到验证集文件夹
for val_name in val_Imgs:
    shutil.move(os.path.join(Images_path, val_name), os.path.join(val_images, val_name))
    val_name = val_name[:-3] + 'txt'  # 将jpg后缀改为txt
    shutil.move(os.path.join(Labels_path, val_name), os.path.join(val_labels, val_name))

# 如果除去验证集后还有剩余图像，将剩余图像移动到训练集文件夹
if (len(Imgs) - len(val_Imgs)) > 0:
    for i in val_Imgs:
        if i in Imgs:
            Imgs.remove(i)
    train_Imgs = Imgs
    for train_name in train_Imgs:
        shutil.move(os.path.join(Images_path, train_name), os.path.join(train_images, train_name))
        train_name = train_name[:-3] + 'txt'  # 将jpg后缀改为txt
        shutil.move(os.path.join(Labels_path, train_name), os.path.join(train_labels, train_name))
        