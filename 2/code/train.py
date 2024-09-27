import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

# 定义卷积神经网络 (CNN) 类
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 卷积层1: 输入1通道，输出16通道，卷积核大小3x3，填充1
            nn.ReLU(),  # 激活函数 ReLU
            nn.MaxPool2d(2, 2),  # 最大池化层: 池化窗口大小2x2，步幅2
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 卷积层2: 输入16通道，输出32通道，卷积核大小3x3，填充1
            nn.ReLU(),  # 激活函数 ReLU
            nn.MaxPool2d(2, 2),  # 最大池化层: 池化窗口大小2x2，步幅2
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 卷积层3: 输入32通道，输出64通道，卷积核大小3x3，填充1
            nn.ReLU(),  # 激活函数 ReLU
        )
        
        # 定义全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 展平层: 将多维输入展平成一维
            nn.Linear(64 * 7 * 7, 128),  # 全连接层1: 输入维度64x7x7，输出维度128
            nn.ReLU(),  # 激活函数 ReLU
            nn.Linear(128, 10)  # 全连接层2: 输入维度128，输出维度10（分类数）
        )
        
    def forward(self, x):
        # 定义前向传播过程
        x = self.conv_layers(x)  # 通过卷积层
        x = self.fc_layers(x)  # 通过全连接层
        return x

# 自定义MNIST数据集类
class CustomMNISTDataset(Dataset):
    def __init__(self, images_dir, labels_path, transform=None):
        """
        初始化自定义数据集
        images_dir: 图像文件夹路径
        labels_path: 标签文件路径
        transform: 图像变换操作
        """
        self.images_dir = images_dir  # 图像文件夹路径
        self.labels = self.load_labels(labels_path)  # 加载标签
        self.transform = transform  # 图像变换操作
        self.image_files = sorted(os.listdir(images_dir), key=lambda x: int(os.path.splitext(x)[0]))  # 根据文件名排序图像
        
    def load_labels(self, labels_path):
        # 加载标签文件
        with open(labels_path, 'r') as f:
            labels = [int(line.strip()) for line in f.readlines()]  # 读取每一行作为标签
        return labels
    
    def __len__(self):
        # 返回数据集大小
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 获取指定索引的图像及标签
        image_path = os.path.join(self.images_dir, f"{idx}.jpg")  # 构造图像路径
        image = Image.open(image_path).convert('L')  # 打开图像并转为灰度图
        if self.transform:
            image = self.transform(image)  # 应用变换操作
        label = self.labels[idx]  # 获取标签
        return image, label

def get_device():
    # 获取计算设备（GPU或CPU）
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_images(dataloader, classes, num_images=16):
    # 显示图像网格
    images, labels = next(iter(dataloader))  # 从数据加载器中获取一个批次的图像和标签
    images = images[:num_images]  # 选择指定数量的图像
    labels = labels[:num_images]  # 选择对应的标签
    grid_img = make_grid(images, nrow=4, normalize=True)  # 制作图像网格
    plt.figure(figsize=(8,8))  # 设置图像大小
    plt.imshow(grid_img.permute(1, 2, 0))  # 显示图像网格
    plt.title('Sample Images')  # 设置标题
    plt.axis('off')  # 关闭坐标轴
    plt.show()  # 展示图像

def train(model, dataloader, criterion, optimizer, device, epochs):
    # 训练模型
    model.to(device)  # 将模型转移到指定设备
    model.train()  # 设置模型为训练模式
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)  # 将图像数据转移到设备
            labels = labels.to(device)  # 将标签数据转移到设备
            
            # 前向传播
            outputs = model(images)  # 通过模型获取输出
            loss = criterion(outputs, labels)  # 计算损失
            
            # 反向传播
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新权重
            
            running_loss += loss.item()  # 累加损失
        
        avg_loss = running_loss / len(dataloader)  # 计算平均损失
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")  # 打印损失信息
    print("Training Finished.")  # 训练完成消息

def evaluate(model, dataloader, criterion, device):
    # 评估模型
    model.to(device)  # 将模型转移到指定设备
    model.eval()  # 设置模型为评估模式
    
    total = 0
    correct = 0
    total_loss = 0.0
    
    with torch.no_grad():  # 在评估模式下不需要计算梯度
        for images, labels in dataloader:
            images = images.to(device)  # 将图像数据转移到设备
            labels = labels.to(device)  # 将标签数据转移到设备
            
            outputs = model(images)  # 通过模型获取输出
            loss = criterion(outputs, labels)  # 计算损失
            
            total_loss += loss.item()  # 累加损失
            _, predicted = torch.max(outputs.data, 1)  # 获取预测标签
            total += labels.size(0)  # 统计总样本数
            correct += (predicted == labels).sum().item()  # 统计正确预测的样本数
    
    avg_loss = total_loss / len(dataloader)  # 计算平均损失
    accuracy = 100 * correct / total  # 计算准确率
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")  # 打印测试损失和准确率
    return accuracy

if __name__ == '__main__':
    # 参数设置
    batch_size = 128
    epochs = 10
    learning_rate = 0.001
    data_root = './data/mnist'
    processed_data_root = './data/process'
    
    # 定义图像变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化
    ])
    
    # 获取设备
    device = get_device()
    print(f"Using device: {device}")
    
    # 加载原始MNIST数据集
    train_dataset = torchvision.datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root=data_root, train=False, transform=transform, download=True)
    
    # 加载增强后的训练集
    enhanced_train_dataset = CustomMNISTDataset(
        images_dir=os.path.join(processed_data_root, 'invert_and_rotate'),
        labels_path=os.path.join(processed_data_root, 'labels.txt'),
        transform=transform
    )
    
    # 合并数据集
    combined_train_dataset = ConcatDataset([train_dataset, enhanced_train_dataset])
    
    # 数据加载器
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 显示部分训练图像
    classes = [str(i) for i in range(10)]
    show_images(train_loader, classes)
    
    # 初始化模型、损失函数和优化器
    model = CNN()
    criterion = nn.CrossEntropyLoss()  # 损失函数: 交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 优化器: Adam
    
    # 训练模型
    train(model, train_loader, criterion, optimizer, device, epochs)
    
    # 测试模型
    evaluate(model, test_loader, criterion, device)
    
    # 保存模型
    os.makedirs('./models', exist_ok=True)  # 创建保存模型的文件夹（如果不存在）
    torch.save(model.state_dict(), './models/cnn_mnist.pth')  # 保存模型参数
    print("Model saved successfully.")
