import argparse
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt

# 定义卷积神经网络 (CNN) 类
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义卷积层
        self.conv_layers = nn.Sequential(
            # 卷积层1: 输入1通道，输出16通道，卷积核大小3x3，步幅1，填充1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # 激活函数 ReLU
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层: 池化窗口大小2x2，步幅2
            
            # 卷积层2: 输入16通道，输出32通道，卷积核大小3x3，步幅1，填充1
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # 激活函数 ReLU
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层: 池化窗口大小2x2，步幅2
            
            # 卷积层3: 输入32通道，输出64通道，卷积核大小3x3，步幅1，填充1
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # 激活函数 ReLU
        )
        # 定义全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 展平层: 将多维输入展平成一维
            nn.Linear(in_features=7 * 7 * 64, out_features=128),  # 全连接层1: 输入维度7x7x64，输出维度128
            nn.ReLU(),  # 激活函数 ReLU
            nn.Linear(in_features=128, out_features=10),  # 全连接层2: 输入维度128，输出维度10（分类数）
        )

    def forward(self, input):
        # 定义前向传播过程
        output = self.conv_layers(input)  # 通过卷积层
        output = self.fc_layers(output)  # 通过全连接层
        return output

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Predict digits using a trained CNN model.')
    parser.add_argument('image_path', type=str, help='Path to the input image.')  # 添加图像路径参数
    return parser.parse_args()

def main():
    args = parse_args()  # 解析命令行参数
    
    # 加载训练好的模型
    model = CNN()  # 实例化模型
    model.load_state_dict(torch.load('./models/cnn_mnist.pth'))  # 加载保存的模型权重
    model.eval()  # 设置模型为评估模式

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.1307], std=[0.3081])  # 标准化图像
    ])

    # 加载和预处理图像
    img = Image.open(args.image_path).convert('L')  # 打开图像并转换为灰度图
    img_trans = transform(img).unsqueeze(0)  # 应用预处理并增加一个批次维度

    # 预测
    with torch.no_grad():  # 在预测时不计算梯度
        output = model(img_trans)  # 通过模型获取输出
        pred = torch.argmax(output, dim=1)  # 获取预测的类别（最大概率对应的索引）

    # 显示图像和预测结果
    plt.imshow(img, cmap='gray', interpolation='none')  # 显示图像
    plt.title(f"Prediction: {pred.item()}")  # 显示预测结果
    plt.xticks([])  # 隐藏x轴刻度
    plt.yticks([])  # 隐藏y轴刻度
    plt.show()  # 展示图像

if __name__ == '__main__':
    main()  
