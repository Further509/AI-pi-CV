# 手写数字识别

目前能在自己的数据集的测试集达到95%以上的准确率

但对自己手写数字的识别准确率不高。

**代码说明**

`convert.py`为对原mnist数据集进行的数据增强

`train.py`为训练代码，可在训练集达到99.05%的准确率

`detect.py`为测试代码

`./models/cnn_mnist.pth`为保存好的模型文件

可实现命令行一行测试

```bash
python detect.py path_to_your_image.jpg
```

