import xml.etree.ElementTree as ET
import os

# 定义转换函数，将边界框坐标转换为YOLO格式
def convert(size, box):
    # 计算边界框中心点的x和y坐标
    x_center = (box[2] + box[0]) / 2.0
    y_center = (box[3] + box[1]) / 2.0
    # 将中心点坐标归一化
    x = x_center / size[0]
    y = y_center / size[1]
    # 计算边界框的宽度和高度，并进行归一化
    w = (box[2] - box[0]) / size[0]
    h = (box[3] - box[1]) / size[1]
    return (x, y, w, h)

# 定义转换XML为YOLO格式文本的函数
def convert_annotation(xml_paths, yolo_paths, classes):
    # 获取xml文件夹中所有xml文件
    xml_files = os.listdir(xml_paths)
    for file in xml_files:
        # 设置xml文件和将要生成的yolo文本文件的路径
        xml_file_path = os.path.join(xml_paths, file)
        yolo_txt_path = os.path.join(yolo_paths, file.split(".")[0] + ".txt")
        # 解析xml文件
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        # 获取图像的宽度和高度
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        # 打开yolo文本文件，准备写入
        with open(yolo_txt_path, 'w') as f:
            # 遍历xml文件中的所有object标签
            for obj in root.iter("object"):
                difficult = obj.find("difficult").text
                # 获取类别名称
                cls = obj.find("name").text
                # 如果类别不在训练类别列表中，或者标注为难以检测，则跳过
                if cls not in classes or difficult == '1':
                    continue
                # 获取类别的索引
                cls_id = classes.index(cls)
                # 获取边界框坐标
                xml_box = obj.find("bndbox")
                box = (float(xml_box.find("xmin").text), float(xml_box.find("ymin").text),
                       float(xml_box.find("xmax").text), float(xml_box.find("ymax").text))
                # 将边界框坐标转换为YOLO格式
                boxex = convert((w, h), box)
                # 将转换后的坐标写入yolo文本文件
                f.write(str(cls_id) + " " + " ".join([str(s) for s in boxex]) + '\n')
        print(f'xml_file :{file} --> txt Saved!')

if __name__ == "__main__":
    # 定义训练中使用的数据类别
    classes_train = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
    
    # 定义xml文件的存储路径
    xml_dir = "Annotations"
    
    # 定义yolo格式文本文件的存储路径
    yolo_txt_dir = "Label"
    
    # 进行格式转换
    convert_annotation(xml_paths=xml_dir, yolo_paths=yolo_txt_dir, classes=classes_train)
    