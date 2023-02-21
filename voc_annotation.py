# ----------------------------------------------------------------------
# 根据 voc2yolo3.py 生成的训练验证图片id文件，
# 生成对应的 图片路径 xmin,ymin,xmax,ymax,class_id 形式txt文件
# 供kmeans.py文件生成anchors使用
# ----------------------------------------------------------------------

import xml.etree.ElementTree as ET
from os import getcwd

# sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets=[('2007', 'train'), ('2007', 'val')]

classes = ["with_mask", "without_mask"]


def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text  # 类别序号
        cls = obj.find('name').text             # 类别名称
        # 不是本数据集要求的类型或者难以识别的区域就跳过
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')             # 真实框左上角和右下角坐标
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()
print(wd)
for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('VOCdevkit/VOC%s/Labels/%s_%s.txt'%(year,year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.png'%(wd, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()

