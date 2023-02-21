# 项目流程
- 制作数据集
- 修改配置文件
- 训练数据集
- 测试
# 1.制作数据集文件
- 使用标注工具创建好数据集后， 将**原始图像**放到 **./VOCdevkit/VOC2007/JPEDImages**文件夹下，将**标注xml文件**放到 **./VOCdevkit/VOC2007/Annotations**文件夹下
- 运行VOCdevkit/VOC2007/voc2yolo3.py 文件，生成训练验证测试集的图片名称文件对应的txt文件，放到 ./VOCdevkit/VOC2007/ImageSets/Main 文件夹下
- 运行 voc_annotation.py 文件（注意修改里面sets和classes内容，和ImageSets文件对应），生成对应的txt文件，放到./VOCdevkit/VOC2007/Labels 文件夹下
- 最后Labels中文件夹中txt文件的内容为 
`[img_path xmin,ymin,xmax,ymax,class_id xmin,ymin,xmax,ymax,class_id ...]`
# 2.修改配置文件（此处为YOLO3版本，其他YOLO版本不同）
- 运行convert.py,将Draknet形式的yolo3-tiny模型转换为Keras的h5模型
- 生成先验框
        
    运行kmeans.py文件,生成 model_data/yolo3-tiny_anchors.txt 文件
- 修改类名称
  
    将voc_classes.txt内容修改，注意和voc_annotation.py中的名称顺序一致
# 3.训练数据集
- 修改train.py文件的 annotation_path classes_path anchors_path
- 运行 train.py
# 4.预测
- 修改 yolo.py 文件中的_defaults参数
- 运行 
```shell
python detect.py --image  # 图片检测(运行后该代码后再输入图片路径)
                  --input # 实时摄像头
                  --input=video_path # 视频检测
                  --output=save_path # 结果保存路径 1.mp4[可选]
```
# 5.指定单张图片锚框最多能识别的类别数量
- 训练时，yolo3/utils.py 下 get_random_data()  中 max_boxes=20 修改为其他值
- 检测时，yolo3/model.py 下 yolo_eval()  中 max_boxes=20 修改为其他值