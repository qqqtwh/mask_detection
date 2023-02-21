
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


def _main():
    annotation_path = 'VOCdevkit/VOC2007/Labels/2007_train.txt'
    log_dir = 'logs/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolov3-tiny_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) #输入维度

    #构建模型
    model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolov3-tiny.h5')

    #可视化
    logging = TensorBoard(log_dir=log_dir)
    #检查节点
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    #验证数据拆分比例
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    #首先使用冻结层进行训练，以获得稳定的损失.
    #根据实际情况调整epochs
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # 使用自定义 yolo_loss Lambda 层.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 2
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=10,
                initial_epoch=0,
                callbacks=[ checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    #model.load_weights("logs/ep034-loss6.105-val_loss6.205.h5")
    #解冻继续训练，对模型进行微调.
    #如果效果不好，则进行更长时间的训练.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # 重新编译
        print('解冻所有层.')

        batch_size = 2 # 批次大小
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=16,
            callbacks=[ checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')


def get_classes(classes_path):
    '''加载类别'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''加载注释'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

#构建yolov3-tiny模型
def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,weights_path='model_data/tiny_yolo_weights.h5'):
    '''创建tiny yolov3'''
    K.clear_session() # 获取新的会话
    image_input = Input(shape=(None, None, 3))#指定输入层
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('使用 {} 个锚框 和 {} 个类别创建Tiny YOLOv3 模型'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('加载模型 {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # 除了最后两个输出层之外其他层全部冻结.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('冻结全部{} 层的前{}层.'.format(len(model_body.layers),num))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

#数据生成器
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''数据生成器'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()
