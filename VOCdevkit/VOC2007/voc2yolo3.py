#-------------------------------------------------
#   xml文件名称就是图片名称
#   读取xml文件名称，生成写有图片名称的训练验证测试txt文件
#-------------------------------------------------

import os
import random 
 
xmlfilepath=r"./Annotations"
saveBasePath=r"./ImageSets/Main/"
 
trainval_percent=1
train_percent=0.9
total_xml = os.listdir(xmlfilepath)


num=len(total_xml)  # 图片xml文件总数量
list=range(num)
tv=int(num*trainval_percent)    # 所有图片当做训练验证集
tr=int(tv*train_percent)        # 取训练验证集的0.9当做训练集

trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)  
 
print("train and val size",tv)
print("traub suze",tr)
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
 
for i  in list:  
    name=total_xml[i][:-4]+'\n'  
    if i in trainval:  
        ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        else:  
            fval.write(name)  
    else:  
        ftest.write(name)  
  
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()
