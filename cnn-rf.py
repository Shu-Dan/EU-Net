import os
import datetime
from osgeo import gdal
import numpy as np
import cv2
from tensorflow.keras.models import Model
from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import pickle
import sklearn.metrics as sm
import tensorflow as tf

def readTif1(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
    return GdalImg_data

def readTif(fileName, xoff = 0, yoff = 0, data_width = 0, data_height = 0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  波段数
    bands = dataset.RasterCount
    #  获取数据
    if(data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj

def TifCroppingArray(img, SideLength):
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目
    ColumnNum = int((img.shape[0] - SideLength * 2) / (256 - SideLength * 2))
    #  行上图像块数目
    RowNum = int((img.shape[1] - SideLength * 2) / (256 - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (256 - SideLength * 2) : i * (256 - SideLength * 2) + 256,
                          j * (256 - SideLength * 2) : j * (256 - SideLength * 2) + 256]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[i * (256 - SideLength * 2) : i * (256 - SideLength * 2) + 256,
                      (img.shape[1] - 256) : img.shape[1]]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[(img.shape[0] - 256) : img.shape[0],
                      j * (256-SideLength*2) : j * (256 - SideLength * 2) + 256]
        TifArray.append(cropped)
    #  向前裁剪右下角
    cropped = img[(img.shape[0] - 256) : img.shape[0],
                  (img.shape[1] - 256) : img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[0] - SideLength * 2) % (256 - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[1] - SideLength * 2) % (256 - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver

def Generator(train_image_path):
    TifArrayReturn = []
    imageList = ['1.tif','2.tif','3.tif','4.tif','5.tif','6.tif']
    for i in imageList:
        img = readTif1(train_image_path + "\\" + i)
        #  GDAL读数据是(BandNum,Width,Height)要转换为->(Width,Height,BandNum)
        m = tf.math.l2_normalize(img, dim=1)

        img = m.numpy()
        img = img.swapaxes(1, 0)
        img = img.swapaxes(1, 2)
        # img=img / 255.0
        TifArrayReturn.append(img)
    return TifArrayReturn

def Generator1(train_image_path):
    imageList = ['1.tif', '2.tif', '3.tif', '4.tif', '5.tif', '6.tif']
    TifArrayReturn = []
    for i in imageList:
        img = readTif1(train_image_path + "\\" + i).astype(np.uint8)
        if (len(img.shape) == 3):
            img = img.swapaxes(1, 0)
            img = img.swapaxes(1, 2)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        TifArrayReturn.append(img)
    return TifArrayReturn

area_perc = 0.5
SavePath = r"G:\sj\cnn-rf\bao\rf_q10_64w_netxx.pickle"
train_image_path = r"G:\sj\cnn-rf\a\JPEGImages11"
train_label_path= r"G:\sj\cnn-rf\a\SegmentationClass11"

#  记录测试消耗时间
testtime = []
#  获取当前时间
starttime = datetime.datetime.now()

TifArray1=Generator(train_image_path)
laArray1=Generator1(train_label_path)

model_path = "Model\\q10_net64wx.h5"
model=load_model(model_path)
activation_model = Model(inputs=model.input, outputs=model.layers[-2].output)
model.summary()


imge=np.array(TifArray1)
lable=np.array(laArray1)
features = activation_model.predict(imge)
# features=imge
x=features.reshape(-1,features.shape[3])
y=lable.reshape(-1)
train_data, test_data, train_label, test_label = model_selection.train_test_split(x, y, random_state=1,
                                                                                  train_size=0.7, test_size=0.3)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(train_data, train_label)
print("训练集：", classifier.score(train_data, train_label))
print("测试集：", classifier.score(test_data, test_label))

pred_test_y = classifier.predict(test_data)
cm = sm.confusion_matrix(test_label, pred_test_y)
print(cm)
cr = sm.classification_report(test_label, pred_test_y)
print(cr)

file = open(SavePath, "wb")
#将模型写入文件：
pickle.dump(classifier, file)
#最后关闭文件：
file.close()

endtime = datetime.datetime.now()
text = "结果耗时间: " + str((endtime - starttime).seconds) + "s"
print(text)
testtime.append(text)
time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
