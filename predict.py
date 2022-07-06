import os
import datetime
import math
from osgeo import gdal
import numpy as np
from tensorflow.keras.models import Model
from keras.models import load_model
import pickle
import tensorflow as tf


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

#  进行归一化
def testGenerator(TifArray):
    TifArrayReturn = []
    for i in range(len(TifArray)):
        TifArrayReturn1 = []
        for j in range(len(TifArray[0])):
            img = TifArray[i][j]

            img = img.swapaxes(1, 2)
            img = img.swapaxes(1, 0)
            m = tf.math.l2_normalize(img, dim=1)
            img = m.numpy()
            img = img.swapaxes(1, 0)
            img = img.swapaxes(1, 2)

            TifArrayReturn1.append(img)
        TifArrayReturn.append(TifArrayReturn1)
    return TifArrayReturn

#  获得结果矩阵
def Result(shape, TifArray, npyfile, num_class, RepetitiveLength, RowOver, ColumnOver):
    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0
    for i,item in enumerate(npyfile):
        img=item
        # img = labelVisualize(item)
        # img = img.astype(np.uint8)
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if(i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 256 - RepetitiveLength, 0 : 256-RepetitiveLength] = img[0 : 256 - RepetitiveLength, 0 : 256 - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                #  原来错误的
                #result[shape[0] - ColumnOver : shape[0], 0 : 256 - RepetitiveLength] = img[0 : ColumnOver, 0 : 256 - RepetitiveLength]
                #  后来修改的
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0 : 256 - RepetitiveLength] = img[256 - ColumnOver - RepetitiveLength : 256, 0 : 256 - RepetitiveLength]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       0:256-RepetitiveLength] = img[RepetitiveLength : 256 - RepetitiveLength, 0 : 256 - RepetitiveLength]
        #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif(i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 256 - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0 : 256 - RepetitiveLength, 256 -  RowOver: 256]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0], shape[1] - RowOver : shape[1]] = img[256 - ColumnOver : 256, 256 - RowOver : 256]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       shape[1] - RowOver : shape[1]] = img[RepetitiveLength : 256 - RepetitiveLength, 256 - RowOver : 256]
            #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 256 - RepetitiveLength,
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[0 : 256 - RepetitiveLength, RepetitiveLength : 256 - RepetitiveLength]
            #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0],
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[256 - ColumnOver : 256, RepetitiveLength : 256 - RepetitiveLength]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       ] = img[RepetitiveLength : 256 - RepetitiveLength, RepetitiveLength : 256 - RepetitiveLength]
    return result

#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset

area_perc = 0.5
TifPath = r"G:\sj\cnn-rf\y\f\84.tif"
ResultPath = r"G:\sj\cnn-rf\jg\sd\xin\84.tif"
RFpath = r"G:\sj\cnn-rf\bao\rf_q10_64w_netxx.pickle"

#  记录测试消耗时间
testtime = []
#  获取当前时间
starttime = datetime.datetime.now()

RepetitiveLength = int((1 - math.sqrt(area_perc)) * 256 / 2)
im_width, im_height, im_bands, im_data, im_geotrans, im_proj = readTif(TifPath)
im_data = im_data.swapaxes(1, 0)
im_data = im_data.swapaxes(1, 2)
TifArray, RowOver, ColumnOver = TifCroppingArray(im_data, RepetitiveLength)


model_path = "Model\\q10_net64wx.h5"
model=load_model(model_path)
TifArray = testGenerator(TifArray)
activation_model = Model(inputs=model.input, outputs=model.layers[-2].output)

file = open(RFpath, "rb")
# 把模型从文件中读取出来
rf_model = pickle.load(file)
# 关闭文件
file.close()

model.summary()
results = []

for i in range(len(TifArray)):
        imge=np.array(TifArray[i])
        features = activation_model.predict(imge)
        x=features
        for j in range(len(TifArray[1])):
            preimage = np.array(x[j])
            # data = features.reshape(-1, features.shape[2])
            preimage = preimage.swapaxes(2, 1)
            preimage = preimage.swapaxes(1, 0)
            data = np.zeros((preimage.shape[0], preimage.shape[1] * preimage.shape[2]))
            for i in range(preimage.shape[0]):
                data[i] = preimage[i].flatten()
            data = data.swapaxes(0, 1)
            pred = rf_model.predict(data)
            pred=pred.reshape(preimage.shape[1], preimage.shape[2])
            pred = pred.astype(np.uint8)
            results.append(pred)

endtime = datetime.datetime.now()
text = "模型预测完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"
print(text)
testtime.append(text)

#保存结果
result_shape = (im_data.shape[0], im_data.shape[1])
result_data = Result(result_shape, TifArray, results, 2, RepetitiveLength, RowOver, ColumnOver)
writeTiff(result_data, im_geotrans, im_proj, ResultPath)
endtime = datetime.datetime.now()
text = "结果拼接完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"
print(text)
testtime.append(text)

time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
with open('timelog_%s.txt'%time, 'w') as f:
    for i in range(len(testtime)):
        f.write(testtime[i])
        f.write("\r\n")

