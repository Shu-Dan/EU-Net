import os
import datetime
from osgeo import gdal
import numpy as np
import cv2
from tensorflow.keras.models import Model
from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import pickle
import sklearn.metrics as sm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from mlxtend.classifier import StackingClassifier


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

def Generator(train_image_path):
    TifArrayReturn = []
    imageList = ['1.tif','2.tif','3.tif','4.tif','5.tif','6.tif','7.tif','8.tif']
    for i in imageList:
        img = readTif1(train_image_path + "\\" + i)
        #  GDAL读数据是(BandNum,Width,Height)要转换为->(Width,Height,BandNum)
        img = img.swapaxes(1, 0)
        img = img.swapaxes(1, 2)
        img=img / 255.0
        TifArrayReturn.append(img)
    return TifArrayReturn

def Generator1(train_image_path):
    imageList = ['1.tif', '2.tif', '3.tif', '4.tif', '5.tif', '6.tif', '7.tif', '8.tif']
    TifArrayReturn = []
    for i in imageList:
        img = readTif1(train_image_path + "\\" + i).astype(np.uint8)
        if (len(img.shape) == 3):
            img = img.swapaxes(1, 0)
            img = img.swapaxes(1, 2)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #  GDAL读数据是(BandNum,Width,Height)要转换为->(Width,Height,BandNum)
        TifArrayReturn.append(img)
    return TifArrayReturn

area_perc = 0.5
SavePath = r"G:\sj\cnn-rf\bao\stack_RLS_sgnetq10_8w.pickle"
train_image_path = r"G:\sj\cnn-rf\a\JPEGImages"
train_label_path= r"G:\sj\cnn-rf\a\SegmentationClass"

#  记录测试消耗时间
testtime = []
#  获取当前时间
starttime = datetime.datetime.now()

TifArray1=Generator(train_image_path)
laArray1=Generator1(train_label_path)

#获取特征
# model_path = "Model\\cnn.h5"
model_path = "Model\\q10_sgnet.hdf5"
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

clf1 = KNeighborsClassifier(n_neighbors=1)
clf4 = LGBMClassifier(num_leaves=63)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = SVC(C=1.0, kernel='rbf',degree=6, cache_size=1024,probability=True)
lr = RandomForestClassifier(n_estimators=100, random_state=42)

sclf = StackingClassifier(classifiers=[ clf2, clf3,clf4],
                          use_probas=True,
                          average_probas=False,
                          meta_classifier=lr)

print('3-fold cross validation:\n')

for clf, label in zip([ clf2,clf3, clf4, sclf],
                      [
                       'Random Forest',
                       'svm',
                       'LGB',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, train_data, train_label,
                                              cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))

sclf.fit(train_data, train_label)
pred_test_y = sclf.predict(test_data)
print("准确率",accuracy_score(test_label, pred_test_y))
cm = sm.confusion_matrix(test_label, pred_test_y)
print(cm)
cr = sm.classification_report(test_label, pred_test_y)
print(cr)

file = open(SavePath, "wb")
#将模型写入文件：
pickle.dump(sclf, file)
#最后关闭文件：
file.close()

endtime = datetime.datetime.now()
text = "结果耗时间: " + str((endtime - starttime).seconds) + "s"
print(text)
testtime.append(text)
time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
