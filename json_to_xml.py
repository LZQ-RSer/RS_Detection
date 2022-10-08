# -*- coding: utf-8 -*-

'''
将json文件转为类似voc中的xml格式
'''

import os
import numpy as np
import codecs
import json
from glob import glob
import cv2
import shutil
# from sklearn.model_selection import train_test_split

# 1.原始labelme标注数据路径
labelme_path = "D:/MAP/work/HHQ/pic/"
# 保存路径
saved_path = "D:/MAP/work/HHQ/pic_xml/"

isUseTest=False#是否创建test集
# 2.创建要求文件夹
# if not os.path.exists(saved_path + "Annotations"):
#     os.makedirs(saved_path + "Annotations")
# if not os.path.exists(saved_path + "JPEGImages/"):
#     os.makedirs(saved_path + "JPEGImages/")
# if not os.path.exists(saved_path + "ImageSets/Main/"):
#     os.makedirs(saved_path + "ImageSets/Main/")
# 3.获取待处理文件
files = glob(labelme_path + "*.json")
## windows路径
files = [i.replace("\\","/").split("/")[-1].split(".json")[0] for i in files]
print(files)
# exit()
# 4.读取标注信息并写入 xml
for json_file_ in files:
    json_filename = labelme_path + json_file_ + ".json"
    json_file = json.load(open(json_filename, "r"))
    # height, width, channels = cv2.imread('ECM/Images/' + json_file_ + ".jpg").shape
    height, width, channels = cv2.imread(json_filename.replace('.json','.jpg')).shape
    with codecs.open(saved_path + json_file_ + ".xml", "w", "utf-8") as xml:

        xml.write('<annotation>\n')
        xml.write('\t<folder>' + saved_path + '</folder>\n')
        xml.write('\t<filename>' + json_file_ + ".jpg" + '</filename>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>ECM_Data</database>\n')
        xml.write('\t\t<annotation>ECM</annotation>\n')
        xml.write('\t\t<image>flickr</image>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t</source>\n')
        xml.write('\t<owner>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t\t<name>XT</name>\n')
        xml.write('\t</owner>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(width) + '</width>\n')
        xml.write('\t\t<height>' + str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        for multi in json_file["shapes"]:
            points = np.array(multi["points"])
            labelName=multi["label"]
            xmin = min(points[:, 0])
            xmax = max(points[:, 0])
            ymin = min(points[:, 1])
            ymax = max(points[:, 1])
            label = multi["label"]
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t<object>\n')
                xml.write('\t\t<name>' + labelName+ '</name>\n')
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>1</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(int(xmin)) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(int(ymin)) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(int(xmax)) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(int(ymax)) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
                print(json_filename, xmin, ymin, xmax, ymax, label)
        xml.write('</annotation>')


exit()
# 5.复制图片到 VOC2007/JPEGImages/下

# 自己的图片路径
# image_files = glob("ECM/Images/" + "*.jpg")
# print("copy image files to VOC007/JPEGImages/")
# for image in image_files:
#     shutil.copy(image, saved_path + "JPEGImages/")
# # 6.split files for txt
# txtsavepath = saved_path + "ImageSets/Main/"
# ftrainval = open(txtsavepath + '/trainval.txt', 'w')
# ftest = open(txtsavepath + '/test.txt', 'w')
# ftrain = open(txtsavepath + '/train.txt', 'w')
# fval = open(txtsavepath + '/val.txt', 'w')
# total_files = glob("./VOC2007/Annotations/*.xml")
# total_files = [i.replace("\\","/").split("/")[-1].split(".xml")[0] for i in total_files]
# trainval_files=[]
# test_files=[]
# if isUseTest:
#     trainval_files, test_files = train_test_split(total_files, test_size=0.2, random_state=42)
# else:
#     trainval_files=total_files
# for file in trainval_files:
#     ftrainval.write(file + "\n")
# # split
# train_files, val_files = train_test_split(trainval_files, test_size=0.15, random_state=55)
# # train
# for file in train_files:
#     ftrain.write(file + "\n")
# # val
# for file in val_files:
#     fval.write(file + "\n")
# for file in test_files:
#     print(file)
#     ftest.write(file + "\n")
# ftrainval.close()
# ftrain.close()
# fval.close()
# ftest.close()
