# -*- coding: utf-8 -*-
# @Time : 2021/2/13 16:36
# @Author : lzq
# @Site : 
# @File : convert_txt.py
# @Software: PyCharm Community Edition
from PIL import Image, ImageFont, ImageDraw
Image.MAX_IMAGE_PIXELS = None
import glob
import matplotlib.pyplot as plt
def read_txt(str_text):
    """
    自定义解析方法
    :param str_text: 
    :return: 
    """
    list = []
    with open(str_text) as f:
        line = f.readline()
        while line:
            # 消除空行
            if line.isspace():
                line = f.readline()
                continue
            # 消除换行
            if '\n' in line:
                line = line.strip("\n")
            # 消除不需要的行
            if line[0] in ["i", "g"]:
                line = f.readline()
                continue
            list.append(line)
            line = f.readline()
    return list
def get_our_data():
    outfile = './test/all.txt'
    list_file = open(outfile, 'w')
    labels = glob.glob('D:/study/data/rssrai2019_object_detection/train/labelTxt/labelTxt/*.txt')
    for label in labels:
        print(label)
        # label = r"D:\study\data\rssrai2019_object_detection\train\labelTxt\labelTxt\P0002.txt"
        image = label.replace('labelTxt', 'images').replace('.txt', '.png')
        # 获取图像宽高,并写入txt
        w, h = Image.open(image).size
        list_file.write(image)
        list_file.write(" ")
        list_file.write(str(w))
        list_file.write(',')
        list_file.write(str(h))
        boxs = read_txt(label)
        for box in boxs:
            lin = box.split(' ')
            b = lin[0:-2]
            b = [round(float(x)) for x in b]
            left = min(b[::2])  # 奇数位置
            right = max(b[::2])
            top = min(b[1::2])  # 偶数位置
            bottom = max(b[1::2])
            class_name = lin[-2]  # 目标类
            object = (left, top, right, bottom, class_name)
            list_file.write(' ' + ",".join([str(a) for a in object]))
        list_file.write('\n')
    list_file.close()


if __name__ == '__main__':
    ############################################
    # 类别
    ############################################
    """
    大型车辆(large vehicle)、游泳池(swimming pool)、直升机(helicopter)、桥梁(bridge)、飞机(plane)、船舶(ship)、足球场(soccer ball field)、篮球场(basketball court)、机场(airport)、
    集装箱起重机(container-crane)、田径场(ground track field)、小汽车(small vehicle)、码头(harbor)、棒球场(baseball diamond)、网球场(tennis court)、转盘(roundabout)、储存罐(storage tank)、直升机场(helipad)

    """
    class_name = ["large-vehicle", "swimming-pool", "helicopter", "bridge", "plane", "ship", "soccer-ball-field",
                  "basketball-court", "airport", "container-crane", "ground-track-field", "small-vehicle", "harbor",
                  "baseball-diamond", "tennis-court", "roundabout", "storage-tank", "helipad"]

    # （1）得到标准数据
    get_our_data()