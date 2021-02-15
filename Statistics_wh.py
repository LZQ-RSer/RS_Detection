# -*- coding: utf-8 -*-
# @Time : 2021/2/7 22:48
# @Author : lzq
# @Site : 
# @File : Statistics.py
# @Software: PyCharm Community Edition 
import numpy as np
import os
from PIL import Image,ImageFont, ImageDraw
Image.MAX_IMAGE_PIXELS = None
import glob
import tqdm
import cv2
import matplotlib.pyplot as plt
import pandas as pd

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
            #消除换行
            if '\n' in line:
                line = line.strip("\n")
            #消除不需要的行
            if line[0] in ["i","g"]:
                line = f.readline()
                continue
            list.append(line)
            line = f.readline()
    return list

def get_image_wh(lines):
    print('all images numbers:',len(lines))
    #宽高数据
    w_dict={}
    h_dict = {}
    i=0
    for line in lines:
        lin = line.split(' ')
        h, w = lin[1].split(',')
        h = int(h)
        w = int(w)
        w_dict.update({i:w})
        h_dict.update({i:h})
        i+=1
    ##(1)绘制图像长宽分布
    plt.scatter(list(w_dict.values()), list(h_dict.values()))
    plt.title("image w h")
    plt.xlabel("w")
    plt.ylabel('h')
    plt.show()


if __name__ == '__main__':
    ############################################
    #类别
    ############################################
    """
    大型车辆(large vehicle)、游泳池(swimming pool)、直升机(helicopter)、桥梁(bridge)、飞机(plane)、船舶(ship)、足球场(soccer ball field)、篮球场(basketball court)、机场(airport)、
    集装箱起重机(container-crane)、田径场(ground track field)、小汽车(small vehicle)、码头(harbor)、棒球场(baseball diamond)、网球场(tennis court)、转盘(roundabout)、储存罐(storage tank)、直升机场(helipad)

    """
    class_name = ["large-vehicle", "swimming-pool", "helicopter", "bridge", "plane", "ship", "soccer-ball-field",
                  "basketball-court", "airport","container-crane", "ground-track-field", "small-vehicle", "harbor",
                  "baseball-diamond", "tennis-court","roundabout", "storage-tank", "helipad"]
    print("all class is:",len(class_name))
    class_name_dict = dict(zip(range(len(class_name)), class_name))
    outfile = './test/all.txt'
    boxs = read_txt(outfile)
    get_image_wh(boxs)