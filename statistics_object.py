# -*- coding: utf-8 -*-
# @Time : 2021/2/13 17:14
# @Author : lzq
# @Site : 
# @File : statistics_object.py
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

def get_object_number(lines,classes_dict,show=False):
    print('all images numbers:',len(lines))
    ##目标类名集合
    label = []
    ##长宽比集合
    w_h = []
    ##每类目标长宽集合
    wh_dict = {}
    num_boxs=[]
    for line in lines:
        lin = line.split(' ')
        boxs = lin[2:]
        num_box = len(boxs)
        num_boxs.append(num_box)

        for box in boxs:
            # print(box)
            lin = box.split(',')
            b = lin[0:4]
            b = [round(float(x)) for x in b]
            # print(b)
            # left = min(b[::2])  # 奇数位置
            left = b[0]
            right = max(b[::2])
            right = b[2]
            # top = min(b[1::2])  # 偶数位置
            top = b[1]
            # bottom = max(b[1::2])
            bottom = b[3]
            class_name=lin[-1]#目标类
            label.append(class_name)
            box_w = right-left
            box_h = bottom-top
            if class_name not in wh_dict:
                box_ws = [box_w]
                box_hs =[box_h]
            else:
                box_ws = wh_dict[class_name]["box_ws"]
                box_ws.append(box_w)
                box_hs = wh_dict[class_name]["box_hs"]
                box_hs.append(box_h)
            wh_dict.update({class_name:{"box_ws":box_ws,"box_hs":box_hs}})
            wh=round((int(right)-int(left))/(int(bottom)-int(top)),0)
            if wh<1:
                wh = round((int(bottom) - int(top)) / (int(right) - int(left)), 0)
            w_h+=[wh]
    print('all object is :',len(label))

    ###宽高比统计
    box_wh_unique = list(set(w_h))
    box_wh_count = [w_h.count(i) for i in box_wh_unique]
    for i, key in enumerate(box_wh_unique):
        print('宽高比{}: 数量:{}'.format(key, box_wh_count[i]))
    ###每一个目标个数统计
    classes_num={}
    for cla in list(classes_dict.values()):
        classes_num.update({cla:label.count(cla)})
    print(classes_num)
    ###每一张图像中的目标个数统计,{目标数：图像个数}
    image_object_num={}
    box_unique = list(set(num_boxs))
    box_count = [num_boxs.count(i) for i in box_unique]
    for i,ob_num in enumerate(box_unique):
        image_object_num.update({ob_num:box_count[i]})
    print(image_object_num)
    print(len(image_object_num))
    # exit()
    if show:
        ###每一个目标个数统计饼状图
        x = list(classes_num.keys())
        y = list(classes_num.values())
        plt.bar(range(len(y)), y,tick_label=x)
        plt.show()

        plt.pie(x = y,labels = x, autopct="%0.2f%%")
        plt.legend()
        plt.show()

        x = list(image_object_num.keys())#[0:20]
        y = list(image_object_num.values())#[0:20]
        ###图像中目标个数柱状图
        plt.bar(range(len(x)), x,fc = 'y')
        plt.show()
        ###目标个数对应的图像数
        plt.bar(range(len(y)), y,fc = 'r')
        plt.show()

        ###目标数：图像数
        plt.pie(x = y,labels = x, autopct="%0.2f%%")
        plt.legend()
        plt.show()

        # print(wh_dict)
        xx = []
        yy =[]
        for mb in wh_dict:
            print(mb)
            x = wh_dict[mb]["box_ws"]
            y = wh_dict[mb]["box_hs"]
            # print(x,y)
            xx.extend(x)
            yy.extend(y)
            plt.scatter(x,y)
            plt.title(mb)
            plt.xlabel("box_ws")
            plt.ylabel('box_hs')
            plt.show()
        ###总的目标宽高散点图
        plt.scatter(xx, yy)
        plt.title("all object")
        plt.xlabel("box_ws")
        plt.ylabel('box_hs')
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
    # outfile = './test/rscup/all2.txt'
    outfile = './test/all.txt'
    boxs = read_txt(outfile)
    get_object_number(boxs,class_name_dict,show=True)