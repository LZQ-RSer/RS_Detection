# -*- coding: utf-8 -*-
# @Time : 2021/2/7 22:27
# @Author : lzq
# @Site : 
# @File : vis_txt.py
# @Software: PyCharm Community Edition 

import numpy as np
import os
from PIL import Image,ImageFont, ImageDraw
Image.MAX_IMAGE_PIXELS = None
import glob
import tqdm
import cv2
import pandas as pd

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
def vis_label(txt,class_name,xz=False):

    """
    可视化标签
    :param txt: 记录标签的txt每一行格式为：图像路径 宽,高 x1,y1,x2,y2，classname
    :param class_name: classname 列表
    :param xz: 旋转框或者水平框
    :return: image
    """
    colors_tableau = [(255, 0, 0), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                      (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                      (148, 103, 189), (197, 176, 213), (140, 0, 75), (196, 156, 148),
                      (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                      (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    line = txt.split(' ')
    image_name = line[0]
    print(image_name)
    ## 从2开始，因为前面有图像路径，图像长宽
    boxs = line[2:]
    thickness = 3
    image = Image.open(image_name)
    for box in boxs:
        # print(box)
        bo = box.split(',')
        b = bo[0:4]
        label = bo[-1]
        # score = bo[9]
        b=[round(float(x)) for x in b]
        p = b
        left = min(b[::2])#奇数位置
        right = max(b[::2])
        top = min(b[1::2])#偶数位置
        bottom = max(b[1::2])
        draw = ImageDraw.Draw(image)
        if xz:
            for i in range(thickness):
                # n =i if i%2 else -i

                draw.polygon([x+i for x in p],outline=colors_tableau[int(class_name.index(label))])
            continue
        top = top - 5
        left = left - 5
        bottom = bottom + 5
        right = right + 5
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors_tableau[int(class_name.index(label))])
        del draw
    return image
if __name__=='__main__':
    #可视化某个真值txt
    class_name = ["aircraft","other"]
    outfile = './test/clip.json.txt'
    img = './test/clip.tif'
    boxs = read_txt(outfile)
    image = vis_label(boxs[0],class_name,False)
    image.show()
    exit()
    image.save('./test/clip_vis.png')
    # exit()