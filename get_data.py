# -*- coding: utf-8 -*-
# @Time : 2021/1/31 11:31
# @Author : lzq
# @Site : 
# @File : get_data.py
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
def voc_to_txt():
    pass
def coco_to_txt():
    pass
def other_to_txt():
    pass

def get_class_number(lines,classes_dict,show=False):
    print('all images numbers:',len(lines))
    c={}
    label = []
    w_h = []
    w_dict={}
    h_dict = {}
    wh_dict = {}
    num_boxs=[]
    i=0
    for line in lines:
        lin = line.split(' ')
        h, w = lin[1].split(',')
        h = int(h)
        w = int(w)
        s=h*w
        w_dict.update({i:w})
        h_dict.update({i:h})
        i+=1
        boxs = lin[2:]
        num_box = len(boxs)
        if num_box>10000 or num_boxs==0:
            print(line)
        num_boxs.append(num_box)

        for box in boxs:
            lin = box.split(',')
            b = lin[0:8]
            b = [round(float(x)) for x in b]
            left = min(b[::2])  # 奇数位置
            right = max(b[::2])
            top = min(b[1::2])  # 偶数位置
            bottom = max(b[1::2])
            class_name=lin[-2]#目标类
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
    box_wh_unique = list(set(w_h))
    box_wh_count = [w_h.count(i) for i in box_wh_unique]
    for i, key in enumerate(box_wh_unique):
        print('宽高比{}: 数量:{}'.format(key, box_wh_count[i]))
    classes_num={}
    for cla in list(classes_dict.values()):
        classes_num.update({cla:label.count(cla)})
    print(classes_num)

    box_unique = list(set(num_boxs))
    box_count = [num_boxs.count(i) for i in box_unique]


    if show:

        x = box_unique
        # x =[str(xx) for xx in x]
        y = box_count
        print(len(x))
        print(len(y))
        plt.bar(range(len(x)),y,width =0.8)
        plt.xticks(range(len(x)),x)
        plt.show()
        exit()

        # print(wh_dict)
        for mb in wh_dict:
            print(mb)
            x = wh_dict[mb]["box_ws"]
            y = wh_dict[mb]["box_hs"]
            plt.scatter(x,y)
            plt.title(mb)
            plt.xlabel("box_ws")
            plt.ylabel('box_hs')
        plt.show()
        # exit()


        # plt.scatter(list(w_dict.values()),list(w_dict.keys()))
        # plt.show()
        plt.scatter(list(h_dict.values()),list(w_dict.values()))
        plt.show()

        # 调节图形大小，宽，高
        plt.figure(figsize=(6, 9))
        # 定义饼状图的标签，标签是列表
        labels = list(classes_num.keys())
        # 每个标签占多大，会自动去算百分比
        sizes =list(classes_num.values())
        # colors = ['red', 'yellowgreen', 'lightskyblue']
        # 将某部分爆炸出来， 使用括号，将第一块分割出来，数值的大小是分割出来的与其他两块的间隙
        # explode = (0.05, 0, 0)

        patches, l_text, p_text = plt.pie(sizes,  labels=labels,
                                          labeldistance=1.1, autopct='%3.1f%%', shadow=False,
                                          startangle=90, pctdistance=0.9)

        # labeldistance，文本的位置离远点有多远，1.1指1.1倍半径的位置
        # autopct，圆里面的文本格式，%3.1f%%表示小数有三位，整数有一位的浮点数
        # shadow，饼是否有阴影
        # startangle，起始角度，0，表示从0开始逆时针转，为第一块。一般选择从90度开始比较好看
        # pctdistance，百分比的text离圆心的距离
        # patches, l_texts, p_texts，为了得到饼图的返回值，p_texts饼图内部文本的，l_texts饼图外label的文本

        # 改变文本的大小
        # 方法是把每一个text遍历。调用set_size方法设置它的属性
        # for t in l_text:
        #     t.set_size = (30)
        # for t in p_text:
        #     t.set_size = (20)
        # 设置x，y轴刻度一致，这样饼图才能是圆的
        plt.axis('equal')
        plt.legend()
        plt.show()
        # exit()

        # Count_df = pd.DataFrame(list(values),index=index)
        # Count_df.plot(kind="bar",y = Count_df.columns)
        # plt.barh(range(len(index)),values,tick_label=index)
        # plt.legend()
        # plt.show()
        # exit()
        wh_df = pd.DataFrame(box_wh_count, index=box_wh_unique, columns=['宽高比数量'])
        wh_df.plot(kind='bar', color="#55aacc")
        plt.show()
    return classes_num


def vis_label(txt,class_name,xz=False):

    """
    可视化标签，旋转框和水平框
    :param txt: 记录标签的txt每一行格式为：图像路径 宽,高 x1,y1,x2,y2,x3,y3,x4,y4,classname,置信度 。。。
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
    boxs = line[2:]
    thickness = 3
    image = Image.open(image_name)
    for box in boxs:
        # print(box)
        bo = box.split(',')
        b = bo[0:8]
        label = bo[8]
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
        # h, w = line[1].split(',')
        # # s = min(int(w),int(h))
        # # font = ImageFont.truetype(font='font/simhei.ttf',
        # #                           size=np.floor(3e-2 * s + 0.5).astype('int32'))
        # font = ImageFont.truetype(font='font/simhei.ttf',
        #                           size=15)
        # print(class_name.index(label))
        # top = max(0, np.floor(top + 0.5).astype('int32'))
        # left = max(0, np.floor(left + 0.5).astype('int32'))
        # bottom = min(s, np.floor(bottom + 0.5).astype('int32'))
        # right = min(s, np.floor(right + 0.5).astype('int32'))

        # label_size = draw.textsize(label, font)
        # # label = label.encode('utf-8')
        # if top - label_size[1] >= 0:
        #     text_origin = np.array([left, top - label_size[1]])
        # else:
        #     text_origin = np.array([left, top + 1])

        # draw.rectangle(
        #     [tuple(text_origin), tuple(text_origin + label_size)],
        #     fill='red')
        # draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        del draw
    return image

def get_our_data():
    print(len(class_name))
    outfile = 'all.txt'
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
        for x in boxs:
            list_file.write(' ')
            x = x.replace(' ', ',')
            list_file.write(x)
        list_file.write('\n')
    list_file.close()


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

    '''
    #（1）得到标准数据
    get_our_data()
    '''

    '''
    #（2）可视化某个真值
    outfile = 'all.txt'
    boxs = read_txt(outfile)
    print(boxs[0])
    image = vis_label(boxs[980],class_name,True)
    # image.show()
    image.save('sfds.png')
    exit()
    '''

    #（3）统计
    print("all class is",len(class_name))
    class_name_dict = dict(zip(range(len(class_name)), class_name))
    outfile = 'all.txt'
    boxs = read_txt(outfile)
    get_class_number(boxs,class_name_dict,show=True)





