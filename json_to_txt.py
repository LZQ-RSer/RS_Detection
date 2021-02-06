# -*- coding: utf-8 -*-
# @Time : 2021/2/5 22:15
# @Author : lzq
# @Site : 
# @File : json_to_txt.py
# @Software: PyCharm Community Edition 
import numpy as np
import os
import glob
import json

def json_to_txt(json_file,ourfile):
    file = open(outfile,'w')
    with open(json_file) as f:
        lines = json.load(f)
        filename = lines['imagePath']
        width = lines['imageWidth']
        height = lines['imageHeight']
        file.write("%s %s,%s" % (filename, width, height))#注意空格
        lines = lines["shapes"]
        for line in lines:
            list = line['points']
            xmin = int(list[0][0])
            ymin = int(list[0][1])
            xmax = int(list[1][0])
            ymax = int(list[1][1])
            cla = line['label']
            b = (xmin, ymin, xmax, ymax, cla)
            file.write(' ' + ",".join([str(a) for a in b]))
        file.write('\n')
    file.close()

if __name__ == '__main__':
    json_file= './test/clip.json'
    outfile = './test/clip.json.txt'
    json_to_txt(json_file,outfile)