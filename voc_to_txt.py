# -*- coding: utf-8 -*-
# @Time : 2021/2/4 23:03
# @Author : lzq
# @Site : 
# @File : voc_to_txt.py
# @Software: PyCharm Community Edition 
import sys
import os
import glob
import xml.etree.ElementTree as ET
def voc_to_txt(xml,outfile):
    with open(outfile, "w") as new_f:
        root = ET.parse(xml).getroot()
        filename = root.find('filename').text
        size = root.find('size')
        width = size.find('width').text
        height = size.find('height').text
        new_f.write("%s %s,%s "%(filename,width,height))
        for obj in root.findall('object'):
            if obj.find('difficult')!=None:
                difficult = obj.find('difficult').text
                if int(difficult)==1:
                    continue
            obj_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text
            new_f.write("%s,%s,%s,%s,%s " % (left, top, right, bottom,obj_name))
        new_f.write('\n')
if __name__ == '__main__':
    xml = "./test/aircraft_79.xml"
    outfile = './test/aircraft_79.txt'
    voc_to_txt(xml,outfile)
