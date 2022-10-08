import argparse
import glob
import os
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm

def parse_args():
    """
        参数配置
    """
    parser = argparse.ArgumentParser(description='xml2json')
    parser.add_argument('--raw_label_dir', help='the path of raw label', default=r'D:\data\昆虫\insects_reinforce2\train\xmls')
    parser.add_argument('--pic_dir', help='the path of picture', default=r'D:\data\昆虫\insects_reinforce2\train\images')
    parser.add_argument('--save_dir', help='the path of new label', default=r'D:\data\昆虫\insects_reinforce2\train\jsons')
    args = parser.parse_args()
    return args

def read_xml_gtbox_and_label(xml_path):
    """
        读取xml内容
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = int(size.find('depth').text)
    points = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        pose = obj.find('pose').text
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymin = float(xmlbox.find('ymin').text)
        ymax = float(xmlbox.find('ymax').text)
        box = [xmin, ymin, xmax, ymax]
        point = [cls, box]
        points.append(point)
    return points, width, height

def main():
    """
        主函数
    """
    args = parse_args()
    labels = glob.glob(args.raw_label_dir + '/*.xml')
    for i, label_abs in tqdm(enumerate(labels), total=len(labels)):
        _, label = os.path.split(label_abs)
        label_name = label.rstrip('.xml')
        img_path = os.path.join(args.pic_dir, label_name + '.jpg')
        points, width, height = read_xml_gtbox_and_label(label_abs)
        json_str = {}
        json_str['version'] = '4.5.6'
        json_str['flags'] = {}
        shapes = []
        for i in range(len(points)):
            shape = {}
            shape['label'] = points[i][0]
            shape['points'] = [[points[i][1][0], points[i][1][1]],
                                [points[i][1][0], points[i][1][3]],
                                [points[i][1][2], points[i][1][3]],
                                [points[i][1][2], points[i][1][1]]]
            shape['group_id'] = None
            shape['shape_type'] = 'polygon'
            shape['flags'] = {}
            shapes.append(shape)
        json_str['shapes'] = shapes
        json_str['imagePath'] = img_path
        json_str['imageData'] = None
        json_str['imageHeight'] = height
        json_str['imageWidth'] = width
        with open(os.path.join(args.save_dir, label_name + '.json'), 'w') as f:
            json.dump(json_str, f, indent=2)

if __name__ == '__main__':
    main()
