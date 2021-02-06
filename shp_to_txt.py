# -*- coding: utf-8 -*-
# @Time : 2021/2/5 22:41
# @Author : lzq
# @Site : 
# @File : shp_to_txt.py
# @Software: PyCharm Community Edition 
from osgeo import gdal, ogr, osr
import numpy as np
import os
def read_tif(filename):
    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    # im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    # im_data = dataset.ReadAsArray(buf_xsize=int(im_width/size),buf_ysize=int(im_height/size))
    del dataset
    return im_proj, im_geotrans, im_width,im_height
def write_tif(filename,im_proj,im_geotrans,im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        # datatype = gdal.GDT_UInt16
        datatype = gdal.GDT_Byte
    else:
        # datatype = gdal.GDT_Float32
        datatype = gdal.GDT_Byte

    if len(im_data.shape)==3:
        im_bands,im_height,im_width = im_data.shape
    else:
        im_bands,(im_height,im_width) = 1,im_data.shape
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(filename,im_width,im_height,im_bands,datatype)
    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset

def shp_to_txt(img, shapefile, txt):

    file = open(txt, 'w')
    prj,geo,w,h = read_tif(img)
    file.write("%s %s,%s" % (img, w, h))  # 注意空格
    if prj=="":
        zx=-1
    else:
        zx=1
    left=geo[0]
    up = geo[3]
    pixx = geo[1]
    pixy = geo[5]*zx
    vector = ogr.Open(shapefile)
    layer = vector.GetLayer()
    n = layer.GetFeatureCount()
    lists = []
    for i in range(n):
        feat = layer.GetFeature(i)
        poly=feat.GetGeometryRef()
        box = poly.GetEnvelope()
        x1 = int((box[0]-left)//pixx)
        y2 = int((box[2]-up)//pixy)
        x2 = int((box[1]-left)//pixx)
        y1 = int((box[3]-up)//pixy)
        feature = layer.GetFeature(i)
        name = feature.GetField("classes")
        # print(name)
        object = (x1,y1,x2,y2,name)
        print(object)
        file.write(' ' + ",".join([str(a) for a in object]))
        lists.append([x1,y1,x2,y2,name])
    ##何时换行
    file.write('\n')
    file.close()
    return lists
def read_tif_size(filename):
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    dataset = gdal.Open(filename)
    # dataset = gdal.AutoCreateWarpedVRT(dataset, None, srs.ExportToWkt(),gdal.GRA_NearestNeighbour)# , gdal.GRA_Bilinear)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    # print(im_width,im_height)
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    # print(im_width,im_height)
    # print(im_geotrans)
    return im_width,im_height,im_geotrans,im_proj,dataset

def clip_image(image,lists, outdir,size):
    name = os.path.split(image)[1][:-4]
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    label2 = lists
    h, w, geo, prj, dataset = read_tif_size(image)
    list_file = open(os.path.join(outdir,'all.txt'), 'w')

    i=0
    for box in label2:
        outdirfile = os.path.join(outdir, name + str(i) + '.tif').replace('\\','/')
        list_file.write("%s %s,%s" % (outdirfile, size, size))#注意空格
        x1, y1,x2,y2,_ = box
        if True:
            bx = np.random.randint(max(x2 - size,0), min(x1,h-size))
            by = np.random.randint(max(y2 - size,0), min(y1,w-size))
            bbox = [bx, by, bx + size, by + size]
            def get_label(labels,bbox):
                def do(lab,bbox):
                    pix_label = []
                    def do2(x):
                        if x < 0:
                            return 0
                        elif x > size:
                            return size
                        else:
                            return x
                    x1,y1,x2,y2,_ = lab
                     # = lab[1]
                    xx1 = x1-bbox[0]
                    yy1 = y1-bbox[1]
                    xx2 = x2-bbox[0]
                    yy2 = y2-bbox[1]
                    pix_label.append(do2(xx1))
                    pix_label.append(do2(yy1))
                    pix_label.append(do2(xx2))
                    pix_label.append(do2(yy2))
                    pix_label.append(box[-1])
                    # pix_label.append([do2(xx1),do2(yy1),do2(xx2),do2(yy2)])
                    return pix_label
                center = [[(label[0] + label[2]) // 2, (label[1] + label[3]) // 2] for label in labels]

                label =[]
                for cen in center:
                    if bbox[0]<=cen[0]<=bbox[2] and bbox[1]<=cen[1]<=bbox[3]:
                        lab=labels[center.index(cen)]
                        lab=do(lab,bbox)
                        label.append(lab)
                return np.array(label)
            boxs = get_label(label2, bbox)
            # list_file = open(outdirfile, 'w')
            for box in boxs:
                b = [box[0],box[1],box[2],box[3],box[-1]]
                b = ' '+",".join(b)
                list_file.write(b)
                # list_file.write('\n')
            geo2 = list(geo)
            geo2[0]=geo[0]+bbox[0]*geo[1]
            geo2[3]=geo[3]+bbox[1]*geo[-1]
            boxxx = dataset.ReadAsArray(bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1])
            write_tif(outdirfile,prj,geo2,boxxx)
            i+=1
        list_file.write('\n')


if __name__=='__main__':
    ###有投影的
    img = './test/clip.tif'
    shapefile = './test/clip.shp'
    txt = './test/clip.json.txt'

    ###无投影的
    # img = './test/aircraft_79.jpg'
    # shapefile = './test/aircraft_79.shp'
    # txt = './test/aircraft_79.shp.txt'

    boxs = shp_to_txt(img, shapefile, txt)
    clip_image(img,boxs,'./test/out',512)