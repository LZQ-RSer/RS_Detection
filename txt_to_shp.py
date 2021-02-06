# -*- coding: utf-8 -*-
# @Time : 2021/2/5 22:46
# @Author : lzq
# @Site : 
# @File : txt_to_shp.py
# @Software: PyCharm Community Edition 
from osgeo import gdal, ogr, osr, gdal_array
import numpy as np

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

def txt_to_shp(img,txt,shapefile):
    line = read_txt(txt)[0]
    line = line.split(' ')[2:]
    box = [x.split(',') for x in line]
    #消除['']
    for b in box:
        if len(b)<=1:
            box.remove(b)
    ##未知行，5列
    boxs = np.array(box).reshape(-1,5)
    dataset = gdal.Open(img)
    im_proj = dataset.GetProjection()##投影信息
    geo = dataset.GetGeoTransform()##地理坐标
    print("im_proj:",im_proj)
    print("geo:",geo)
    # exit()

    ##获得空间参考坐标系编码
    # proj = osr.SpatialReference(wkt=im_proj)
    # space = proj.GetAttrValue('AUTHORITY', 1)

    if im_proj!='':
        # srs = osr.SpatialReference()
        # srs.SetWellKnownGeogCS('WGS84')
        # dataset = gdal.AutoCreateWarpedVRT(dataset, None, srs.ExportToWkt())  # , gdal.GRA_Bilinear)
        zuobiao=1
        z_x=1
    else:
        zuobiao=0
        z_x=-1

    #为了支持中文，添加下面这句话
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8","NO")
    # 为了使属性表字段支持中文，请添加下面这句
    gdal.SetConfigOption("SHAPE_ENCONDING","")
    strVectorFile = shapefile
    # 注册所有的驱动
    ogr.RegisterAll()
    # 创建数据，这里以创建ESRI的shp文件为例
    strDriverName = "ESRI Shapefile"
    driver = ogr.GetDriverByName(strDriverName)
    if driver == None:
        print("驱动不可用:%s",strDriverName)
        return
    # 创建数据源
    ds = driver.CreateDataSource(strVectorFile)
    if ds == None:
        print("创建文件失败：【%s】",strVectorFile)
        return


    oLayer = ds.CreateLayer("object",geom_type=ogr.wkbPolygon)

    # 下面创建属性表
    # 先创建一个叫FieldID的整型属性
    # oFieldID = ogr.FieldDefn('score',ogr.OFTReal)
    # oLayer.CreateField(oFieldID,1)

    # 再创建一个叫FeatureName的字符型属性，字符长度为50
    oFieldName = ogr.FieldDefn("classes", ogr.OFTString)
    oFieldName.SetWidth(10)
    oLayer.CreateField(oFieldName, 1)
    oDefn = oLayer.GetLayerDefn()

    for box in boxs:
        # box[0:-1] = [int(b) for b in box[0:-1]]
        print(box)
        if zuobiao:
            p0 = int(box[0]) * geo[1] + geo[0]
            p1 = int(box[1]) * geo[-1] + geo[3]

            p2 = int(box[2]) * geo[1] + geo[0]
            p3 = int(box[3]) * geo[-1] + geo[3]
        else:
            p0 = int(box[0])
            p1 = int(box[1])

            p2 = int(box[2])
            p3 = int(box[3])
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(float(p0),float(p1*z_x))
        ring.AddPoint(float(p0),float(p3*z_x))
        ring.AddPoint(float(p2),float(p3*z_x))
        ring.AddPoint(float(p2),float(p1*z_x))
        ring.AddPoint(float(p0),float(p1*z_x))
        poly1 = ogr.Geometry(ogr.wkbPolygon)
        poly1.AddGeometry(ring)

        oFeatureRectangle = ogr.Feature(oDefn)
        # oFeatureRectangle.SetField(0, 0)
        oFeatureRectangle.SetField(0, box[4])
        oFeatureRectangle.SetGeometry(poly1)
        oLayer.CreateFeature(oFeatureRectangle)
    if im_proj!='':
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        sr.MorphToESRI()
        prjfile = open(shapefile.replace('shp','prj'),'w')
        prjfile.write(sr.ExportToWkt())
        prjfile.close()
    ds.Destroy()
    print("数据集创建完成！\n")



if __name__=='__main__':
    ###有投影的
    # img = './test/clip.tif'
    # txt = './test/clip.json.txt'
    # shapefile = './test/clip.shp'
    ###无投影的
    img = './test/aircraft_79.jpg'
    txt = './test/aircraft_79.txt'
    shapefile = './test/aircraft_79.shp'

    txt_to_shp(img,txt,shapefile)