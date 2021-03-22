# import cv2
# image = cv2.imread('tmp.jpg', 1)
# rows, cols, channel = image.shape
# affineShrinkTranslationRotation = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
# ShrinkTranslationRotation = cv2.warpAffine(image, affineShrinkTranslationRotation, (cols, rows), borderValue=125)
# cv2.imshow('iii',ShrinkTranslationRotation)
# cv2.waitKey(0)


# python3
# import numpy as np
#
# def py_nms(dets, thresh):
#     """Pure Python NMS baseline."""
#     #x1、y1、x2、y2、以及score赋值
#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]
#     scores = dets[:, 4]
#
#     #每一个候选框的面积
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     #order是按照score降序排序的
#     order = scores.argsort()[::-1]
#
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         #计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])
#
#         #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)
#         inter = w * h
#         #计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)
#
#         #找到重叠度不高于阈值的矩形框索引
#         inds = np.where(ovr <= thresh)[0]
#         #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
#         order = order[inds + 1]
#     return keep
#
# # test
# if __name__ == "__main__":
#     dets = np.array([[30, 20, 230, 200, 1],
#                      [50, 50, 260, 220, 0.9],
#                      [210, 30, 420, 5, 0.8],
#                      [430, 280, 460, 360, 0.7]])
#     thresh = 0.35
#     keep_dets = py_nms(dets, thresh)
#     print(keep_dets)
#     print(dets[keep_dets])



import pandas as pd
import os
import argparse
from models import *
import traceback
from test import test
import csv
path = '/media/hkuit164/WD20EJRX/result/train_result/rgb/rgb_result_sean.csv'
weight_folder='/media/hkuit164/MB155_4/1'
data_folder = '/media/hkuit164/WD20EJRX/yolov3-channel-and-layer-pruning/data/test/'
df = pd.read_csv(path)
# name is id
for name in os.listdir(weight_folder):
    try:
        type = df[df['ID'] == int(name)][:]['tpye']
        activate = df[df['ID'] == int(name)][:]['activation']
        img_size = df[df['ID'] == int(name)][:]['img_size']
        # data = df[df['ID'] == int(name)][:]['data']
        weight_path = os.path.join(weight_folder,name)
        cfg = os.path.join('cfg','yolov3-'+list(type)[0]+'-1cls-'+list(activate)[0]+'.cfg')
        result=[]
        weight_name = os.path.join(weight_path, 'best.weight')
        if not os.path.exists(os.path.join(weight_path,'best.weight')):
            if not os.path.exists(os.path.join(weight_path,'best.pt')):
                print('Not found best.pt')
            else:
                convert(cfg=cfg, weights=os.path.join(weight_path,'best.pt'))
                #for data
                for i in ['all']:
                    data = data_folder+i+'/'+i+'.data'
                    cmd = 'python test.py --cfg {0} --data {1} ' \
                          '--weights {2} --batch-size 8 --img-size {3} ' \
                          '--conf-thres 0.5 --id {4} --csv_path {5}  --write_error_img'.format(cfg, data, weight_name,list(img_size)[0], int(name),path)
                    os.system(cmd)
        else:
            # for i in ['far', 'mul', 'single_front', 'single_side', 'all']:
            for i in ['all']:
                data = data_folder + i + '/' + i + '.data'
                cmd = 'python test.py --cfg {0} --data {1} ' \
                  '--weights {2} --batch-size 8 --img-size {3} ' \
                  '--conf-thres 0.5 --id {4} --csv_path {5}  --write_error_img'.format(cfg, data, weight_name,list(img_size)[0], int(name), path)
                # print(cmd)
                os.system(cmd)
    except:
        if os.path.exists('test_error.txt'):
            os.remove('test_error.txt')
        with open('test_error.txt', 'a+') as f:
            f.write(name)
            f.write('\n')
            f.write('----------------------------------------------\n')
            traceback.print_exc(file=f)


