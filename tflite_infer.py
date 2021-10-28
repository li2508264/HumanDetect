import os
import cv2
import tensorflow as tf
# import tensorflow.contrib as tc
slim = tf.contrib.slim
import numpy as np
def uint82float(a,m,stdv):
    a = a.astype('float32')
    return (a-m)*stdv
def bbox_iou(pb, gt):
    tl = np.maximum(pb[0,0:2], gt[0,0:2])
    br = np.minimum(pb[0,2:4], gt[0,2:4])
    wl = np.minimum(pb[0,0:2], gt[0,0:2])
    wr = np.maximum(pb[0,2:4], gt[0,2:4])
    insect = br-tl
    insect = np.clip(insect,0,99999)
    pw = (pb[0,2]-pb[0,0])
    ph = (pb[0,3]-pb[0,1])
    gw = (gt[0,2]-gt[0,0])
    gh =(gt[0,3]-gt[0,1])
    area_i = insect[0]*insect[1]
    area_a = pw*ph
    area_b = gw*gh
    outRect = wr-wl
    outRect = outRect[0]*outRect[1]
    iou = area_i / (area_a + area_b - area_i)
    # outRect = iou - (outRect-(area_a + area_b - area_i))/outRect
    return iou
def nms(conf,boxes):
    ind = np.argsort(conf)
    ind = ind[::-1]
    conf = conf[ind]
    boxes = boxes[ind,:]
    dp = []
    for i in range(conf.shape[0]):

        if i in dp:
            continue
        for j in range(i+1,conf.shape[0]):
            if j in dp:
                continue
            iou = bbox_iou(boxes[i:i+1,:],boxes[j:j+1,:])
            if iou>0.1:
                dp.append(j)
    res = []
    for i in range(ind.shape[0]):
        if i not in dp:
            res.append(ind[i])
    return np.asarray(res)
Fnet = tf.lite.Interpreter("./Fnet.tflite")
Fnet.allocate_tensors()
input_details = Fnet.get_input_details()
output_details = Fnet.get_output_details()
m1 = output_details[0]['quantization'][1]
stdv1 = output_details[0]['quantization'][0]
frmpath = ".//img//"
fl = os.listdir(frmpath)
input_size = 128
num = 0
for i in fl:
    if num<00:
        num+=1
        continue
    im = cv2.imread(frmpath+i)
    img = im.copy()
    newsize = 0
    if(im.shape[1]>im.shape[0]):
        newsize = im.shape[1]
    else:
        newsize = im.shape[0]
    bg = np.zeros((newsize,newsize,3),dtype = 'uint8')
    bg[0:im.shape[0],0:im.shape[1],:] = im
    im = cv2.resize(bg,(128,128))

    im = np.expand_dims(im,0)
    # im = im.astype('float32')/255.0
    Fnet.set_tensor(input_details[0]['index'], im)
    Fnet.invoke()
    heatmap = Fnet.get_tensor(output_details[0]['index'])
    heatmap = uint82float(heatmap,m1,stdv1)
    ht = heatmap[0]*0.16666
    m, n = np.where(ht[:, :, 0] > 0.9)
    # m,n = nms_point(m,n,ht[:,:,0])
    conf = []
    for t in range(m.shape[0]):
        conf.append(ht[m[t], n[t], 0])
    conf = np.asarray(conf)
    boxes = np.zeros((m.shape[0], 4), dtype='int32')
    for t in range(m.shape[0]):
        mt = m[t] * 8 + ht[m[t], n[t], 3] * 8
        nt = n[t] * 8 + ht[m[t], n[t], 4] * 8
        mt = mt * (newsize / 128.0)
        nt = nt * (newsize / 128.0)
        w = ht[m[t], n[t], 1] * newsize
        h = ht[m[t], n[t], 2] * newsize
        boxes[t, 0] = int(nt - 0.5 * w)
        boxes[t, 2] = int(nt + 0.5 * w)
        boxes[t, 1] = int(mt - 0.5 * h)
        boxes[t, 3] = int(mt + 0.5 * h)
    if (boxes.shape[0] == 0):
        continue
    print(boxes.shape)
    res = nms(conf, boxes)
    boxes = boxes[res]
    conf = conf[res]
    for t in range(boxes.shape[0]):
        img = cv2.rectangle(img, (boxes[t, 0], boxes[t, 1]), (boxes[t, 2], boxes[t, 3]), (255), 1)
    cv2.imshow('test', img)
    cv2.waitKey(0)