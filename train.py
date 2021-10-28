import os
import cv2
import tensorflow as tf
# import tensorflow.contrib as tc
slim = tf.contrib.slim
import numpy as np
import pdb
from blaze_loss import _Loss
from utils import bbox_utils,train_utils
from angment import _crop,_fliph,tcs,_expand,_distort
batch_norm_params = {
            'scale': True,
            'decay': 0.99,
            'epsilon': 0.0001
        }
hyper_params = train_utils.get_hyper_params()
def uint82float(a, m, stdv):
    a = a.astype('float32')
    return (a - m) * stdv


def bbox_iou(pb, gt):
    tl = np.maximum(pb[0, 0:2], gt[0, 0:2])
    br = np.minimum(pb[0, 2:4], gt[0, 2:4])
    wl = np.minimum(pb[0, 0:2], gt[0, 0:2])
    wr = np.maximum(pb[0, 2:4], gt[0, 2:4])
    insect = br - tl
    insect = np.clip(insect, 0, 99999)
    pw = (pb[0, 2] - pb[0, 0])
    ph = (pb[0, 3] - pb[0, 1])
    gw = (gt[0, 2] - gt[0, 0])
    gh = (gt[0, 3] - gt[0, 1])
    area_i = insect[0] * insect[1]
    area_a = pw * ph
    area_b = gw * gh
    outRect = wr - wl
    outRect = outRect[0] * outRect[1]
    iou = area_i / (area_a + area_b - area_i)
    # outRect = iou - (outRect-(area_a + area_b - area_i))/outRect
    return iou


def nms(conf, boxes):#conf [n] boxes[n*4]
    ind = np.argsort(conf)
    ind = ind[::-1]
    conf = conf[ind]
    boxes = boxes[ind, :]
    dp = []
    for i in range(conf.shape[0]):

        if i in dp:
            continue
        for j in range(i + 1, conf.shape[0]):
            if j in dp:
                continue
            iou = bbox_iou(boxes[i:i+1, :], boxes[j:j+1, :])

            if iou > 0.15:
                dp.append(j)
    res = []
    for i in range(ind.shape[0]):
        if i not in dp:
            res.append(ind[i])
    return np.asarray(res)
def _parse_function(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'boxes': tf.FixedLenFeature([9*4],tf.float32),
            'img_raw':tf.FixedLenFeature([],tf.string)
        }
    )
    image = tf.decode_raw(features['img_raw'], tf.uint8)  # compare with input of tfrecord
    image = tf.cast(image, tf.float32)
    boxes = tf.cast(features['boxes'], tf.float32)
    boxes = tf.reshape(boxes,(9,4))
    image = tf.reshape(image,(320,320,3))
    imbatch = image / 255.0
    return imbatch,boxes

class BlazeFace(object):
    def __init__(self, is_training=True, input_w=128,input_h=128):
        self.input_w = input_w
        self.input_h = input_h
        self.is_training = is_training
        self.normalizer = tf.keras.layers.BatchNormalization
        self._create_placeholders()
        xs = np.zeros((1,1344,4))
        xs[:,:,0:2] = -3
        self.mybias = tf.constant(xs, dtype=tf.float32)
        self._build_model()

    def _create_placeholders(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_h, self.input_w, 3], name='input')

    def _build_model(self):
        self.output = self._Conv2d(self.input,64,5,2)
        self.output = self.BlazeBlock(self.output, 64, 5, 1)
        self.output = self.BlazeBlock(self.output, 64, 5, 1)
        self.output = self.BlazeBlock(self.output, 96, 5, 2)
        self.output = self.BlazeBlock(self.output, 96, 5, 1)
        self.output = self.BlazeBlock(self.output, 96, 5, 1)
        self.output = self.Double_BlazeBlock(self.output, 64, 192, 5, 2)
        self.output = self.Double_BlazeBlock(self.output, 64, 192, 5, 1)
        self.output16 = self.Double_BlazeBlock(self.output, 64, 192, 5, 1)
        self.loc16 = self._Loc16(self.output16,5,3,1)*0.16666
    def BlazeBlock(self, input, out, kernel_size,stride):
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            activation_fn=tf.nn.relu6,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.),
                            biases_initializer=None):
            with slim.arg_scope([slim.batch_norm], center=True, scale=True, is_training=self.is_training):
                output = slim.separable_conv2d(input, num_outputs=None, kernel_size=kernel_size, stride=stride,
                            padding='SAME',
                            normalizer_fn=None,
                            normalizer_params=None,
                            activation_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.),
                            biases_initializer=None)
                output = slim.conv2d(output, num_outputs=out, kernel_size=1)
                # shortcut = slim.max_pool2d(input,kernel_size = stride,stride = stride,padding = 'SAME')
                # shortcut = slim.conv2d(shortcut, num_outputs=out, kernel_size=1)
                return output#+shortcut
    def Double_BlazeBlock(self, input, mid,out, kernel_size,stride):
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            activation_fn=tf.nn.relu6,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.),
                            biases_initializer=None):
            with slim.arg_scope([slim.batch_norm], center=True, scale=True, is_training=self.is_training):
                output = slim.separable_conv2d(input, num_outputs=None, kernel_size=kernel_size, stride=stride,
                            padding='SAME',
                            normalizer_fn=None,
                            normalizer_params=None,
                            activation_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.1),
                            biases_initializer=None)
                output = slim.conv2d(output, num_outputs=mid, kernel_size=1)
                output = slim.separable_conv2d(output, num_outputs=None, kernel_size=kernel_size, stride=1,
                            padding='SAME',
                            normalizer_fn=None,
                            normalizer_params=None,
                            activation_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.),
                            biases_initializer=None)
                output = slim.conv2d(output, num_outputs=out, kernel_size=1)
                return output
    def _Conv2d(self, input, out, kernel_size,stride):
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            activation_fn=tf.nn.relu6,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.),
                            biases_initializer=None):
            with slim.arg_scope([slim.batch_norm], center=True, scale=True, is_training=self.is_training):
                output = slim.conv2d(input, num_outputs=out, kernel_size=kernel_size,stride = stride)
                return output
    def _Loc16(self, input, out, kernel_size,stride):
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            normalizer_fn=None,
                            normalizer_params=None,
                            activation_fn=tf.nn.relu6,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.),
                            biases_initializer=None):
            output = slim.separable_conv2d(input, num_outputs=None, kernel_size=kernel_size, stride=stride,
                padding='SAME',
                normalizer_fn=None,
                normalizer_params=None,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=slim.l2_regularizer(0.),
                biases_initializer=None)
            output = slim.conv2d(output, num_outputs=out, kernel_size=1)
            return output

def bbox_iou(pb, gt):
    tl = np.maximum(pb[0,:], gt[0,:])
    br = np.minimum(pb[1,:], gt[1,:])
    wl = np.minimum(pb[0,:], gt[0,:])
    wr = np.maximum(pb[1,:], gt[1,:])
    insect = br-tl
    insect = np.clip(insect,0,99999)
    pw = (pb[1,0]-pb[0,0])
    ph = (pb[1,1]-pb[0,1])
    gw = (gt[1,0]-gt[0,0])
    gh =(gt[1,1]-gt[0,1])
    area_i = insect[0]*insect[1]
    area_a = pw*ph
    area_b = gw*gh
    outRect = wr-wl
    outRect = outRect[0]*outRect[1]
    iou = area_i / (area_a + area_b - area_i)
    # outRect = iou - (outRect-(area_a + area_b - area_i))/outRect
    return iou

# boxes = np.asarray([[[0.3125-0.1484375,0.3125-0.1484375,0.3125+0.1484375,0.3125+0.1484375]]])
# boxes = np.zeros([10,1,4])
# landmark = np.asarray([[[[0.3125,0.3125],[0.3225,0.3025],[0.3135,0.3115],[0.3125,0.3125],[0.3125,0.3125],[0.3125,0.3125]]]])

# gt_landmarks = tf.placeholder(dtype=tf.float32,shape=[1,1,6,2])
# sess2 = tf.Session()
# gloc,glabel,m = sess2.run([gt_loc,gt_labels,mask],feed_dict={gt_boxes:boxes,gt_landmarks:landmark})
# p = sess2.run(prior_boxes)

def warmup(lr,step,warm_steps,epoch_size,warm_ratio,decay_ratio):
    if step<warm_steps:
        lr = float(step)/warm_steps*warm_ratio+0.00001
        return lr
    if step%epoch_size ==0:
        lr = lr*decay_ratio
    return lr
def box2center(box):
    box[:,:,2] = box[:,:,2]-box[:,:,0]
    box[:,:,3] = box[:,:,3]-box[:,:,1]
    box[:,:,1] += 0.5*box[:,:,3]
    box[:, :, 0] += 0.5 * box[:,:, 2]
    return box
def Gau(d,throld = 0.03):
    std = 2
    m = 2*std**2
    t = np.exp(-(d)/m)/(np.pi*m)
    return t
def gener_heatmap(cwh,width,height,input = 128.0):
    stride = input/width
    heatmap = np.zeros((cwh.shape[0],height,width,5),dtype = 'float32')
    for i in range(cwh.shape[0]):
        for b in range(cwh.shape[1]):
            if np.sum(cwh[i,b])==0:
                break
            m = int(cwh[i,b,0]//stride)
            n = int(cwh[i,b,1]//stride)
            bm = int(cwh[i,b,0]%stride)
            bn = int(cwh[i,b,1] % stride)
            heatmap[i,n,m,1] = cwh[i,b,2]/input
            heatmap[i, n,m, 2] = cwh[i,b, 3]/input
            heatmap[i,n,m,3] = bm/stride
            heatmap[i, n,m, 4] = bn/stride
            for j in range(height):
                for k in range(width):
                    heatmap[i,k,j,0] = max(Gau((j-m)**2+(k-n)**2),heatmap[i,k,j,0])#多个heatmap时的argmax
        if np.max(heatmap[i,:,:, 0])>0:
            heatmap[i,:,:, 0] /= np.max(heatmap[i,:,:, 0])
        tmp = heatmap[i,:,:, 0]
        tmp[np.where(tmp<0.5)] = 0
        heatmap[i, :, :, 0] = tmp
    # for i in range(heatmap.shape[0]):
    #     imh = heatmap[i, :, :, 0] * 255
    #     imh = imh.astype('uint8')
    #     cv2.imwrite('./test/'+str(i)+'.png',imh)
    #     cv2.imshow('imh', imh)
    #     cv2.waitKey(0)

    heat_mask = heatmap.copy()
    heat_mask[np.where(heat_mask!=0)] = 1
    return heatmap,heat_mask
if __name__ == '__main__':
    dira = "/root/src/People_Detect/tfdata_big/"
    fls = os.listdir(dira)
    d = []
    for i in fls:
        d.append(dira + i)
    bgrpath = "/root/src/Face_Detect/dataset/bg/"
    bglist = os.listdir(bgrpath)
    bglist = [bgrpath + i for i in bglist]
    batch_size = 32
    epoch_size = int(len(fls)/batch_size)
    dataset = tf.data.TFRecordDataset(d)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(batch_size*10).repeat().prefetch(batch_size*10).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    imbatch,bboxes = iterator.get_next()
    train = True
    heatmap_ = tf.placeholder(dtype=tf.float32, shape=[None, 16,16, 5])
    heatmask_ = tf.placeholder(dtype=tf.float32, shape=[None, 16,16, 5])
    model = BlazeFace(True)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    HeatL = _Loss()
    g = tf.get_default_graph()
    if train:

        loss = HeatL.loc(heatmap_,model.loc16,heatmask_)
        #tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=2000)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(keep_prob).minimize(loss)
    else:
        tf.contrib.quantize.create_eval_graph(input_graph=g)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.Session(config=config)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    lr = 0.001
    dec = 0.95
    once = 1
    vlist = []
    for l in  tf.trainable_variables():

        vlist.append(l)
    saver = tf.compat.v1.train.Saver(max_to_keep=10,var_list = vlist)
    saver1 = tf.compat.v1.train.Saver()
    #saver.restore(sess, "./checkpoint/model/Fnetquan_30000_mioutest4.3910613_mioutrain0.029082093.ckpt")
    ####################数据集测试
    for step in range(50001):
        lr = warmup(lr,step,3000,epoch_size,0.01,dec)
        image, boxes = sess.run([imbatch, bboxes])
        # image  = (image*255)
        # image = image.astype('uint8')
        ##数据处理
        # print(boxes)

        for j in range(image.shape[0]):##数据增强前就已经是图像已经除以255 范围[0,1],box是在256尺度下
        #     #print(np.max(boxes))
            image[j],boxes[j] = _crop(image[j], boxes[j], bglist[np.random.randint(0, len(bglist))])
        #     #print(np.max(boxes))
        #     if np.random.randint(2)==0:
        #         image[j],boxes[j] =_fliph(image[j], boxes[j])
        # #
        #
        #     if np.random.randint(2)==0:
        #         image[j], boxes[j] = _expand(image[j], boxes[j], bglist[np.random.randint(0, len(bglist))])
            image[j],boxes[j] = tcs(image[j],boxes[j],bglist[np.random.randint(0, len(bglist))])
        #
        #     if np.random.randint(2)==0:
        #         image[j] = _distort(image[j])
        img_input = np.zeros((image.shape[0],128,128,3),dtype='float32')
        for j in range(image.shape[0]):
            img_input[j,:,:,:] = cv2.resize(image[j,:,:,:],(128,128))
            boxes[j] = boxes[j]*(128.0/image.shape[1])
        cwh = box2center(boxes)
        heatmap_gt ,heat_mask = gener_heatmap(cwh,16,16)

        sess.run(train_step,feed_dict={model.input:img_input,heatmap_:heatmap_gt,heatmask_:heat_mask,keep_prob:lr})
        if step%500==0:
            pre_heat,l = sess.run([model.loc16,loss],feed_dict={model.input:img_input,heatmap_:heatmap_gt,heatmask_:heat_mask})#gloc[n,1344,4] label [n,1344,1]

            for p in range(pre_heat.shape[0]):
                m,n = np.where(heatmap_gt[p,:,:,0]==np.max(heatmap_gt[p,:,:,0]))
                mp, np = np.where(pre_heat[p, :, :, 0] == np.max(pre_heat[p, :, :, 0]))
                print("gt max is %d %d ,predict is %d %d",m,n,mp,np)
                print("w,h is %f %f pre is %f %f",heatmap_gt[p,m,n,1],heatmap_gt[p,m,n,2],pre_heat[p,m,n,1],pre_heat[p,m,n,2])
                print("bias  is %f %f pre is %f %f", heatmap_gt[p, m, n, 3], heatmap_gt[p, m, n, 4], pre_heat[p, m, n,3],
                      pre_heat[p, m, n, 4])
                #print(label.shape,label2.shape)
            print("%d step lr %f loc_loss: %f "%(step,lr,l))
        if step%1000==0:
            saver1.save(sess, "./model_netbig2/Fnet_" + str(step).zfill(5) + "_heatmaploss" + str(l)  +".ckpt")