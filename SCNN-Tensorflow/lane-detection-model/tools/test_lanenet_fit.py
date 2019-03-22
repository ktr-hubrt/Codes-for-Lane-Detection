#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
测试LaneNet模型
"""
import os
import os.path as ops
import argparse
import time
import math

import tensorflow as tf
import glob
import glog as log
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from PIL import Image
# from scipy import sparse
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from lanenet_model import lanenet_merge_model
from lanenet_model import lanenet_cluster
# from lanenet_model import lanenet_postprocess
from config import global_config
os.environ["CUDA_VISIBLE_DEVICES"] = '5'

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='true')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=1)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()

def slice_lab(lab,size,flag):
    if flag==0:
        _x_bin = 1
        _xbegin = 0
        _xsize = 1
        _y_bin = 6
        _ybegin = 2
        _ysize = 3
    if flag==4:
        _x_bin = 3
        _xbegin = 1
        _xsize = 1
        _y_bin = 2
        _ybegin = 0
        _ysize = 1
    xbin_size = size[1] / _x_bin
    xbeg = xbin_size * _xbegin
    xsize = xbin_size * _xsize

    ybin_size = size[0]/ _y_bin
    ybeg = ybin_size * _ybegin
    ysize =  ybin_size * _ysize

    lab = tf.slice(lab, [0,ybeg,xbeg,0], [-1, ysize, xsize, -1])
    return lab

def f_2(x, A, B, C):
    return A*x*x + B*x + C

def test_lanenet_batch(image_dir, weights_path, batch_size, use_gpu, save_dir=None):
    """

    :param image_dir:
    :param weights_path:
    :param batch_size:
    :param use_gpu:
    :param save_dir:
    :return:
    """
    class_0_iou = AverageMeter()
    class_1_iou = AverageMeter()
    prep_time = AverageMeter()
    pred_time = AverageMeter()
    post_time = AverageMeter()
    assert ops.isfile(image_dir), '{:s} not exist'.format(image_dir)
    root_path = 'data'
    # root_path = '../SCNN/data/CULane/'
    log.info('initiating...')
    lines = open(image_dir, 'r').readlines()
    # image_path_list = glob.glob('{:s}/**/*.jpg'.format(image_dir), recursive=True) + \
                      # glob.glob('{:s}/**/*.png'.format(image_dir), recursive=True) + \
                      # glob.glob('{:s}/**/*.jpeg'.format(image_dir), recursive=True)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 288, 800, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet()
    pred_ret = net.inference(input_tensor, phase_tensor, name='lanenet_loss')
    feature_for_seg = tf.squeeze(tf.nn.sigmoid(pred_ret['lane_seg']),3)
    feature_for_reg = pred_ret['lane_reg']
    # import pdb;pdb.set_trace()
    prediction = feature_for_reg
    # left_prediction = prediction[:,:,:,0]*feature_for_seg+0.001
    left_prediction = prediction[:,:,:,0]*feature_for_seg+0.001
    # feature_for_line = pred_ret['prob_output']
    # feature_for_line = tf.argmax(feature_for_line, axis=-1)
    #feature_for_score = tf.nn.softmax(pred_ret['lane_instance_predictions'])
    #feature_for_score = tf.reduce_max(feature_for_score,3)
    # import pdb;pdb.set_trace()
    cluster = lanenet_cluster.LaneNetCluster()
    # postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
        # print('use gpu！')
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)
    f_list = open('result_list.txt','w')
    # f = open(time.strftime("%Y-%m-%d-%H-%M", time.localtime())+'.txt','w')
    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        epoch_nums = int(math.ceil(len(lines) / batch_size))

        for epoch in range(epoch_nums):
            log.info('[Epoch:{:d}] read images'.format(epoch))
            t_start = time.time()
            image_path_epoch = lines[epoch * batch_size:(epoch + 1) * batch_size]
            # if '05251517_0433.MP4/00210' not in image_path_epoch[0]:
            #     continue
            # print(image_path_epoch[0])
            # image_list = [open(tmp.strip()[28:], 'r').readlines() for tmp in image_path_epoch]
            image_path_list = [root_path +tmp.strip() for tmp in image_path_epoch]
            image_list_epoch = [cv2.imread(tmp, cv2.IMREAD_COLOR) for tmp in image_path_list]
            image_vis_list = image_list_epoch
            # flag_list =  [tmp[2][0] for tmp in image_list]
            # import pdb;pdb.set_trace()
            seg_gt_path = ['data/result_test/' +tmp.strip().replace('.jpg','_sur.png') for tmp in image_path_epoch]
            seg_gt_list = [cv2.imread(tmp,0) for tmp in seg_gt_path]
            import pdb;pdb.set_trace()
            # lane_gt_path = ['data/result_test/' +tmp.strip().replace('.jpg','_lane.png') for tmp in image_path_epoch]
            # lane_gt_list = [cv2.imread(tmp,0) for tmp in lane_gt_path]
            use_gt = 0
            if use_gt ==1:
                reg_l_path = ['data/result_test/' +tmp.strip().replace('.jpg','_l_3.png') for tmp in image_path_epoch]
                reg_l_list = [cv2.imread(tmp,0)[96:] for tmp in reg_l_path]
                reg_r_path = ['data/result_test/' +tmp.strip().replace('.jpg','_r_3.png') for tmp in image_path_epoch]
                reg_r_list = [cv2.imread(tmp,0)[96:] for tmp in reg_r_path]
            image_list_epoch = [cv2.resize(tmp, (800, 288), interpolation=cv2.INTER_LINEAR) for tmp in image_list_epoch]
            image_list_epoch = [tmp - VGG_MEAN for tmp in image_list_epoch]
            t_cost = time.time() - t_start
            prep_time.update(t_cost / len(image_path_epoch))
            log.info('[Epoch:{:d}] preposs {:d} images, total: {:.5f}s, average: {:.5f}s'.format(
                epoch, len(image_path_epoch), t_cost, prep_time.avg))

            t_start = time.time()
            if 1:
                binary_seg_images, dis_maps, l_dis = sess.run(
                    [feature_for_seg, feature_for_reg, left_prediction], feed_dict={input_tensor: image_list_epoch})
            if use_gt ==1:
                # import pdb;pdb.set_trace()
                l_map = np.reshape(reg_l_list[0], [192,800,1])/250.0
                r_map = np.reshape(reg_r_list[0], [192,800,1])/250.0
                dis_maps = np.reshape(np.concatenate((l_map, r_map), axis=2),[1,192,800,2])

                # import pdb;pdb.set_trace()
            t_cost = time.time() - t_start
            pred_time.update(t_cost / len(image_path_epoch))
            log.info('[Epoch:{:d}] pred {:d} images, total: {:.5f}s, average: {:.5f}s'.format(
                epoch, len(image_path_epoch), t_cost, pred_time.avg))

            cluster_time = []
            for index, binary_seg_image in enumerate(binary_seg_images):
                t_start = time.time()

                mask_png = np.zeros(shape=[288, 800], dtype=np.float32)
                mask_png[96:,:] = binary_seg_image
                #mask_png_1 = np.zeros(shape=[288, 800], dtype=np.float32)
                #mask_png_1[96:,:] = binary_score_images[index]
                #mask_png_2 = np.zeros(shape=[288, 800], dtype=np.float32)
                #mask_png_2[96:,:] = instance_seg_images[index]
                # import pdb;pdb.set_trace()
                mask_image, iou_mask_image, single_class_iou, curve_parameter, points_all = cluster.get_lane_mask_centre(
                                                   binary_seg_ret=mask_png,
                                                   #instance_seg_ret=mask_png_2,
                                                   #binary_score_images=mask_png_1,
                                                   gt_seg_ret=seg_gt_list[index],
                                                   raw_image=image_vis_list[index],
                                                   reg_image=dis_maps[index],
                                                   )
                # mask_image, iou_mask_image, single_class_iou, curve_parameter, points_all = cluster.get_lane_mask_boom(binary_seg_ret=binary_seg_image,
                #                                    binary_score_images=binary_score_images[index],
                #                                    gt_seg_ret=seg_gt_list[index],
                #                                    gt_lane_ret=lane_gt_list[index],
                #                                    reg_image=dis_maps[index],
                #                                    )
                cluster_time.append(time.time() - t_start)
                mask_image = cv2.resize(mask_image, (800,
                                                     288),
                                                     interpolation=cv2.INTER_LINEAR)
                # import pdb;pdb.set_trace()
                if save_dir is None:
                    import pdb;pdb.set_trace()

                if save_dir is not None:
                    image_vis_list[index] = cv2.resize(image_vis_list[index], (800,288),
                                     interpolation=cv2.INTER_LINEAR)
                    mask_image = cv2.addWeighted(image_vis_list[index], 1.0, mask_image, 1.0, 0)
                    image_name = image_path_epoch[index][1:].strip().replace('/','_')
                    #image_name = ops.split(image_path_list[index])[-1]
                    image_save_path = ops.join('/data3/CULane/test_image', save_dir, image_name)
                    font = cv2.FONT_HERSHEY_TRIPLEX

                    # for i in range(len(single_class_iou)):
                    #     tmp_iou = single_class_iou[i]
                    #     cv2.putText(mask_image,'seg_class {:d} iou {:.3f}'.format(i,tmp_iou),(5,50+i*20),font,0.6,(255,0,0),2,False)
                    log.info(image_save_path)
                    if len(curve_parameter)>1:
                        f_list.write(image_path_epoch[index])
                    #f_list.write('\n')
                    # gt_points = open('/home/nisheng.lh/SCNN/data/CULane'+image_path_epoch[index].strip().replace('jpg','lines.txt'), 'r').readlines()
                    # if len(gt_points)>len(curve_parameter):
                    #     import pdb;pdb.set_trace()
                    prefix = '/data2/lvhui/SCNN/tools/prob2lines/output/'+save_dir
                    file_name = image_path_epoch[index].strip().replace('jpg','lines.txt')
                    directory = prefix+file_name[:file_name.index(file_name.split('/')[-1])]
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    f = open(prefix+file_name,'w')
                    for line_ind in range(len(points_all)):
                        for j in range(0,len(points_all[line_ind]),2):
                            # import pdb;pdb.set_trace()
                            f.write(str(int(points_all[line_ind][j])))
                            f.write(' ')
                            f.write(str(int(points_all[line_ind][j+1])))
                            f.write(' ')
                        f.write('\n')
                    # points_all=[]
                    # for line_ind in range(len(gt_points)):
                    #     gt_p = gt_points[line_ind].strip().split(' ')
                    #     curve = curve_parameter[line_ind]
                    #     points = []
                    #     for j in range(1,len(gt_p),2):
                    #         points.append(f_2(float(gt_p[j])*288/590, curve[0], curve[1], curve[2])*1640/480)
                    #         points.append(float(gt_p[j]))
                    #     points_all.append(points)
                        # import pdb;pdb.set_trace()
                    # f.write('nisheng ')
                    # f.write(seg_gt_path[index])
                    # f.write(' ')
                    # for i in range(len(curve_parameter)):
                    #     if curve_parameter[i] is None:
                    #         f.write('None')
                    #         f.write(' ')
                    #     else:
                    #         for j in range(3):
                    #             f.write(str(curve_parameter[i].tolist()[j]))
                    #             f.write(' ')
                    # f.write('\n')
                    f.close() 
                    #import pdb;pdb.set_trace()
                    cv2.imwrite(image_save_path.replace('.jpg','_raw.jpg'),image_vis_list[index])
                    cv2.imwrite(image_save_path, mask_image)
                    cv2.imwrite(image_save_path.replace('.jpg','_sur.jpg'), iou_mask_image)
                    # import pdb;pdb.set_trace()
                    # cv2.imwrite(image_save_path.replace('.jpg','_gt.png'), seg_gt_list[index]*51)
                    # cv2.imwrite(image_save_path.split('.')[0]+'_mix.png', mask_img)
                    
                    class_0_iou.update(single_class_iou[0])
                    class_1_iou.update(single_class_iou[1])
                    # log.info('[Epoch:{:d}] Detection image {:s} complete'.format(epoch, image_name))
            post_time.update(np.mean(cluster_time))
            log.info('[Epoch:{:d}] postpross {:d} images, total: {:.5f}s, average: {:.5f}s'.format(
                epoch, len(image_path_epoch), np.sum(cluster_time), post_time.avg))
            # import pdb;pdb.set_trace()
            log.info('class_0_iou:{:.5f},class_1_iou:{:.5f}'.format(
                class_0_iou.avg,class_1_iou.avg))
    f_list.close() 
    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()
    save_dir = ops.join('/data3/CULane/test_image', args.save_dir)
    if args.save_dir is not None and not ops.exists(save_dir):
        log.error('{:s} not exist and has been made'.format(args.save_dir))
        os.makedirs(save_dir)

    if args.is_batch.lower() == 'false':
        # test hnet model on single image
        test_lanenet(args.image_path, args.weights_path, args.use_gpu)
    else:
        # test hnet model on a batch of image
        test_lanenet_batch(image_dir=args.image_path, weights_path=args.weights_path,
                           save_dir=args.save_dir, use_gpu=args.use_gpu, batch_size=args.batch_size)
