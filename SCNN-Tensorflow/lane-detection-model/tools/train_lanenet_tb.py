#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-18 下午7:31
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : train_lanenet.py
# @IDE: PyCharm Community Edition
"""
训练lanenet模型
"""
import argparse
import os
import os.path as ops
import time

import glog as log
import numpy as np
import tensorflow as tf

import sys

from config import global_config
from lanenet_model import lanenet_merge_model
from data_provider import lanenet_data_processor
from tensorflow.python import pywrap_tensorflow

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, help='The training dataset dir path')
    parser.add_argument('--net', type=str, help='Which base net work to use', default='vgg')
    parser.add_argument('--weights_path', type=str, help='The pretrained weights path')
    parser.add_argument('--version', type=str, help='the verson', default='v0.0')
    return parser.parse_args()

def freeze_para(tower_grads):
    
    gradients = []
    for i in range(len(tower_grads)):
        grad = []
        for j in range(len(tower_grads[i])):
            #if 'conv1x1_c4_2' in tower_grads[i][j][0].name or 'conv1x1_c4_3_' in tower_grads[i][j][0].name:
            #import pdb;pdb.set_trace()
            if 'decode_emb' in tower_grads[i][j][1].name or 'lossWt' in tower_grads[i][j][1].name:
                grad.append(tower_grads[i][j])
        gradients.append(grad)
    # import pdb;pdb.set_trace()
    print(gradients)
    return gradients

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def forward(batch_queue, net, phase, scope, optimizer=None):
    img_batch, label_instance_batch, label_existence_batch, lane_binary_batch, lane_lmap_batch, lane_rmap_batch = batch_queue.dequeue()
    tf.summary.image('img_gt', tf.cast(img_batch+lanenet_data_processor.VGG_MEAN, tf.uint8), 1)
    inference = net.inference(img_batch, phase, 'lanenet_loss')
    _ = net.loss(inference, label_instance_batch, label_existence_batch, lane_binary_batch, lane_lmap_batch, lane_rmap_batch, 'lanenet_loss')
    total_loss = tf.add_n(tf.get_collection('total_loss', scope))
    instance_loss = tf.add_n(tf.get_collection('instance_seg_loss', scope))
    existence_loss = tf.add_n(tf.get_collection('existence_pre_loss', scope))

    out_logits = tf.add_n(tf.get_collection('instance_seg_logits', scope))
    # calculate the accuracy
    out_logits = tf.nn.softmax(logits=out_logits)
    out_logits_out = tf.argmax(out_logits, axis=-1)

    pred_0 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(label_instance_batch, 0), tf.int32),
                                          tf.cast(tf.equal(out_logits_out, 0), tf.int32)),
                              dtype=tf.int32)
    pred_1 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(label_instance_batch, 1), tf.int32),
                                          tf.cast(tf.equal(out_logits_out, 1), tf.int32)),
                              dtype=tf.int32)
    pred_2 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(label_instance_batch, 2), tf.int32),
                                          tf.cast(tf.equal(out_logits_out, 2), tf.int32)),
                              dtype=tf.int32)
    pred_3 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(label_instance_batch, 3), tf.int32),
                                          tf.cast(tf.equal(out_logits_out, 3), tf.int32)),
                              dtype=tf.int32)
    pred_4 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(label_instance_batch, 4), tf.int32),
                                          tf.cast(tf.equal(out_logits_out, 4), tf.int32)),
                              dtype=tf.int32)
    gt_all = tf.count_nonzero(tf.cast(tf.greater(label_instance_batch, 0), tf.int32), dtype=tf.int32)
    gt_back = tf.count_nonzero(tf.cast(tf.equal(label_instance_batch, 0), tf.int32), dtype=tf.int32)

    pred_all = tf.add(tf.add(tf.add(pred_1, pred_2), pred_3), pred_4)

    accuracy = tf.divide(tf.cast(pred_all, tf.float32), tf.cast(gt_all, tf.float32))
    accuracy_back = tf.divide(tf.cast(pred_0, tf.float32), tf.cast(gt_back, tf.float32))

    # Compute mIoU of Lanes
    overlap_1 = pred_1
    union_1 = tf.add(tf.count_nonzero(tf.cast(tf.equal(label_instance_batch, 1),
                                              tf.int32), dtype=tf.int32),
                     tf.count_nonzero(tf.cast(tf.equal(out_logits_out, 1),
                                              tf.int32), dtype=tf.int32))
    union_1 = tf.subtract(union_1, overlap_1)
    IoU_1 = tf.divide(tf.cast(overlap_1, tf.float32), tf.cast(union_1, tf.float32))

    overlap_2 = pred_2
    union_2 = tf.add(tf.count_nonzero(tf.cast(tf.equal(label_instance_batch, 2),
                                              tf.int32), dtype=tf.int32),
                     tf.count_nonzero(tf.cast(tf.equal(out_logits_out, 2),
                                              tf.int32), dtype=tf.int32))
    union_2 = tf.subtract(union_2, overlap_2)
    IoU_2 = tf.divide(tf.cast(overlap_2, tf.float32), tf.cast(union_2, tf.float32))

    overlap_3 = pred_3
    union_3 = tf.add(tf.count_nonzero(tf.cast(tf.equal(label_instance_batch, 3),
                                              tf.int32), dtype=tf.int32),
                     tf.count_nonzero(tf.cast(tf.equal(out_logits_out, 3),
                                              tf.int32), dtype=tf.int32))
    union_3 = tf.subtract(union_3, overlap_3)
    IoU_3 = tf.divide(tf.cast(overlap_3, tf.float32), tf.cast(union_3, tf.float32))

    overlap_4 = pred_4
    union_4 = tf.add(tf.count_nonzero(tf.cast(tf.equal(label_instance_batch, 4),
                                              tf.int64), dtype=tf.int32),
                     tf.count_nonzero(tf.cast(tf.equal(out_logits_out, 4),
                                              tf.int64), dtype=tf.int32))
    union_4 = tf.subtract(union_4, overlap_4)
    IoU_4 = tf.divide(tf.cast(overlap_4, tf.float32), tf.cast(union_4, tf.float32))

    IoU = tf.reduce_mean(tf.stack([IoU_1, IoU_2, IoU_3, IoU_4]))

    tf.get_variable_scope().reuse_variables()

    if optimizer is not None:
        grads = optimizer.compute_gradients(total_loss)
    else:
        grads = None
    return total_loss, instance_loss, existence_loss, accuracy, accuracy_back, IoU, out_logits_out, grads

def get_variables_in_checkpoint_file(file_name):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print(
                "It's likely that your checkpoint file has been compressed "
                "with SNAPPY.")

def load_model(sess, model_path):
    #TODO(global variables ?? how about _adam weights)
    variables = tf.global_variables()
    var_keep_dic = get_variables_in_checkpoint_file(model_path)
    # if 'global_step' in var_keep_dic:
    #     var_keep_dic.pop('global_step')

    fileObject = open('variables.txt', 'w')
    for v in variables:
      fileObject.write(v.name.split(':')[0])
      fileObject.write('\n')
    fileObject.close()
    fileObject = open('var_keep_dic.txt', 'w')
    for v in var_keep_dic:
      if 'BoxPre' in v or 'road/seg' in v or 'chedao' in v or 'left' in v or 'right' in v:
        continue
      fileObject.write(v)
      fileObject.write('\n')
    fileObject.close()
    # # vis_var_keep_dic = []
    # import pdb;pdb.set_trace()
    not_to_restore_list = [
        'Variable',
        #'_lossWt',
        # 'conv_6',
        # 'reg',
        #'pix',
        ]
    variables_to_restore = []
    flag = True
    for v in variables:
        # if v.name.split(':')[0] in var_keep_dic and 'LaneDet/' not in v.name and 'merge' not in v.name:
        for not_to_restore_name in not_to_restore_list:
            if not_to_restore_name in v.name:
                flag = False
                continue
        if v.name.split(':')[0] in var_keep_dic and flag:
            # print('Varibles restored: %s' % v.name)
            variables_to_restore.append(v)
            # vis_var_keep_dic.append(v.name.split(':')[0])
        else:
            # print('Unrestored Variables: %s' % v.name)
            pass
        flag = True
    # print('Extra Variables in ckpt', set(var_keep_dic) - set(vis_var_keep_dic))
    import pdb;pdb.set_trace()
    if len(variables_to_restore) > 0:
        print(len(variables))
        print(len(variables_to_restore))
        fileObject = open('variables_restore.txt', 'w')
        for v in variables_to_restore:
            fileObject.write(v.name.split(':')[0])
            fileObject.write('\n')
        fileObject.close()
        
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, model_path)

        #import pdb;pdb.set_trace()
    else:
        print('No variables in {} fits the network'.format(model_path))

def train_net(dataset_dir, version, weights_path=None, net_flag='vgg'):
    train_dataset_file = ops.join(dataset_dir, 'train_gt.txt')
    val_dataset_file = ops.join(dataset_dir, 'val_gt.txt')

    assert ops.exists(train_dataset_file)

    phase = tf.placeholder(dtype=tf.string, shape=None, name='net_phase')

    train_dataset = lanenet_data_processor.DataSet(train_dataset_file)
    # val_dataset = lanenet_data_processor.DataSet(val_dataset_file)

    net = lanenet_merge_model.LaneNet()
    global_summaries = set([])
    tower_grads = []

    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.polynomial_decay(CFG.TRAIN.LEARNING_RATE, global_step,
                                              CFG.TRAIN.EPOCHS, power=0.9)
    global_summaries.add(tf.summary.scalar('Learning_Rate', learning_rate))

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    img, label_instance, label_existence, lane_binary, lane_lmap, lane_rmap = train_dataset.next_batch(CFG.TRAIN.BATCH_SIZE)
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
        [img, label_instance, label_existence, lane_binary, lane_lmap, lane_rmap], capacity=2 * CFG.TRAIN.GPU_NUM, num_threads=CFG.TRAIN.CPU_NUM)

    # val_img, val_label_instance, val_label_existence = val_dataset.next_batch(CFG.TRAIN.VAL_BATCH_SIZE)
    # val_batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
    #     [val_img, val_label_instance, val_label_existence], capacity=2 * CFG.TRAIN.GPU_NUM,
    #     num_threads=CFG.TRAIN.CPU_NUM)
    global_summaries = list(global_summaries)
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(CFG.TRAIN.GPU_NUM):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i) as scope:
                    total_loss, instance_loss, existence_loss, accuracy, accuracy_back, _, out_logits_out, \
                        grad = forward(batch_queue, net, phase, scope, optimizer)
                    tower_grads.append(grad)
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    global_summaries += summaries
                # with tf.name_scope('test_%d' % i) as scope:
                #     val_op_total_loss, val_op_instance_loss, val_op_existence_loss, val_op_accuracy, \
                #         val_op_accuracy_back, val_op_IoU, _, _ = forward(val_batch_queue, net, phase, scope)

    grads = average_gradients(tower_grads)

    train_op = optimizer.apply_gradients(grads, global_step=global_step)

    train_cost_time_mean = []
    train_instance_loss_mean = []
    train_existence_loss_mean = []
    train_accuracy_mean = []
    train_accuracy_back_mean = []

    global_summaries.append(tf.summary.scalar(name='total_cost', tensor=total_loss))
    global_summaries.append(tf.summary.scalar(name='instance_cost', tensor=instance_loss))
    global_summaries.append(tf.summary.scalar(name='exist_cost', tensor=existence_loss))
    # global_summaries.append(tf.summary.image('img_gt', tf.cast(input_tensor+VGG_MEAN, tf.uint8), 1))
    summary_op = tf.summary.merge(global_summaries, name='summary_op')
    

    saver = tf.train.Saver()
    model_save_dir = 'model/'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_name = '{:s}_{:s}'.format(net_flag, version)
    model_save_path = ops.join(model_save_dir, model_name)
    # Set tf summary
    tboard_save_path = model_save_path
    model_save_path = ops.join(model_save_path, 'model')
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    summary_writer = tf.summary.FileWriter(tboard_save_path)

    sess_config = tf.ConfigProto(device_count={'GPU': CFG.TRAIN.GPU_NUM}, allow_soft_placement=True)
    # sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    with tf.Session(config=sess_config) as sess:
        with sess.as_default():
            init = tf.global_variables_initializer()
            sess.run(init)
            if weights_path is None:
                log.info('Training from scratch')
                
            else:
                log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
                load_model(sess,weights_path)

            # 加载预训练参数
            if net_flag == 'vgg' and weights_path is None:
                pretrained_weights = np.load(
                    '/data3/vgg16.npy',
                    encoding='latin1').item()
                num_ini = 0
                for vv in tf.trainable_variables():
                    weights = vv.name.split('/')
                    if len(weights) >= 3 and weights[-3] in pretrained_weights:
                        try:
                            weights_key = weights[-3]
                            weights = pretrained_weights[weights_key][0]
                            _op = tf.assign(vv, weights)
                            sess.run(_op)
                            num_ini += 1
                        except Exception as e:
                            continue
                print(num_ini)
        tf.train.start_queue_runners(sess=sess)
        for epoch in range(CFG.TRAIN.EPOCHS):
            t_start = time.time()
            _, c, train_accuracy, train_accuracy_back, train_instance_loss, train_existence_loss, train_summary, _ = \
                sess.run([train_op, total_loss, accuracy, accuracy_back, instance_loss, existence_loss, summary_op, out_logits_out],
                         feed_dict={phase: 'train'})

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)
            train_instance_loss_mean.append(train_instance_loss)
            train_existence_loss_mean.append(train_existence_loss)
            train_accuracy_mean.append(train_accuracy)
            train_accuracy_back_mean.append(train_accuracy_back)

            if epoch % CFG.TRAIN.DISPLAY_STEP == 0 or epoch<250:
                print(
                    'Epoch: {:d} loss_ins= {:6f} ({:6f}) loss_ext= {:6f} ({:6f}) accuracy= {:6f} ({:6f}) '
                    'accuracy_back= {:6f} ({:6f}) mean_time= {:5f}s '.format(epoch + 1, train_instance_loss,
                                                                             np.mean(train_instance_loss_mean),
                                                                             train_existence_loss,
                                                                             np.mean(train_existence_loss_mean),
                                                                             train_accuracy,
                                                                             np.mean(train_accuracy_mean),
                                                                             train_accuracy_back,
                                                                             np.mean(train_accuracy_back_mean),
                                                                             np.mean(train_cost_time_mean)))
            if epoch % 100 == 0:
                summary_writer.add_summary(summary=train_summary, global_step=epoch)

            if epoch % 500 == 0:
                train_cost_time_mean.clear()
                train_instance_loss_mean.clear()
                train_existence_loss_mean.clear()
                train_accuracy_mean.clear()
                train_accuracy_back_mean.clear()

            if epoch % 1000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

            if epoch >= 0:
                continue

            val_cost_time_mean = []
            val_instance_loss_mean = []
            val_existence_loss_mean = []
            val_accuracy_mean = []
            val_accuracy_back_mean = []
            val_IoU_mean = []

            for epoch_val in range(int(len(val_dataset) / CFG.TRAIN.VAL_BATCH_SIZE / CFG.TRAIN.GPU_NUM)):
                t_start_val = time.time()
                c_val, val_accuracy, val_accuracy_back, val_IoU, val_instance_loss, val_existence_loss = \
                    sess.run(
                        [val_op_total_loss, val_op_accuracy, val_op_accuracy_back,
                         val_op_IoU, val_op_instance_loss, val_op_existence_loss],
                        feed_dict={phase: 'test'})

                cost_time_val = time.time() - t_start_val
                val_cost_time_mean.append(cost_time_val)
                val_instance_loss_mean.append(val_instance_loss)
                val_existence_loss_mean.append(val_existence_loss)
                val_accuracy_mean.append(val_accuracy)
                val_accuracy_back_mean.append(val_accuracy_back)
                val_IoU_mean.append(val_IoU)

                if epoch_val % 1 == 0:
                    print('Epoch_Val: {:d} loss_ins= {:6f} ({:6f}) '
                          'loss_ext= {:6f} ({:6f}) accuracy= {:6f} ({:6f}) accuracy_back= {:6f} ({:6f}) '
                          'mIoU= {:6f} ({:6f}) mean_time= {:5f}s '.
                          format(epoch_val + 1, val_instance_loss, np.mean(val_instance_loss_mean), val_existence_loss,
                                 np.mean(val_existence_loss_mean), val_accuracy, np.mean(val_accuracy_mean),
                                 val_accuracy_back, np.mean(val_accuracy_back_mean), val_IoU, np.mean(val_IoU_mean),
                                 np.mean(val_cost_time_mean)))

            val_cost_time_mean.clear()
            val_instance_loss_mean.clear()
            val_existence_loss_mean.clear()
            val_accuracy_mean.clear()
            val_accuracy_back_mean.clear()
            val_IoU_mean.clear()


if __name__ == '__main__':
    # init args
    args = init_args()

    # train lanenet
    train_net(args.dataset_dir, args.version, args.weights_path, net_flag=args.net)
