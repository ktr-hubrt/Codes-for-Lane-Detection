#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午5:28
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_merge_model.py
# @IDE: PyCharm Community Edition
"""
Build Lane detection model
"""
import tensorflow as tf

from encoder_decoder_model import vgg_encoder,res_encoder
from encoder_decoder_model import cnn_basenet
# from lanenet_model import LaneDetPredictor
from config import global_config

CFG = global_config.cfg

def _slice_feature(feature_maps):
    _x_bin = 1
    _xbegin = 0
    _xsize = 1
    _y_bin = 3
    _ybegin = 1
    _ysize = 2
    if _x_bin==1 and _xbegin==0 and _xsize==1 \
        and _y_bin==1 and _ybegin==0 and _ysize==1:
      return feature_maps
    size = feature_maps.shape.as_list()[1:3]
    assert size[0] % _y_bin == 0
    assert size[1] % _x_bin == 0

    ybin = size[0] // _y_bin
    ybeg = _ybegin * ybin
    ysize = _ysize * ybin

    xbin = size[1] // _x_bin
    xbeg = _xbegin * xbin
    xsize = _xsize * xbin
    # import pdb;pdb.set_trace()
    size = feature_maps.shape.as_list()
    slice_feature_maps = tf.strided_slice(feature_maps,
            begin=[0, ybeg, xbeg, 0],
            end=[size[0], ybeg+ysize, xbeg+xsize, size[3]])
    #slice_feature_maps = tf.slice(feature_maps,
    #        begin=[0, ybeg, xbeg, 0],
    #        size=[-1, ysize, xsize, -1])

    return slice_feature_maps

def _regress_loss_new(prediction, left_gt, right_gt, mask, name=None):
    with tf.variable_scope(name + '/regress_loss'):
        # det_gt_mask_l = tf.cast(tf.greater(gt[0],0), tf.int32)
        prediction = tf.nn.sigmoid(tf.cast(prediction, tf.float32))
        prediction = tf.cast(prediction, tf.float32)

        left_gt = _slice_feature(tf.expand_dims(left_gt, 3))
        left_gt = tf.cast(left_gt, tf.float32)/250

        right_gt = _slice_feature(tf.expand_dims(right_gt, 3))
        right_gt = tf.cast(right_gt, tf.float32)/250

        # line_gt = self._slice_feature_(tf.expand_dims(gt[2], 3))
        # line_gt = tf.squeeze(line_gt, axis=[3])
        # det_gt_mask_line = tf.cast(tf.greater(line_gt,0), tf.float32)
        mask = _slice_feature(tf.expand_dims(mask, 3))
        mask = tf.cast(tf.squeeze(mask, axis=[3]), tf.float32)

        left_prediction = prediction[:,:,:,0]*mask#*det_gt_mask_l
        right_prediction = prediction[:,:,:,1]*mask#*det_gt_mask_r
        # line_prediction = prediction[:,:,:,2]#*det_gt_mask_line
        # tf.summary.image(self._name+'/lab_left_gt', tf.cast(tf.expand_dims(left_gt*200, 3), tf.uint8), 2)
        # tf.summary.image(self._name+'/lab_right_gt', tf.cast(tf.expand_dims(right_gt*200, 3), tf.uint8), 2)
        # tf.summary.image(self._name+'/lab_line_gt', tf.cast(tf.expand_dims(line_gt*200, 3), tf.uint8), 2)
        # import pdb;pdb.set_trace()
        tf.summary.image(name+'/lab_left_gt', tf.concat(axis=2,
              values=[tf.cast(left_gt*200, tf.uint8), tf.cast(tf.expand_dims(left_prediction*200, 3), tf.uint8)]
              ), 1)
        tf.summary.image(name+'/lab_right_gt', tf.concat(axis=2,
              values=[tf.cast(right_gt*200, tf.uint8), tf.cast(tf.expand_dims(right_prediction*200, 3), tf.uint8)]
              ), 1)
        # tf.summary.image(self._name+'/lab_line_gt', tf.concat(axis=2,
        #       values=[tf.cast(tf.expand_dims(line_gt*50, 3), tf.uint8), tf.cast(tf.expand_dims(line_prediction*50, 3), tf.uint8)]
        #       ), 2)
        # left_prediction = left_prediction*mask
        # right_prediction = right_prediction*mask
        left_gt = tf.squeeze(left_gt)
        right_gt = tf.squeeze(right_gt)
        indices_l = tf.cast(tf.equal(left_prediction,left_gt), tf.float32)
        indices_r = tf.cast(tf.equal(right_prediction, right_gt), tf.float32)
        min_dis_l = tf.minimum(left_prediction,left_gt)+indices_l
        min_dis_r = tf.minimum(right_prediction,right_gt)+indices_r
        max_dis_l = tf.maximum(left_prediction,left_gt)+indices_l
        max_dis_r = tf.maximum(right_prediction,right_gt)+indices_r
        value = (min_dis_l+min_dis_r)/(max_dis_l+max_dis_r)
        mat = 1.0 - value
        tf.summary.image(name+'/reg_mat', tf.cast(tf.expand_dims(mat*255, 3), tf.uint8), 1)
        raw_loss_reg = tf.reduce_mean(mat)
        # import pdb;pdb.set_trace()
        return raw_loss_reg

def _seg_loss_gauss(prediction, gt, name=None):
    with tf.variable_scope(name + '/binary_loss'):
        feature_size = prediction.get_shape().as_list()[0:3]
        prediction = tf.nn.sigmoid(tf.cast(prediction, tf.float32))
        gt = tf.cast(gt, tf.float32)

        gt = _slice_feature(tf.expand_dims(gt, 3))
        # import pdb;pdb.set_trace()
        tf.summary.image(name+'/gauss_gt', tf.concat(axis=2,
          values=[ tf.cast(gt, tf.uint8), tf.cast(prediction*255, tf.uint8)]
          ), 1)
        prediction = tf.squeeze(prediction)
        gt = tf.squeeze(gt)
        raw_loss = tf.losses.mean_squared_error(prediction, gt/250)*20

        half_prediction = tf.slice(prediction,
                    begin=[0, 0, 0],
                    size=[feature_size[0], feature_size[1]//2, feature_size[2]])
        half_gt = tf.slice(gt,
                    begin=[0, 0, 0],
                    size=[feature_size[0], feature_size[1]//2, feature_size[2]])
        loss_plus = tf.losses.mean_squared_error(half_prediction, half_gt/250)*20
        return raw_loss+loss_plus

class LaneNet(cnn_basenet.CNNBaseModel):
    """
    Lane detection model
    """


    @staticmethod
    def inference(input_tensor, phase, name):
        """
        feed forward
        :param name:
        :param input_tensor:
        :param phase:
        :return:
        """
        # with tf.variable_scope(name):
        #     with tf.variable_scope('inference'):
        #         encoder = vgg_encoder.VGG16Encoder(phase=phase)
        #         # import pdb;pdb.set_trace()
        #         input_tensor = _slice_feature(input_tensor)
        #         encode_ret = encoder.encode_re(input_tensor=input_tensor, name='encode')
        #         return encode_ret
        encoder = res_encoder.ResEncoder(phase=phase)
        # import pdb;pdb.set_trace()
        input_tensor = _slice_feature(input_tensor)
        # encode_ret = encoder.encode_re(input_tensor=input_tensor, name='encode')
        encode_ret = encoder.decode(input_tensor=input_tensor, name='encode')
        return encode_ret

    @staticmethod
    def test_inference(input_tensor, phase, name):
        inference_ret = LaneNet.inference(input_tensor, phase, name)
        with tf.variable_scope(name):
            # feed forward to obtain logits
            # Compute loss

            decode_logits = inference_ret['prob_output']
            binary_seg_ret = tf.nn.softmax(logits=decode_logits)
            prob_list = []
            kernel = tf.get_variable('kernel', [9, 9, 1, 1], initializer=tf.constant_initializer(1.0 / 81),
                                     trainable=False)

            with tf.variable_scope("convs_smooth"):
                prob_smooth = tf.nn.conv2d(tf.cast(tf.expand_dims(binary_seg_ret[:, :, :, 0], axis=3), tf.float32),
                                           kernel, [1, 1, 1, 1], 'SAME')
                prob_list.append(prob_smooth)

            for cnt in range(1, binary_seg_ret.get_shape().as_list()[3]):
                with tf.variable_scope("convs_smooth", reuse=True):
                    prob_smooth = tf.nn.conv2d(
                        tf.cast(tf.expand_dims(binary_seg_ret[:, :, :, cnt], axis=3), tf.float32), kernel, [1, 1, 1, 1],
                        'SAME')
                    prob_list.append(prob_smooth)
            processed_prob = tf.stack(prob_list, axis=4)
            processed_prob = tf.squeeze(processed_prob, axis=3)
            binary_seg_ret = processed_prob

            # Predict lane existence:
            existence_logit = inference_ret['existence_output']
            existence_output = tf.nn.sigmoid(existence_logit)

            return binary_seg_ret, existence_output

    @staticmethod
    def loss(inference, binary_label, existence_label, lane_binary, lane_lmap, lane_rmap, name):
        """
        :param name:
        :param inference:
        :param existence_label:
        :param binary_label:
        :return:
        """
        # feed forward to obtain logits

        with tf.variable_scope(name):
            # import pdb;pdb.set_trace()
            binary_label = tf.expand_dims(binary_label,3)
            binary_label = _slice_feature(binary_label)
            
            # import pdb;pdb.set_trace()
            inference_ret = inference

            # Compute the segmentation loss

            decode_logits = inference_ret['prob_output']
            decode_logits_reshape = tf.reshape(
                decode_logits,
                shape=[decode_logits.get_shape().as_list()[0],
                       decode_logits.get_shape().as_list()[1] * decode_logits.get_shape().as_list()[2],
                       decode_logits.get_shape().as_list()[3]])
            # import pdb;pdb.set_trace()
            tf.summary.image(name+'/line_gt', tf.concat(axis=2,
                          values=[ tf.cast(binary_label*50, tf.uint8), tf.cast(tf.expand_dims(tf.argmax(decode_logits,axis=3)*50,3), tf.uint8)]
                          ), 1)
            binary_label = tf.squeeze(binary_label)
            binary_label_reshape = tf.reshape(
                binary_label,
                shape=[binary_label.get_shape().as_list()[0],
                       binary_label.get_shape().as_list()[1] * binary_label.get_shape().as_list()[2]])
            binary_label_reshape = tf.one_hot(binary_label_reshape, depth=5)
            class_weights = tf.constant([[0.4, 1.0, 1.0, 1.0, 1.0]])
            weights_loss = tf.reduce_sum(tf.multiply(binary_label_reshape, class_weights), 2)
            binary_segmentation_loss = tf.losses.softmax_cross_entropy(onehot_labels=binary_label_reshape,
                                                                       logits=decode_logits_reshape,
                                                                       weights=weights_loss)
            binary_segmentation_loss = tf.reduce_mean(binary_segmentation_loss)

            # Compute the sigmoid loss

            existence_logits = inference_ret['existence_output']
            existence_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=existence_label, logits=existence_logits)
            existence_loss = tf.reduce_mean(existence_loss)

            # Compute the lane segmentation loss
            lane_logits = inference_ret['lane_seg']
            if 1:
                # import pdb;pdb.set_trace()
                mask = tf.cast(tf.greater(lane_lmap,0),tf.float32)
                lane_binary = mask *tf.cast(lane_binary,tf.float32)
            lane_segmentation_loss = _seg_loss_gauss(lane_logits, lane_binary, 'lanedet')

            # Compute the lane regression loss
            lane_dismap = inference_ret['lane_reg']
            lane_regress_loss = _regress_loss_new(lane_dismap, lane_lmap, lane_rmap, lane_binary, 'lanedet')
            # import pdb;pdb.set_trace()
        # Compute the overall loss

        total_loss = 10 * lane_regress_loss + 0.1 * lane_segmentation_loss + 0.06 *binary_segmentation_loss +0.01*existence_loss
        ret = {
            'total_loss': total_loss,
            'instance_seg_logits': decode_logits,
            'instance_seg_loss': binary_segmentation_loss,
            'existence_logits': existence_logits,
            'existence_pre_loss': existence_loss,
            'lane_seg_loss': lane_segmentation_loss,
            'lane_reg_loss': lane_regress_loss,
        }
        # import pdb;pdb.set_trace()
        padd = tf.zeros([CFG.TRAIN.BATCH_SIZE,CFG.TRAIN.IMG_HEIGHT//3, CFG.TRAIN.IMG_WIDTH,5])
        decode_logits = tf.concat([padd,decode_logits],1)
        tf.add_to_collection('total_loss', total_loss)
        tf.add_to_collection('instance_seg_logits', decode_logits)
        tf.add_to_collection('instance_seg_loss', binary_segmentation_loss)
        tf.add_to_collection('existence_logits', existence_logits)
        tf.add_to_collection('existence_pre_loss',existence_loss)
        tf.add_to_collection('lane_seg_loss',lane_segmentation_loss)
        tf.add_to_collection('lane_reg_loss',lane_regress_loss)

        return ret
