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

from encoder_decoder_model import vgg_encoder
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
    # slice_feature_maps = tf.strided_slice(feature_maps,
    #         begin=[0, ybeg, xbeg, 0],
    #         end=[size[0], ybeg+ysize, xbeg+xsize, size[3]])
    slice_feature_maps = tf.slice(feature_maps,
           begin=[0, ybeg, xbeg, 0],
           size=[-1, ysize, xsize, -1])

    return slice_feature_maps

def _regress_loss_new(prediction, left_gt, right_gt, mask, name=None):
    with tf.variable_scope(name + '/regress_loss'):
        # det_gt_mask_l = tf.cast(tf.greater(gt[0],0), tf.int32)
        prediction = tf.cast(prediction, tf.float32)

        left_gt = _slice_feature(tf.expand_dims(left_gt, 3))
        left_gt = tf.cast(left_gt, tf.float32)/250.0

        right_gt = _slice_feature(tf.expand_dims(right_gt, 3))
        right_gt = tf.cast(right_gt, tf.float32)/250.0

        # line_gt = self._slice_feature_(tf.expand_dims(gt[2], 3))
        # line_gt = tf.squeeze(line_gt, axis=[3])
        # det_gt_mask_line = tf.cast(tf.greater(line_gt,0), tf.float32)
        mask = _slice_feature(tf.expand_dims(mask, 3))
        mask = tf.cast(tf.squeeze(mask, axis=[3]), tf.float32)

        left_prediction = prediction[:,:,:,0]*mask+0.001#*det_gt_mask_l
        right_prediction = prediction[:,:,:,1]*mask+0.001#*det_gt_mask_r
        # line_prediction = prediction[:,:,:,2]#*det_gt_mask_line
        # tf.summary.image(self._name+'/lab_left_gt', tf.cast(tf.expand_dims(left_gt*200, 3), tf.uint8), 2)
        # tf.summary.image(self._name+'/lab_right_gt', tf.cast(tf.expand_dims(right_gt*200, 3), tf.uint8), 2)
        # tf.summary.image(self._name+'/lab_line_gt', tf.cast(tf.expand_dims(line_gt*200, 3), tf.uint8), 2)
        # import pdb;pdb.set_trace()
        tf.summary.image(name+'/lab_left_gt', tf.concat(axis=2,
              values=[tf.cast(left_gt*250, tf.uint8), tf.cast(tf.expand_dims(left_prediction*250, 3), tf.uint8)]
              ), 1)
        tf.summary.image(name+'/lab_right_gt', tf.concat(axis=2,
              values=[tf.cast(right_gt*250, tf.uint8), tf.cast(tf.expand_dims(right_prediction*250, 3), tf.uint8)]
              ), 1)
        # tf.summary.image(self._name+'/lab_line_gt', tf.concat(axis=2,
        #       values=[tf.cast(tf.expand_dims(line_gt*50, 3), tf.uint8), tf.cast(tf.expand_dims(line_prediction*50, 3), tf.uint8)]
        #       ), 2)
        # left_prediction = left_prediction*mask
        # right_prediction = right_prediction*mask
        left_gt = tf.squeeze(left_gt)+0.001
        right_gt = tf.squeeze(right_gt)+0.001

        min_dis_l = tf.minimum(left_prediction,left_gt)
        min_dis_r = tf.minimum(right_prediction,right_gt)
        max_dis_l = tf.maximum(left_prediction,left_gt)
        max_dis_r = tf.maximum(right_prediction,right_gt)
        value = (min_dis_l+min_dis_r)/(max_dis_l+max_dis_r)
        mat = 1.0 - value
        tf.summary.image(name+'/reg_mat', tf.cast(tf.expand_dims(mat*255, 3), tf.uint8), 1)
        raw_loss_reg = tf.reduce_mean(mat)
        # import pdb;pdb.set_trace()
        # raw_loss_reg = tf.Print(raw_loss_reg,[tf.reduce_max(left_gt),tf.reduce_max(right_gt)],'gt:')
        # raw_loss_reg = tf.Print(raw_loss_reg,[tf.reduce_max(left_prediction),tf.reduce_max(right_prediction)],'pred:')
        return raw_loss_reg

def _seg_loss_gauss(prediction, gt, name=None):
    with tf.variable_scope(name + '/binary_loss'):
      feature_size = prediction.get_shape().as_list()[0:3]
      prediction = tf.nn.sigmoid(tf.cast(prediction, tf.float32))
      gt = tf.cast(gt, tf.float32)

      gt = _slice_feature(tf.expand_dims(gt, 3))

      tf.summary.image(name+'/gauss_gt', tf.concat(axis=2,
          values=[ tf.cast(gt, tf.uint8), tf.cast(prediction*250, tf.uint8)]
          ), 1)
      prediction = tf.squeeze(prediction)
      gt = tf.squeeze(gt)
      raw_loss = tf.square(prediction - gt/250)
      index = tf.cast(tf.greater(gt,0),tf.float32)
      tmp = raw_loss*index*19.0
      raw_loss = raw_loss + tmp
      raw = tf.reduce_mean(raw_loss)
      
      loss_plus = tf.slice(raw_loss,
                    begin=[0, 0, 0],
                    size=[feature_size[0], feature_size[1]//2, feature_size[2]])
      plus = tf.reduce_mean(loss_plus)
      return raw+plus

def _seg_loss_hard(prediction, images, gt, name, aux_loss_type=1):
    with tf.variable_scope(name + '/seg_loss'):
      feature_size = prediction.get_shape().as_list()
      gt = tf.expand_dims(gt,3)
      gt = tf.cast(gt, tf.int32)
      gt = _slice_feature(gt)

      seg_out = tf.expand_dims(tf.argmax(prediction, axis=3),axis=3)
      tf.summary.image(name+'/lab_gt', tf.concat(axis=2,
          values=[ tf.cast(gt, tf.uint8)*(255//4), tf.cast(seg_out, tf.uint8)*(255//4)]
          ), 1)

      # prediction_reshape = tf.reshape(
      #           prediction,
      #           shape=[feature_size[0],
      #                  feature_size[1] * feature_size[2],
      #                  feature_size[3]])
      # import pdb; pdb.set_trace()
      gt_reshape = tf.reshape(
                gt,
                shape=[feature_size[0],
                       feature_size[1], 
                       feature_size[2]])
      gt_reshape = tf.one_hot(gt_reshape, depth=5)
      class_weights = tf.constant([[0.4, 1.0, 1.0, 1.0, 1.0]])
      weights_loss = tf.reduce_sum(tf.multiply(gt_reshape, class_weights), 3)
      # raw_loss = tf.losses.softmax_cross_entropy(onehot_labels=gt_reshape,
                                                 # logits=prediction_reshape,
                                                 # weights=weights_loss)
      gt = tf.reshape(gt, [-1])
      prediction = tf.reshape(prediction, [-1, 5])
      raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
      raw_loss = tf.reshape(raw_loss,feature_size[0:3])
      raw_loss = tf.multiply(raw_loss,weights_loss)
      if aux_loss_type==1:
        region_size=10
        half_region_size = region_size//2
        #images = [tf.image.resize_nearest_neighbor(tf.expand_dims(x,0), size) for x in images]
        images = tf.cast(images, tf.float32)
        #images = tf.squeeze(images, axis=[1])
        images = _slice_feature(images)
        B_img = images[:,:,:,0]
        G_img = images[:,:,:,1]
        R_img = images[:,:,:,2]
        gray_img = tf.multiply(tf.cast(R_img,tf.float32), 0.299) + tf.multiply(tf.cast(G_img,tf.float32), 0.587) +  tf.multiply(tf.cast(B_img,tf.float32), 0.114)
        gt = tf.reshape(gt,feature_size[0:3])
        loss_add = []
        # import pdb; pdb.set_trace()
        for y in range(feature_size[1]):
          min_y = y-half_region_size if y>half_region_size else 0
          max_y = y+half_region_size if y+half_region_size<feature_size[1] else feature_size[1]
          label_region = gt[:, min_y:max_y,:]
          label_centre = gt[:, y, :]
          label_centre = tf.expand_dims(label_centre, 1)
          label_centre = tf.tile(label_centre, [1,max_y-min_y,1])
          lab_indices = tf.cast(tf.equal(label_region, label_centre), tf.float32)
          gray_img_y = tf.expand_dims(gray_img[:, y, :], 1)
          gray_img_y = tf.tile(gray_img_y, [1,max_y-min_y,1])
          grey_diff = gray_img_y - gray_img[:, min_y:max_y,:] + 0.001
          loss_weight_mat = tf.add(tf.multiply(lab_indices,tf.exp(-1.0/(tf.abs(grey_diff)))),tf.multiply((1-lab_indices),tf.exp(-tf.abs(grey_diff))))
          loss_weight = tf.reduce_sum(loss_weight_mat) + 0.001
          loss_weight_mat /= loss_weight
          loss_add_value = tf.reduce_sum(raw_loss[:,min_y:max_y,:]*loss_weight_mat)
          loss_add.append(loss_add_value)

        # import pdb; pdb.set_trace()
        for x in range(feature_size[2]):
          min_x = x-half_region_size if x>half_region_size else 0
          max_x = x+half_region_size if x+half_region_size<feature_size[2] else feature_size[2]
          label_region = gt[:, :, min_x:max_x]
          label_centre = gt[:, :, x]
          label_centre = tf.expand_dims(label_centre, 2)
          label_centre = tf.tile(label_centre, [1,1,max_x-min_x])
          lab_indices = tf.cast(tf.equal(label_region, label_centre), tf.float32)
          gray_img_x = tf.expand_dims(gray_img[:, :, x], 2)
          gray_img_x = tf.tile(gray_img_x, [1,1,max_x-min_x])
          grey_diff = gray_img_x - gray_img[:, :, min_x:max_x] + 0.001
          loss_weight_mat = tf.add(tf.multiply(lab_indices,tf.exp(-1.0/(tf.abs(grey_diff)))),tf.multiply((1-lab_indices),tf.exp(-tf.abs(grey_diff))))
          loss_weight = tf.reduce_sum(loss_weight_mat) + 0.001
          loss_weight_mat /= loss_weight
          loss_add_value = tf.reduce_sum(raw_loss[:,:,min_x:max_x]*loss_weight_mat)
          loss_add.append(loss_add_value)
        # import pdb; pdb.set_trace()
        raw_loss_hard = tf.stack(loss_add)
        # raw_loss_hard = tf.Print(raw_loss_hard ,[raw_loss_hard.shape],message='tips:',summarize=100)
        raw_loss_hard = tf.reduce_mean(raw_loss_hard) 
        
      else:
        raw_loss_hard = 0
      raw_loss = tf.reduce_mean(raw_loss)
      return raw_loss + raw_loss_hard

def _seg_loss_hard_lane(prediction, images, gt, name, aux_loss_type=1):
    with tf.variable_scope(name + '/seg_loss'):
      feature_size = prediction.get_shape().as_list()
      gt = tf.expand_dims(gt,3)
      gt = tf.cast(gt, tf.int32)
      gt = _slice_feature(gt)

      seg_out = tf.expand_dims(tf.argmax(prediction, axis=3),axis=3)
      tf.summary.image(name+'/lab_gt', tf.concat(axis=2,
          values=[ tf.cast(gt, tf.uint8)*(255//4), tf.cast(seg_out, tf.uint8)*(255//4)]
          ), 1)

      # prediction_reshape = tf.reshape(
      #           prediction,
      #           shape=[feature_size[0],
      #                  feature_size[1] * feature_size[2],
      #                  feature_size[3]])
      # import pdb; pdb.set_trace()
      gt_reshape = tf.reshape(
                gt,
                shape=[feature_size[0],
                       feature_size[1], 
                       feature_size[2]])
      gt_reshape = tf.one_hot(gt_reshape, depth=5)
      class_weights = tf.constant([[0.4, 1.0, 1.0, 1.0, 1.0]])
      weights_loss = tf.reduce_sum(tf.multiply(gt_reshape, class_weights), 3)
      # raw_loss = tf.losses.softmax_cross_entropy(onehot_labels=gt_reshape,
                                                 # logits=prediction_reshape,
                                                 # weights=weights_loss)
      gt = tf.reshape(gt, [-1])
      prediction = tf.reshape(prediction, [-1, 5])
      raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
      raw_loss = tf.reshape(raw_loss,feature_size[0:3])
      raw_loss = tf.multiply(raw_loss,weights_loss)
      if aux_loss_type==1:
        region_size=10
        half_region_size = region_size//2
        #images = [tf.image.resize_nearest_neighbor(tf.expand_dims(x,0), size) for x in images]
        images = tf.cast(images, tf.float32)
        #images = tf.squeeze(images, axis=[1])
        images = _slice_feature(images)
        B_img = images[:,:,:,0]
        G_img = images[:,:,:,1]
        R_img = images[:,:,:,2]
        gray_img = tf.multiply(tf.cast(R_img,tf.float32), 0.299) + tf.multiply(tf.cast(G_img,tf.float32), 0.587) +  tf.multiply(tf.cast(B_img,tf.float32), 0.114)
        gt = tf.reshape(gt,feature_size[0:3])
        loss_add = []
        # import pdb; pdb.set_trace()
        for y in range(feature_size[1]):
          min_y = y-half_region_size if y>half_region_size else 0
          max_y = y+half_region_size if y+half_region_size<feature_size[1] else feature_size[1]
          label_region = gt[:, min_y:max_y,:]
          label_centre = gt[:, y, :]
          label_centre = tf.expand_dims(label_centre, 1)
          label_centre = tf.tile(label_centre, [1,max_y-min_y,1])
          lab_indices = tf.cast(tf.equal(label_region, label_centre), tf.float32)
          gray_img_y = tf.expand_dims(gray_img[:, y, :], 1)
          gray_img_y = tf.tile(gray_img_y, [1,max_y-min_y,1])
          grey_diff = gray_img_y - gray_img[:, min_y:max_y,:] + 0.001
          loss_weight_mat = tf.add(tf.multiply(lab_indices,tf.exp(-1.0/(tf.abs(grey_diff)))),tf.multiply((1-lab_indices),tf.exp(-tf.abs(grey_diff))))
          loss_weight = tf.reduce_sum(loss_weight_mat) + 0.001
          loss_weight_mat /= loss_weight
          loss_add_value = tf.reduce_sum(raw_loss[:,min_y:max_y,:]*loss_weight_mat)
          loss_add.append(loss_add_value)

        # import pdb; pdb.set_trace()
        for x in range(feature_size[2]):
          min_x = x-half_region_size if x>half_region_size else 0
          max_x = x+half_region_size if x+half_region_size<feature_size[2] else feature_size[2]
          label_region = gt[:, :, min_x:max_x]
          label_centre = gt[:, :, x]
          label_centre = tf.expand_dims(label_centre, 2)
          label_centre = tf.tile(label_centre, [1,1,max_x-min_x])
          lab_indices = tf.cast(tf.equal(label_region, label_centre), tf.float32)
          gray_img_x = tf.expand_dims(gray_img[:, :, x], 2)
          gray_img_x = tf.tile(gray_img_x, [1,1,max_x-min_x])
          grey_diff = gray_img_x - gray_img[:, :, min_x:max_x] + 0.001
          loss_weight_mat = tf.add(tf.multiply(lab_indices,tf.exp(-1.0/(tf.abs(grey_diff)))),tf.multiply((1-lab_indices),tf.exp(-tf.abs(grey_diff))))
          loss_weight = tf.reduce_sum(loss_weight_mat) + 0.001
          loss_weight_mat /= loss_weight
          loss_add_value = tf.reduce_sum(raw_loss[:,:,min_x:max_x]*loss_weight_mat)
          loss_add.append(loss_add_value)
        # import pdb; pdb.set_trace()
        raw_loss_hard = tf.stack(loss_add)
        # raw_loss_hard = tf.Print(raw_loss_hard ,[raw_loss_hard.shape],message='tips:',summarize=100)
        raw_loss_hard = tf.reduce_mean(raw_loss_hard) 
        
      else:
        raw_loss_hard = 0
      raw_loss = tf.reduce_mean(raw_loss)
      return raw_loss + raw_loss_hard

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
        with tf.variable_scope(name):
            with tf.variable_scope('inference'):
                encoder = vgg_encoder.VGG16Encoder(phase=phase)
                # import pdb;pdb.set_trace()
                input_tensor = _slice_feature(input_tensor)
                encode_ret = encoder.encode_re(input_tensor=input_tensor, name='encode')

            return encode_ret

    @staticmethod
    def test_inference(input_tensor, phase, name):
        inference_ret = LaneNet.inference(input_tensor, phase, name)
        with tf.variable_scope(name):
            # feed forward to obtain logits
            # Compute loss

            decode_logits = inference_ret['prob_output']
            padd = tf.zeros([8,CFG.TRAIN.IMG_HEIGHT//3, CFG.TRAIN.IMG_WIDTH,5])
            decode_logits = tf.concat([padd,decode_logits],1)
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

            lane_seg = inference_ret['lane_seg']
            feature_for_seg = tf.squeeze(tf.nn.sigmoid(lane_seg),3)
            feature_for_reg = inference_ret['lane_reg']
            # import pdb;pdb.set_trace()
            feature_for_reg = tf.nn.sigmoid(tf.cast(feature_for_reg, tf.float32))

            return binary_seg_ret, existence_output, feature_for_seg, feature_for_reg

    @staticmethod
    def loss(inference, binary_label, existence_label, lane_binary, lane_lmap, lane_rmap, images, name):
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
            inference_ret = inference
            binary_segmentation_loss = 1
            existence_loss = 1

            # # Compute the segmentation loss

            decode_logits = inference_ret['prob_output']

            binary_segmentation_loss = _seg_loss_hard(decode_logits, images, binary_label, 'line', aux_loss_type=0)

            # # Compute the HSP loss of line
            # hard_line_loss = _seg_loss_hard(decode_logits, images, binary_label, 'hard_line')

            # # Compute the sigmoid loss

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

        # total_loss = binary_segmentation_loss + 0.1*existence_loss + 0.5*hard_line_loss
        total_loss = 0.1*lane_segmentation_loss + lane_regress_loss + 0.1*binary_segmentation_loss
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
        # tf.add_to_collection('lane_hard_loss',hard_line_loss)

        return ret
