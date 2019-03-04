#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import 
from collections import OrderedDict

import tensorflow as tf
from functools import partial
from encoder_decoder_model import cnn_basenet
from encoder_decoder_model.basemodel import resnet50, resnet_arg_scope, resnet_v1
from config import global_config
CFG = global_config.cfg
slim = tf.contrib.slim
resnet_arg_scope = partial(resnet_arg_scope, bn_trainable=CFG.RESNET.BN_TRAIN)

class ResEncoder(cnn_basenet.CNNBaseModel):
    """
    实现了一个基于MOBILE NET V1的特征编码类
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(ResEncoder, self).__init__()
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return self._phase== 'train'

    def _conv_stage(self, input_tensor, k_size, out_dims, name,
                    stride=1, pad='SAME'):
        """
        将卷积和激活封装在一起
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.conv2d(inputdata=input_tensor, out_channel=out_dims,
                               kernel_size=k_size, stride=stride,
                               use_bias=False, padding=pad, name='conv')

            bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def encode_and_decode(self, input_tensor, name, trainable=True):
        """Extract features from preprocessed inputs.

        Args:
          input_tensor: a [batch, height, width, channels] float tensor
            representing a batch of images.

        Returns:
          feature_maps: a list of tensors where the ith tensor has shape
            [batch, height_i, width_i, depth_i]
        """
        #input_tensor.get_shape().assert_has_rank(4)
        #shape_assert = tf.Assert(
        #    tf.logical_and(tf.greater_equal(tf.shape(input_tensor)[1], 33),
        #                   tf.greater_equal(tf.shape(input_tensor)[2], 33)),
        #    ['image size must at least be 33 in both height and width.'])

        #with tf.control_dependencies([shape_assert]):
        blocks = resnet50(input_tensor, self._is_training, bn_trainable=True)
        # line segmentation branch
        global_fms = []
        global_outs = []
        last_fm = None
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(name+'_line'):
          for i, block in enumerate(reversed(blocks)):
              with slim.arg_scope(resnet_arg_scope(bn_is_training=self._is_training)):
                  lateral = slim.conv2d(block, 64, [1, 1],
                      trainable=trainable, weights_initializer=initializer,
                      padding='SAME', activation_fn=tf.nn.relu,
                      scope='lateral/res{}'.format(5-i))

              if last_fm is not None:
                  sz = tf.shape(lateral)
                  upsample = tf.image.resize_bilinear(last_fm, (sz[1], sz[2]),
                      name='upsample/res{}'.format(5-i))
                  upsample = slim.conv2d(upsample, 64, [1, 1],
                      trainable=trainable, weights_initializer=initializer,
                      padding='SAME', activation_fn=None,
                      scope='merge/res{}'.format(5-i))
                  last_fm = upsample + lateral
              else:
                  last_fm = lateral

              # with slim.arg_scope(resnet_arg_scope(bn_is_training=self._is_training)):
              #     tmp = slim.conv2d(last_fm, 64, [1, 1],
              #         trainable=trainable, weights_initializer=initializer,
              #         padding='SAME', activation_fn=tf.nn.relu,
              #         scope='tmp/res{}'.format(5-i))
              #     out = slim.conv2d(tmp, 64, [3, 3],
              #         trainable=trainable, weights_initializer=initializer,
              #         padding='SAME', activation_fn=None,
              #         scope='pyramid/res{}'.format(5-i))
              global_fms.append(last_fm)
              global_outs.append(tf.image.resize_bilinear(last_fm, (CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH)))
              # global_outs.append(out)
          global_fms.reverse()
          global_outs.reverse()
          global_outs = tf.concat(global_outs,3)
          with slim.arg_scope(resnet_arg_scope(bn_is_training=self._is_training)):
                  line_out = slim.conv2d(global_outs, 5, [1, 1],
                      trainable=trainable, weights_initializer=initializer,
                      padding='SAME', activation_fn=None,
                      scope='final_out_line')

        # regress branch
        global_fms = []
        global_outs_reg = []
        last_fm = None
        with tf.variable_scope(name+'_reg'):
          for i, block in enumerate(reversed(blocks)):
              with slim.arg_scope(resnet_arg_scope(bn_is_training=self._is_training)):
                  lateral = slim.conv2d(block, 256, [1, 1],
                      trainable=trainable, weights_initializer=initializer,
                      padding='SAME', activation_fn=tf.nn.relu,
                      scope='lateral/res{}'.format(5-i))

              if last_fm is not None:
                  sz = tf.shape(lateral)
                  upsample = tf.image.resize_bilinear(last_fm, (sz[1], sz[2]),
                      name='upsample/res{}'.format(5-i))
                  upsample = slim.conv2d(upsample, 256, [1, 1],
                      trainable=trainable, weights_initializer=initializer,
                      padding='SAME', activation_fn=None,
                      scope='merge/res{}'.format(5-i))
                  last_fm = upsample + lateral
              else:
                  last_fm = lateral

              # with slim.arg_scope(resnet_arg_scope(bn_is_training=self._is_training)):
              #     tmp = slim.conv2d(last_fm, 256, [1, 1],
              #         trainable=trainable, weights_initializer=initializer,
              #         padding='SAME', activation_fn=tf.nn.relu,
              #         scope='tmp/res{}'.format(5-i))
              #     out = slim.conv2d(tmp, 64, [3, 3],
              #         trainable=trainable, weights_initializer=initializer,
              #         padding='SAME', activation_fn=None,
              #         scope='pyramid/res{}'.format(5-i))
              global_fms.append(last_fm)
              global_outs_reg.append(tf.image.resize_bilinear(last_fm, (CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH)))
              # global_outs.append(out)
          global_fms.reverse()
          global_outs_reg.reverse()
          global_outs_reg = tf.concat(global_outs_reg,axis=3)
          with slim.arg_scope(resnet_arg_scope(bn_is_training=self._is_training)):
                  reg_out = slim.conv2d(global_outs_reg, 2, [1, 1],
                      trainable=trainable, weights_initializer=initializer,
                      padding='SAME', activation_fn=None,
                      scope='final_out_reg')

        # lane seg branch
        global_fms = []
        global_outs_seg = []
        last_fm = None
        with tf.variable_scope(name+'_lane'):
          for i, block in enumerate(reversed(blocks)):
              with slim.arg_scope(resnet_arg_scope(bn_is_training=self._is_training)):
                  lateral = slim.conv2d(block, 256, [1, 1],
                      trainable=trainable, weights_initializer=initializer,
                      padding='SAME', activation_fn=tf.nn.relu,
                      scope='lateral/res{}'.format(5-i))

              if last_fm is not None:
                  sz = tf.shape(lateral)
                  upsample = tf.image.resize_bilinear(last_fm, (sz[1], sz[2]),
                      name='upsample/res{}'.format(5-i))
                  upsample = slim.conv2d(upsample, 256, [1, 1],
                      trainable=trainable, weights_initializer=initializer,
                      padding='SAME', activation_fn=None,
                      scope='merge/res{}'.format(5-i))
                  last_fm = upsample + lateral
              else:
                  last_fm = lateral

              # with slim.arg_scope(resnet_arg_scope(bn_is_training=self._is_training)):
              #     tmp = slim.conv2d(last_fm, 256, [1, 1],
              #         trainable=trainable, weights_initializer=initializer,
              #         padding='SAME', activation_fn=tf.nn.relu,
              #         scope='tmp/res{}'.format(5-i))
              #     out = slim.conv2d(tmp, 64, [3, 3],
              #         trainable=trainable, weights_initializer=initializer,
              #         padding='SAME', activation_fn=None,
              #         scope='pyramid/res{}'.format(5-i))
              global_fms.append(last_fm)
              global_outs_seg.append(tf.image.resize_bilinear(last_fm, (CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH)))
              # global_outs.append(out)
          global_fms.reverse()
          global_outs_seg.reverse()
          global_outs_seg = tf.concat(global_outs_seg,axis=3)
          with slim.arg_scope(resnet_arg_scope(bn_is_training=self._is_training)):
                  seg_out = slim.conv2d(global_outs_seg, 1, [1, 1],
                      trainable=trainable, weights_initializer=initializer,
                      padding='SAME', activation_fn=None,
                      scope='final_out_seg')

        ret = {}  
        ret['seg'] = seg_out
        ret['prob_output'] = line_out
        ret['reg'] = reg_out
        return ret

    def decode(self, input_tensor, name, trainable=True):
        """Extract features from preprocessed inputs.

        Args:
          input_tensor: a [batch, height, width, channels] float tensor
            representing a batch of images.

        Returns:
          feature_maps: a list of tensors where the ith tensor has shape
            [batch, height_i, width_i, depth_i]
        """
        #input_tensor.get_shape().assert_has_rank(4)
        #shape_assert = tf.Assert(
        #    tf.logical_and(tf.greater_equal(tf.shape(input_tensor)[1], 33),
        #                   tf.greater_equal(tf.shape(input_tensor)[2], 33)),
        #    ['image size must at least be 33 in both height and width.'])

        #with tf.control_dependencies([shape_assert]):
        blocks = resnet50(input_tensor, self._is_training, bn_trainable=True)
        
        # line segmentation branch
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(name+'_line'):
          upsample_1 = self.deconv2d(inputdata=blocks[-1], out_channel=64, kernel_size=4,
                                     stride=2, use_bias=False, name='deconv_1')
          # upsample = self._conv_stage(input_tensor=upsample, k_size=3,
          #                               out_dims=256, name='conv_1')
          upsample = self.deconv2d(inputdata=upsample_1, out_channel=64, kernel_size=4,
                                     stride=2, use_bias=False, name='deconv_2')
          # upsample = self._conv_stage(input_tensor=upsample, k_size=3,
          #                               out_dims=256, name='conv_2')
          outs_line = self.deconv2d(inputdata=upsample, out_channel=64, kernel_size=4,
                                     stride=2, use_bias=False, name='deconv_3')
          with slim.arg_scope(resnet_arg_scope(bn_is_training=self._is_training)):
            line_out = slim.conv2d(outs_line, 5, [1, 1],
                        trainable=trainable, weights_initializer=initializer,
                        padding='SAME', activation_fn=None,
                        scope='final_out_line')
          features = upsample_1  # N x H x W x C
          softmax = tf.nn.softmax(features)

          avg_pool = self.avgpooling(softmax, kernel_size=2, stride=2)
          _, H, W, C = avg_pool.get_shape().as_list()
          reshape_output = tf.reshape(avg_pool, [-1, H * W * C])
          fc_output = self.fullyconnect(reshape_output, 128)
          relu_output = self.relu(inputdata=fc_output, name='relu6')
          fc_output = self.fullyconnect(relu_output, 4)
          existence_output = fc_output


        # regress branch
        with tf.variable_scope(name+'_reg'):
          upsample = self.deconv2d(inputdata=blocks[-1], out_channel=64, kernel_size=4,
                                     stride=2, use_bias=False, name='deconv_1')
          # upsample = self._conv_stage(input_tensor=upsample, k_size=3,
          #                               out_dims=256, name='conv_1')
          upsample = self.deconv2d(inputdata=upsample, out_channel=64, kernel_size=4,
                                     stride=2, use_bias=False, name='deconv_2')
          # upsample = self._conv_stage(input_tensor=upsample, k_size=3,
          #                               out_dims=256, name='conv_2')
          outs_reg = self.deconv2d(inputdata=upsample, out_channel=64, kernel_size=4,
                                     stride=2, use_bias=False, name='deconv_3')
          with slim.arg_scope(resnet_arg_scope(bn_is_training=self._is_training)):
                  reg_out = slim.conv2d(outs_reg, 2, [1, 1],
                      trainable=trainable, weights_initializer=initializer,
                      padding='SAME', activation_fn=None,
                      scope='final_out_reg')

        # lane seg branch
        with tf.variable_scope(name+'_seg'):
          upsample = self.deconv2d(inputdata=blocks[-1], out_channel=64, kernel_size=4,
                                     stride=2, use_bias=False, name='deconv_1')
          # upsample = self._conv_stage(input_tensor=upsample, k_size=3,
          #                               out_dims=256, name='conv_1')
          upsample = self.deconv2d(inputdata=upsample, out_channel=64, kernel_size=4,
                                     stride=2, use_bias=False, name='deconv_2')
          # upsample = self._conv_stage(input_tensor=upsample, k_size=3,
          #                               out_dims=256, name='conv_2')
          outs_seg = self.deconv2d(inputdata=upsample, out_channel=64, kernel_size=4,
                                     stride=2, use_bias=False, name='deconv_3')
          with slim.arg_scope(resnet_arg_scope(bn_is_training=self._is_training)):
                  seg_out = slim.conv2d(outs_seg, 1, [1, 1],
                      trainable=trainable, weights_initializer=initializer,
                      padding='SAME', activation_fn=None,
                      scope='final_out_seg')
                  # ret['lane_reg'] = tf.image.resize_images(conv_output_3, [IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH])
        IMG_HEIGHT =  CFG.TRAIN.IMG_HEIGHT*2//3
        ret = {}  
        ret['lane_seg'] = tf.image.resize_images(seg_out, [IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH])
        ret['prob_output'] = tf.image.resize_images(line_out, [IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH])
        ret['existence_output'] = existence_output
        ret['lane_reg'] = tf.image.resize_images(reg_out, [IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH])
        # import pdb;pdb.set_trace()
        return ret


if __name__ == '__main__':
    a = tf.placeholder(dtype=tf.float32, shape=[1, 288, 800, 3], name='input')
    encoder = ResEncoder(phase=tf.constant('train', dtype=tf.string))
    # ret = encoder.encode_and_decode(a, name='decode')
    ret = encoder.decode(a, name='decode')
    import pdb;pdb.set_trace()
