# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import math
import numpy as np
beta=2.

def placeholder_inputs(batch_size=16,num_frame_per_clib=16, crop_size=224, rgb_channels=3, flow_channels=2):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
    batch_size: The batch size will be baked into both placeholders.
    num_frame_per_clib: The num of frame per clib.
    crop_size: The crop size of per clib.
    channels: The input channel of per clib.

    Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    rgb_images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           num_frame_per_clib,
                                                           crop_size,
                                                           crop_size,
                                                           rgb_channels))
    flow_images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           num_frame_per_clib,
                                                           crop_size,
                                                           crop_size,
                                                           flow_channels))
    
    labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size
                                                         ))
    
    # test_rgb_placeholder = tf.placeholder(tf.float32, shape=(test_size,
    #                                                         num_frame_per_clib,
    #                                                         crop_size,
    #                                                         crop_size,
    #                                                         rgb_channels))
    # test_flow_placeholder = tf.placeholder(tf.float32, shape=(test_size,
    #                                                         num_frame_per_clib,
    #                                                         crop_size,
    #                                                         crop_size,
    #                                                         flow_channels))
    # test_labels_placeholder = tf.placeholder(tf.int64, shape=(test_size
                                                          # ))
    is_training = tf.placeholder(tf.bool)
    return rgb_images_placeholder, flow_images_placeholder, labels_placeholder,is_training


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

    
def tower_loss( logit, labels):
    print(labels)
    print(logit)
    print(logit.shape)
    cross_entropy_mean = tf.reduce_mean(
                  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logit)
                  )

    # Calculate the total loss for the current tower.
    total_loss = cross_entropy_mean
    return total_loss


def tower_acc(logit, labels):
    correct_pred = tf.equal(tf.argmax(logit, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, wd):
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var)*wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var

def cov(feat):
    shape = feat.shape.as_list()
    # 将张量reshape为[batch_size,d(WHT),c(channel)]的形状
    feat = tf.reshape(feat,[shape[0],-1,shape[-1]])
    feat_var = tf.Variable(feat,name='feat_var')
    print('A')
    for i in range(shape[0]):
        for j in range(shape[-1]):
            # 计算当前batch和channel下feature map的均值和方差
            tensor_mean,tensor_variance = tf.nn.moments(feat[i,:,j], axes=[0])
            feat_var[i,:,j].assign(tf.nn.batch_normalization(feat[i,:,j],tensor_mean,tensor_variance,0,1,0))
#            with tf.Session() as sess:
#                #res=aa.eval(sess.run(tf.initialize_all_variables()))
#                resu=feat_var.eval(sess.run(tf.variables_initializer([feat_var])))
#            feat_ten = tf.constant(resu,dtype=tf.float32)
    feat_ten=tf.convert_to_tensor(feat_var)
    # 先转置再相乘 
    # [0, 1, 2]是正常显示，那么交换哪两个数字，就是把对应的输入张量的对应的维度对应交换即可
    print("A+")
    result = tf.matmul(feat_ten,tf.transpose(feat_ten,[0,2,1]))
    print(result)
    print("A++")
    return result

def normsq(feat1,feat2):
    feat1_cov = cov(feat1)
    feat2_cov = cov(feat2)   
    # 返回相关矩阵的形状
    print("B")
    shape_cov = feat1_cov.shape.as_list()
    print("B+")
    v_par=tf.zeros(shape=[shape_cov[0],])#初始化一个全零的tensor
    norm = tf.Variable(v_par,name='norm',dtype=tf.float32)
    print("B++")
    for i in range(shape_cov[0]):
        re_sq=tf.math.square(tf.norm(tf.subtract(feat1_cov[i],feat2_cov[i])))
        print(re_sq)
        norm[i].assign(re_sq)
    print("B+++")
    del feat1_cov
    del feat2_cov
    return norm

def frp(loss1,loss2):#Net1向Net2学习，判断Net1权重是否改变
    print("D")
    d_loss=tf.subtract(loss1,loss2)
    result= tf.cond(d_loss>0, 
                    lambda: tf.math.exp(tf.subtract(tf.multiply(beta,d_loss),1.)),
                    lambda: tf.constant(0.))
    print("D+")
    return result
    
# a=tf.constant([[1,1],[0,1],[1,0]], dtype=tf.float16)
# b=tf.constant([0,1,1])
# print('hahaha')
# result=tower_loss(a,b)
# print(result)
# print(tf.Session().run(result))
