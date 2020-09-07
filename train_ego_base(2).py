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
import sys
sys.path.append('../../')
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import math
import numpy as np
from i3d import InceptionI3d
from utils import *
from tensorflow.python import pywrap_tensorflow
import seaborn as sns
import matplotlib.pyplot as plt

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
step_count=0
flag_ssa=0

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2817, 'Number of steps to run trainer.')#20 epoch
# flags.DEFINE_integer('max_steps', 2, 'Number of steps to run trainer.')#20 epoch
flags.DEFINE_integer('batch_size',6, 'Batch size.')
flags.DEFINE_integer('val_batch_size',156, 'validation Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 64, 'Nummber of frames per clib')
# flags.DEFINE_integer('test_size',15,'Nuber of videos foe each test')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'RGB_channels for input')
flags.DEFINE_integer('flow_channels', 2, 'FLOW_channels for input')
flags.DEFINE_integer('classics', 15, 'The num of class')
flags.DEFINE_integer('sample_rate', 1, 'in the input_data.py file, num_frames_per_clip/sample_rate')
FLAGS = flags.FLAGS
model_save_dir = './models/ego_1400_6_300_0.0001_decay_ssa_base_lr001'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def run_training():
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.
    global step_count
    global flag_ssa
    
    # Create model directory
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print('Hello')
    rgb_pre_model_save_dir = "/home/s1924153/checkpoints/rgb_imagenet"
    flow_pre_model_save_dir = "/home/s1924153/checkpoints/flow_imagenet"
    print(rgb_pre_model_save_dir)
    print('Seeyou')
    with tf.Graph().as_default():
        global_step = tf.get_variable(
                        'global_step',
                        [],
                        initializer=tf.constant_initializer(0),
                        trainable=False
                        )
        rgb_images_placeholder, flow_images_placeholder, labels_placeholder,is_training = placeholder_inputs(
                        FLAGS.batch_size * gpu_num,
                        FLAGS.num_frame_per_clib,
                        FLAGS.crop_size,
                        FLAGS.rgb_channels,
                        FLAGS.flow_channels
                        )
        
        val_rbg_logit_placeholder = tf.placeholder(tf.float32, shape=((FLAGS.val_batch_size//FLAGS.batch_size)*FLAGS.batch_size,FLAGS.classics))
        val_flow_logit_placeholder = tf.placeholder(tf.float32, shape=((FLAGS.val_batch_size//FLAGS.batch_size)*FLAGS.batch_size,FLAGS.classics))
        val_pred_placeholder = tf.placeholder(tf.float32, shape=((FLAGS.val_batch_size//FLAGS.batch_size)*FLAGS.batch_size,FLAGS.classics))
        val_lab_placeholder = tf.placeholder(tf.int64, shape=((FLAGS.val_batch_size//FLAGS.batch_size)*FLAGS.batch_size))
        
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=765, decay_rate=0.1, staircase=True)
        opt_rgb = tf.train.AdamOptimizer(learning_rate)
        opt_flow = tf.train.AdamOptimizer(learning_rate)
        #opt_stable = tf.train.MomentumOptimizer(learning_rate, 0.9)
        with tf.variable_scope('RGB'):
            rgb_logit, rgbends = InceptionI3d(
                                    num_classes=FLAGS.classics,
                                    spatial_squeeze=True,
                                    final_endpoint='Logits'
                                    )(rgb_images_placeholder, is_training)
            
        with tf.variable_scope('Flow'):
            flow_logit, flowends = InceptionI3d(
                                    num_classes=FLAGS.classics,
                                    spatial_squeeze=True,
                                    final_endpoint='Logits'
                                    )(flow_images_placeholder, is_training)

        #SSALoss=ssa(rgbends['Mixed_5c'],flowends['Mixed_5c'])
        sess=tf.Session()
        rgb_loss = tower_loss(
                                rgb_logit,
                                labels_placeholder
                                )
        flow_loss = tower_loss(
                                flow_logit,
                                labels_placeholder
                                )
        val_rgb_loss = tower_loss(
                                val_rbg_logit_placeholder,
                                val_lab_placeholder
                                )
        val_flow_loss = tower_loss(
                                val_flow_logit_placeholder,
                                val_lab_placeholder
                                )
        
        predict = tf.add(tf.nn.softmax(rgb_logit), tf.nn.softmax(flow_logit))
        # val_predict = tf.add(tf.nn.softmax(val_rgb_logit), tf.nn.softmax(val_flow_logit))
        accuracy = tower_acc(predict, labels_placeholder)
        val_accuracy = tower_acc(val_pred_placeholder, val_lab_placeholder)
        
        rgb_variable_list = {}
        flow_variable_list = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_list[variable.name] = variable

        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow':
                flow_variable_list[variable.name] = variable
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            #每75个step的后15个step进行ssa操作
            
            rgb_grads = opt_rgb.compute_gradients(rgb_loss, var_list=rgb_variable_list)
            flow_grads = opt_flow.compute_gradients(flow_loss, var_list=flow_variable_list)

            apply_gradient_rgb = opt_rgb.apply_gradients(rgb_grads, global_step=global_step)
            apply_gradient_flow = opt_flow.apply_gradients(flow_grads, global_step=global_step)
            print(">>>>>><<<<<<<")
            train_op = tf.group(apply_gradient_rgb, apply_gradient_flow)
            print("<<<<<<>>>>>>>")
            null_op = tf.no_op()

        # Create a saver for loading trained checkpoints.
        rgb_variable_map = {}
        flow_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB' and 'Adam' not in variable.name.split('/')[-1] and variable.name.split('/')[2] != 'Logits':
                #rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow'and 'Adam' not in variable.name.split('/')[-1] and variable.name.split('/')[2] != 'Logits':
                flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
        
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        print("M1")
        # Create a session for running Ops on the Graph.
        sess = tf.Session(
                        config=tf.ConfigProto(allow_soft_placement=True)
                        )
        sess.run(init)
        print("M2")
        # Create summary writter
        accuracy_scalar=tf.summary.scalar('accuracy', accuracy)
        rgb_loss_scalar=tf.summary.scalar('rgb_loss', rgb_loss)
        flow_loss_scalar=tf.summary.scalar('flow_loss', flow_loss)
        learning_rate_scalar=tf.summary.scalar('learning_rate', learning_rate)
        
        
        val_acc=tf.summary.scalar('val_accuracy', val_accuracy)
        val_r_loss=tf.summary.scalar('val_rgb_loss', val_rgb_loss)
        val_f_loss=tf.summary.scalar('val_flow_loss', val_flow_loss)
        
    merged = tf.summary.merge([accuracy_scalar,rgb_loss_scalar,flow_loss_scalar,learning_rate_scalar])
    val_merged = tf.summary.merge([val_acc,val_r_loss,val_f_loss])
    # load pre_train models
    print(rgb_pre_model_save_dir)
    ckpt = tf.train.get_checkpoint_state(rgb_pre_model_save_dir)
    print(ckpt)
    print(ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        rgb_saver.restore(sess, ckpt.model_checkpoint_path)
        print("load complete!")
    ckpt = tf.train.get_checkpoint_state(flow_pre_model_save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        flow_saver.restore(sess, ckpt.model_checkpoint_path)
        print("load complete!")
    print("M2+")
    train_writer = tf.summary.FileWriter('./visual_logs/train_ego_1400_6_300_0.0001_decay_4_base_lr001', sess.graph)
    test_writer = tf.summary.FileWriter('./visual_logs/test_ego_1400_6_300_0.0001_decay_4_base_lr001', sess.graph)
    for step in xrange(FLAGS.max_steps):

        start_time = time.time()
        rgb_train_images, flow_train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
                      filename=r'/home/s1924153/train.list',
                      batch_size=FLAGS.batch_size * gpu_num,
                      num_frames_per_clip=FLAGS.num_frame_per_clib,
                      sample_rate=FLAGS.sample_rate,
                      crop_size=FLAGS.crop_size,
                      shuffle=True,
                      add_flow = True
                      )
        

        sess.run(train_op, feed_dict={
                      rgb_images_placeholder: rgb_train_images,
                      flow_images_placeholder: flow_train_images,
                      labels_placeholder: train_labels,
                      is_training: True
                      })

        duration = time.time() - start_time
        print('Step %d: %.3f sec' % (step, duration))

        
        #Save a checkpoint and evaluate the model periodically.
        if step % 94 == 0 or (step + 1) == FLAGS.max_steps:
            print('Training Data Eval: Epoch '+str(step//94))
            summary, acc, loss_rgb, loss_flow = sess.run(
                            [merged, accuracy, rgb_loss, flow_loss],
                            feed_dict={rgb_images_placeholder: rgb_train_images,
                                        flow_images_placeholder: flow_train_images,
                                        labels_placeholder: train_labels,
                                        is_training: False
                                      })
            print("accuracy: " + "{:.5f}".format(acc))
            print("rgb_loss: " + "{:.5f}".format(loss_rgb))
            print("flow_loss: " + "{:.5f}".format(loss_flow))
            train_writer.add_summary(summary, step)
            print('Validation Data Eval: Epoch '+str(step//94))
            
            total_pred = []
            total_label = []
            total_rgb_logit = []
            total_flow_logit = []
            pos = 0
            for i in range(FLAGS.val_batch_size//FLAGS.batch_size):
                rgb_val_images, flow_val_images, val_labels, pos, _, _ = input_data.read_clip_and_label(
                            filename=r'//home/s1924153/test.list',
                            batch_size=FLAGS.batch_size * gpu_num, #test_size,#FLAGS.batch_size * gpu_num,#每次validation的个数仅仅是batchsize个
                            start_pos=pos,
                            num_frames_per_clip=FLAGS.num_frame_per_clib,
                            sample_rate=FLAGS.sample_rate,
                            crop_size=FLAGS.crop_size,
                            shuffle=False,
                            add_flow = True
                            )
                pred,r_logit,f_logit = sess.run([predict,rgb_logit,flow_logit],feed_dict={rgb_images_placeholder: rgb_val_images,
                                                    flow_images_placeholder: flow_val_images,
                                                    labels_placeholder: val_labels,
                                                    is_training: False
                                                    })
             
                for ii in range(FLAGS.batch_size):
                    total_rgb_logit.append(r_logit[ii])
                    total_flow_logit.append(f_logit[ii])
                    total_pred.append(pred[ii])
                    total_label.append(val_labels[ii])
                
            val_summary, acc,r_loss,f_loss = sess.run([val_merged,val_accuracy,val_rgb_loss,val_flow_loss], feed_dict={val_pred_placeholder: np.array(total_pred),
                                                                                                               val_lab_placeholder: np.array(total_label),
                                                                                                               val_rbg_logit_placeholder:np.array(total_rgb_logit),
                                                                                                               val_flow_logit_placeholder:np.array(total_flow_logit)
                                                                                                               }) 
            
            fileObject = open('ego_1400_6_300_0.0001_decay_4_base_lr001.txt', 'a')
            fileObject.write('Validation Data Eval: Epoch'+str(step//94))
            fileObject.write('\n')
            fileObject.write('rbg_loss is '+str(r_loss))
            fileObject.write('\n')
            fileObject.write('flow_loss '+str(f_loss))
            fileObject.write('\n')
            fileObject.write('accuracy '+ str(acc))
            fileObject.write('\n')
            fileObject.write('###############################################################################')
            fileObject.write('\n')
            fileObject.close()
                
            # summary = sess.run(val_merged)    
            print("rbg_loss: " + "{:.5f}".format(r_loss))
            print("flow_loss: " + "{:.5f}".format(f_loss))
            print("accuracy: " + "{:.5f}".format(acc))
          
            test_writer.add_summary(val_summary, step)
    
        if (step + 1) == FLAGS.max_steps:
            saver.save(sess, os.path.join(model_save_dir, 'i3d_ego_1400_6_300_0.0001_decay_4_base_lr001'), global_step=step)
    
            pred_result=[]
            for sample in total_pred:
                pred_result.append(np.argmax(sample))
            fpred = open('pred_result_ego_1400_6_300_0.0001_decay_4_base_lr001.txt','a')
            flabel = open('label_result_ego_1400_6_300_0.0001_decay_4_base_lr001.txt','a')
            for preds in pred_result:
                fpred.write(str(preds))
            for true_label in total_label:
                flabel.write(str(total_label))
            fpred.close()
            flabel.close()
            
            confusionMat=tf.math.confusion_matrix(total_label, pred_result)#返回tensor，行是正确label，列代表预测结果
            arr_confusionMat = tf.Session().run(confusionMat)
            plt.subplots(figsize=(9, 9))
            sns.heatmap(arr_confusionMat, annot=True, vmax=1, square=True, cmap="RdBu_r")
            plt.savefig('heatmap_ego_1400_6_300_0.0001_decay_4_base_lr001.jpg')
                
            
        step_count=step_count+1
        
    
    print("done")


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
