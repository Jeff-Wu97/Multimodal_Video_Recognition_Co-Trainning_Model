# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 02:04:40 2020

@author: wuzhe
"""

from __future__ import print_function
import numpy as np
import os
import imageio
import cv2
import shutil
import re
from glob import glob
from moviepy.editor import VideoFileClip

#输入参数
#视频的尺寸320x240
#短边缩放后尺寸
scalling=[341,256]#W,H
#裁剪尺寸（输出）
out = [224,224]#W,H
#视频文件路径
#file_dir = r"C:\Users\wuzhe\Desktop\UCF101\UCF-101\ApplyLipstick\v_ApplyLipstick_g24_c04.avi"
##RGB输出路径(文件夹)存储结构：UCF-101->类名->视频名->模态（i,x,y）->帧文件,帧文件为图片

####################################################################################
#尺度缩放：短边缩放至256，再采中央224x224区域,生成RGB
def crop_center(img,cropx,cropy):
    [y,x] = [img.shape[0],img.shape[1]]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]

def produceRGB(videodir,RGBdir):
    [outW,outH] = out
    with imageio.get_reader(videodir,  'ffmpeg') as vid:
        nframes = vid.get_meta_data()['nframes']
        for i, frame in enumerate(vid):
            n_frames = i
            frame = cv2.resize(frame, (scalling[0], scalling[1]), interpolation = cv2.INTER_CUBIC)
            frame = crop_center(frame,out[0],out[1])#将256*256的图片裁剪成224*224
            imageio.imwrite(RGBdir + r"\frame_%d.jpg" %i, frame)
        
######################################################################################    
#这是生成flowx和flow y的代码
def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.DualTVL1OpticalFlow_create()
    flow0=TVL1.calc(prev, curr, None)
    assert flow0.dtype == np.float32
    
    flow1 = (flow0 + bound) * (255.0 / (2 * bound))
    flow2 = np.round(flow1).astype(int)
    flow2[flow2 >= 255] = 255
    flow2[flow2 <= 0] = 0

    return flow2

def cal_for_frames(video_path):
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()
    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    for i, frame_curr in enumerate(frames):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow

def save_flow(video_flows, flowx_path,flowy_path):
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flowx_path.format('u'), "{:06d}.jpg".format(i)),
                    flow[:, :, 0])
        cv2.imwrite(os.path.join(flowy_path.format('v'), "{:06d}.jpg".format(i)),
                    flow[:, :, 1])

def extract_flow(video_path, flowx_path,flowy_path):
    flow = cal_for_frames(video_path)
    save_flow(flow, flowx_path,flowy_path)
    print('complete:' + flowx_path+" & " +flowy_path)
    return

def frames(video_path):
    clip = VideoFileClip(video_path)
    return clip.duration
##################################################################################
##File Operation
    
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
        print("-- new folder --",path," OK!")
        infl=1
    else:
        print('Got the target folder')
        infl= 0
    return infl

####################################################################################
#创建所有类的文件夹
#找到一个类下的所有视频
#创建该视频下的 RGB文件夹 FlowX文件夹 FlowY文件夹
#产生RGB
#产生FlowXY

rootpath=r"C:\Users\wuzhe\Desktop\UCF101\UCF-101" #DATASET文件夹路径
newrootpath=r"C:\Users\wuzhe\Desktop\UCF101"

#获取所有类的名字
dirs = os.listdir(rootpath)
#在新目录下创建新的类文件夹
for label in dirs:
    new_class_path=os.path.join(newrootpath,label)#一个类别的帧的文件夹路径
    flag_c=mkdir(new_class_path)#创建文件夹存储一类的帧

#对一个类中的视频
    v_classpath=os.path.join(rootpath,label)#视频中一个类文件夹的路径
    v_dirs=os.listdir(v_classpath)#视频中一个类文件夹里的所有视频列表
    
    for video in v_dirs:
        video_path=os.path.join(v_classpath,video)#视频中一个类文件夹中一个视频的路径
        new_v_path=os.path.join(new_class_path,video)#生成一个视频的帧文件夹的路径
        flag_v=mkdir(new_v_path)##创建文件夹存储一个视频的帧
        if flag_v == 0:
            continue
        new_RGBfolder=os.path.join(new_v_path,*"i")
        fi=mkdir(new_RGBfolder)#创建储存RGB帧的文件夹
        new_fxfolder=os.path.join(new_v_path,*"x")
        fx=mkdir(new_fxfolder)#创建储存flowx帧的文件夹
        new_fyfolder=os.path.join(new_v_path,*"y")
        fy=mkdir(new_fyfolder)#创建储存flowy帧的文件夹
        
        produceRGB(video_path,new_RGBfolder)
        extract_flow(new_RGBfolder,new_fxfolder,new_fyfolder)
        print("Complete for video",video)
    print("Complete for ",label,"!")
    
