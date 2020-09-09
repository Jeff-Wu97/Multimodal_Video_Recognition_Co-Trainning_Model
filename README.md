# Multimodal_Video_Recognition_Co-Trainning_Model

This is a model for MLP course project. The project focuses on the recognition of hand gesture in the video, which is achieved by two co-trained I3D networks. Our model takes two diﬀerent modalities of frames from the video, RGB and optical, as input. The co-training network is supposed to have the better performance on hand gesture recognition than a single-branch network. In this project, the spatiotemporal semantic alignment optimization method is applied to optimized the video recognition system. The designed model can achieve accuracy more than 99.3% on the EgoGesture dataset,which contains diﬀerent subjects of hand gesture on various scenes.


## Environment

Tensorflow-gpu-1.5

Tensorflow_probability-0.7

Sonnet-1.25

Opencv-3.4.2

Imageio
