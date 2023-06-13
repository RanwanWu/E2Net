# RGB-T Salient Object Detection via Excavating and Enhancing CNN Features
![image](figs/overview.png)  
   Figure.1 The overall architecture of the proposed E2Net model, which is published in Applied Intlligence 🎆.  
   
# 1.Requirements
Python v3.6, Pytorch 0.4.0+, Cuda 10.0, TensorboardX 2.0, opencv-python

# 2.Data Preparation
Download the train data from [here](https://pan.baidu.com/s/1HQMjdqY1C6m9_joybUv2Dw)[code:NEPU], test data from [here](https://pan.baidu.com/s/1xIvwBd8LjmJRwkIMQWqaVQ)[code:NEPU], test_in_train data from [here](https://pan.baidu.com/s/1HChMhmnZh3YCLQpxutLvLg)[code:NEPU]. Then put them under the following directory:  

    -Dataset\   
       -train\  
       -test\ 
           -VT821\
           -VT1000\
           -VT5000_test\
       -test_in_train\
