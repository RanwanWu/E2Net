import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.E2Net import E2Net
from data import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=224, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='./Dataset/test/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#load the model
model = E2Net()
#Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./pre/') when training.
model.load_state_dict(torch.load('./pre/E2Net/E2Net.pth'), False)
model.cuda()
model.eval()

#test
test_datasets = ['VT821','VT1000','VT5000_test']
for dataset in test_datasets:
    save_path = './Salmaps/E2Net/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    T_root=dataset_path +dataset +'/T/'
    test_loader = test_dataset(image_root, gt_root,T_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt,T, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        T = T.cuda()
        res = model(image,T)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path+name,res*255)
    print('Test Done!')
