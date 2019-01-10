# -- coding: utf-8 --
# 01-09-2019

import numpy as np
import math
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image

from DeepMAR import DeepMAR_ResNet50
from utils import load_state_dict 
from utils import set_devices


class Agender(object):
    def __init__(self, gpuid=(0,)):

        # init the gpu ids      
        self.sys_device_ids = gpuid
        set_devices(self.sys_device_ids) 

        # read attributes from model
        self.att_list = ['Male', 'age<15', '15<age<30', '30<age<45', '45<age<60', 'age>60']

        # read model
        self.model_weight_file='./0109_epoch168.pth'         
        model_kwargs = dict()
        model_kwargs['num_att'] = len(self.att_list)
        model_kwargs['last_conv_stride'] = 2
        self.model_kwargs = model_kwargs

    def predict(self, imgname):
        # dataset transform
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
        test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,])

        # Attribute multitask model
        model = DeepMAR_ResNet50(**self.model_kwargs)
        map_location = (lambda storage, loc:storage)
        ckpt = torch.load(self.model_weight_file, map_location=map_location)
        model.load_state_dict(ckpt['state_dicts'][0])
        model.cuda()
        model.eval()

        # load image 
        img = Image.open(imgname)
        img_trans = test_transform(img) 
        img_trans = torch.unsqueeze(img_trans, dim=0)
        img_var = Variable(img_trans).cuda()
        score = model(img_var).data.cpu().numpy()    

        #output the label        
        ages = [score[0,i] for i in range(1, len(self.att_list))]
        age_idx = np.argmax(ages)
        age = self.att_list[age_idx+1]

        gender = 'Male' if score[0,0] >=0 else 'Female'

        return age, gender 
