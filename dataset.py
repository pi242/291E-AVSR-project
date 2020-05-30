import math
import torch
import glob
import os
import torch.nn as nn
import numpy as np
import torch.utils.data as td
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
import torchvision.models as models
from torch.autograd import Variable

class MyDataset(td.Dataset):
    def __init__(self, root_dir, mode='train', audio_mode='audio_clean', vid_out='resnet'):
        super(MyDataset, self).__init__()
        self.mode = mode
        self.audio_mode = audio_mode
        self.vid_out = vid_out
        if self.vid_out == 'resnet':
            self.resnet18 = models.resnet18(pretrained=True)
            modules=list(self.resnet18.children())[:-1]
            self.resnet18=nn.Sequential(*modules).double()
            for p in self.resnet18.parameters():
                p.requires_grad = False
                    
        if(mode=='train'):
            self.folder_dir = os.path.join(root_dir, 'train_npy')
            self.maxlen_label=129 
            self.maxlen_audio=204 
            self.maxlen_video=155 
        else:
            self.folder_dir = os.path.join(root_dir, 'test_npy')
            self.maxlen_label=129 
            self.maxlen_audio=204 
            self.maxlen_video=155
        
        self.files = glob.glob(self.folder_dir + "/*/*.npy", recursive=True)
        
    def __len__(self):
        return len(self.files)
    
    def __repr__(self):
        return "MyDataset(mode={})".format(self.mode)
    
    def __getitem__(self, idx):

        sample = np.load(self.files[idx],allow_pickle=True)

        audio_dim = sample.item().get("audio_dim")
        audio = sample.item().get(self.audio_mode)
        
        if(self.maxlen_audio-audio_dim[0]>0):
            audio_padding = np.zeros((self.maxlen_audio-audio_dim[0],audio_dim[1]))
            audio = np.concatenate((audio,audio_padding),axis=0)

        labels_length = sample.item().get("labels_length")
        labels = sample.item().get("labels")
        
        if(self.maxlen_label-labels_length):
            label_padding = -np.ones((self.maxlen_label-labels_length))
            labels = np.concatenate((labels,label_padding),axis=0)

        video_dim = sample.item().get("video_dim")
        video = sample.item().get("video")
        aus = sample.item().get("aus")
        
        mean = np.mean(video,axis=(0,1,2))
        std = np.std(video,axis=(0,1,2))

        new_mean=[0.485, 0.456, 0.406]
        new_std=[0.229, 0.224, 0.225]

        normalized_video = new_mean + (video-mean)*(new_std/std)

        rolled_video = np.rollaxis(normalized_video, 3, 1) 
        video_dim = (video_dim[0],video_dim[3],video_dim[1],video_dim[2])
        
        if (self.vid_out=='resnet'):
            tensor_video = torch.from_numpy(rolled_video).type(torch.DoubleTensor)
            renset_video = self.resnet18(tensor_video.double()).numpy()
            renset_video = renset_video.squeeze((2,3))
        
            resnet_video_dim = (video_dim[0],512)
            
            if(self.maxlen_video-video_dim[0]):
                video_padding = np.zeros((self.maxlen_video-resnet_video_dim[0],resnet_video_dim[1]))
                renset_video = np.concatenate((renset_video,video_padding),axis=0)
            return (audio, audio_dim), (renset_video, aus, resnet_video_dim), (labels, labels_length)
        else:
            if(self.maxlen_video-video_dim[0]):
                video_padding = np.zeros((self.maxlen_video-video_dim[0],video_dim[1],video_dim[2],video_dim[3]))
                final_video = np.concatenate((rolled_video,video_padding),axis=0)

            return (audio, audio_dim), (final_video, aus, video_dim), (labels, labels_length)