import copy
from operator import length_hint
import os.path as osp
from collections import defaultdict
from pathlib import Path
import glob
import os
import random
import pickle
from mmcv.fileio.io import load
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

from mmcv import scandir
import mmcv

from .base_dataset import BaseDataset
from .video_dataset import *
from .registry import DATASETS

from .pipelines import Compose


@DATASETS.register_module()
class Kinetics_dataset_rgb(Video_dataset_base):
    
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.load_annotations()

    def load_annotations(self):
        
        self.samples = []
        list_path = osp.join(self.list_path, f'{self.split}_list.txt')


        with open(list_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                sample = dict()
                vname, num_frames, _ = line.strip('\n').split()
        
                sample['frames_path'] = sorted(glob.glob(osp.join(self.root, vname, '*.jpg')))
                sample['num_frames'] = len(sample['frames_path'])
                
                if sample['num_frames'] < self.clip_length * self.step: continue
                
                self.samples.append(sample)
    
    
    def prepare_train_data(self, idx):

        sample = self.samples[idx]
        frames_path = sample['frames_path']
        num_frames = sample['num_frames']

        offsets = self.temporal_sampling(num_frames, self.num_clips, self.clip_length, self.step, mode=self.temporal_sampling_mode)
        
        # load frame
        frames = self._parser_rgb_rawframe(offsets, frames_path, self.clip_length, step=self.step)

        data = {
            'imgs': frames,
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        } 

        return self.pipeline(data)