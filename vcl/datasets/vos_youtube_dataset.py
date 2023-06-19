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
from .pipelines.my_aug import ClipRandomSizedCrop, ClipMotionStatistic, ClipRandomHorizontalFlip

from .pipelines import Compose
from vcl.utils import *


@DATASETS.register_module()
class VOS_youtube_dataset_rgb(Video_dataset_base):
    def __init__(self, data_prefix, 
                       rand_step=False,
                       year='2018',
                       **kwargs
                       ):
        super().__init__(**kwargs)

        self.data_prefix = data_prefix
        self.year = year
        self.rand_step = rand_step
        self.load_annotations()

    def __len__(self):
        return len(self.samples)

    def load_annotations(self):
        
        self.samples = []
        self.video_dir = osp.join(self.root, self.year, self.data_prefix['RGB'])
        list_path = osp.join(self.list_path, f'youtube{self.year}_{self.split}.json')
        data = mmcv.load(list_path)
        
        for vname, frames in data.items():
            sample = dict()
            sample['frames_path'] = []
            for frame in frames:
                sample['frames_path'].append(osp.join(self.video_dir, vname, frame))
                
            sample['num_frames'] = len(sample['frames_path'])
            if sample['num_frames'] < self.clip_length * self.step:
                continue
        
            self.samples.append(sample)
        
        logger = get_root_logger()
        logger.info(" Load dataset with {} videos ".format(len(self.samples)))

    
    
    def prepare_train_data(self, idx):
        
        if self.data_backend == 'lmdb' and self.env == None and self.txn == None:
            self._init_db(self.video_dir)

        sample = self.samples[idx]
        frames_path = sample['frames_path']
        num_frames = sample['num_frames']
        
        step = np.random.choice([2,5,8],p=[0.4,0.4,0.2]) if self.rand_step else self.step

        offsets = self.temporal_sampling(num_frames, self.num_clips, self.clip_length, step, mode=self.temporal_sampling_mode)
        
        # load frame
        if self.data_backend == 'raw':
            frames = self._parser_rgb_rawframe(offsets, frames_path, self.clip_length, step=self.step)
        else:
            frames = self._parser_rgb_lmdb(self.txn, offsets, frames_path, self.clip_length, step=self.step)

        data = {
            'imgs': frames,
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        } 

        return self.pipeline(data)
    
    
@DATASETS.register_module()
class VOS_youtube_dataset_rgb_V2(VOS_youtube_dataset_rgb):    
    
    def prepare_train_data(self, idx):

        sample = self.samples[idx]
        frames_path = sample['frames_path']
        num_frames = sample['num_frames']

        offsets = self.temporal_sampling(num_frames, self.num_clips, self.clip_length, self.step, mode=self.temporal_sampling_mode)
        offsets_ = self.temporal_sampling(num_frames, 1, self.num_clips, self.step, mode=self.temporal_sampling_mode)
        
        # load frame
        if self.data_backend == 'raw':
            frames = self._parser_rgb_rawframe(offsets, frames_path, self.clip_length, step=self.step)
            frames_ = self._parser_rgb_rawframe(offsets_, frames_path, self.num_clips, step=self.step)
        else:
            frames = self._parser_rgb_lmdb(offsets, frames_path, self.clip_length, step=self.step)
            frames_ = self._parser_rgb_lmdb(offsets_, frames_path, self.clip_length, step=self.step)
        

        data = {
            'imgs': frames_,
            'imgs_spa_aug': frames,
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        } 

        return self.pipeline(data)

@DATASETS.register_module()
class VOS_youtube_dataset_rgb_withbbox(VOS_youtube_dataset_rgb):

    def load_annotations(self):
        
        self.samples = []
        self.video_dir = osp.join(self.root, self.year, self.data_prefix['RGB'])
        self.mask_dir = osp.join(self.root, self.year, self.data_prefix['ANNO'])
        self.meta = mmcv.load(osp.join(self.list_path, 'ytvos_s256_flow_raft.json'))[self.split]

        for vid in self.meta:
            sample = dict()
            vname = vid["base_path"].split('/')[-1]
            sample['frames_path'] = []
            sample['frames_bbox'] = []
            sample['num_frames'] = len(vid['frame'])
 
            for frame in vid['frame']:
                sample['frames_path'].append(osp.join(self.video_dir, vname, frame['img_path']))
                sample['frames_bbox'].append(frame['objs'][0]['bbox'])
                
            if sample['num_frames'] < self.clip_length * self.step: continue

            self.samples.append(sample)
        
        logger = get_root_logger()
        logger.info(" Load dataset with {} videos ".format(len(self.samples)))
    
    def prepare_train_data(self, idx):

        sample = self.samples[idx]

        frames_path = sample['frames_path']
        frames_bbox = sample['frames_bbox']
        num_frames = sample['num_frames']

        offsets = self.temporal_sampling(num_frames, self.num_clips, self.clip_length, self.step, mode=self.temporal_sampling_mode)

        # load frame
        if self.data_backend == 'raw':
            frames = self._parser_rgb_rawframe(offsets, frames_path, self.clip_length, step=self.step)
        else:
            frames = self._parser_rgb_lmdb(offsets, frames_path, self.clip_length, step=self.step)
        
        bboxs = []
        for offset in offsets:
            for i in range(self.clip_length):
                bboxs.append(frames_bbox[offset+i])

        data = {
            'imgs': frames,
            'bboxs': bboxs,
            'mask_sample_size': (32, 32),
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        }

        return self.pipeline(data)


@DATASETS.register_module()
class VOS_youtube_dataset_rgb_withbbox_V2(VOS_youtube_dataset_rgb_withbbox):

    def prepare_train_data(self, idx):

        sample = self.samples[idx]

        frames_path = sample['frames_path']
        frames_bbox = sample['frames_bbox']
        num_frames = sample['num_frames']

        offsets = self.temporal_sampling(num_frames, self.num_clips, self.clip_length, self.step, mode=self.temporal_sampling_mode)
        offsets_ = self.temporal_sampling(num_frames, 1, self.num_clips, self.step, mode=self.temporal_sampling_mode)

        # load frame
        if self.data_backend == 'raw':
            frames = self._parser_rgb_rawframe(offsets, frames_path, self.clip_length, step=self.step)
            frames_ = self._parser_rgb_rawframe(offsets_, frames_path, self.num_clips, step=self.step)
        else:
            frames = self._parser_rgb_lmdb(offsets, frames_path, self.clip_length, step=self.step)
            frames_ = self._parser_rgb_lmdb(offsets_, frames_path, self.clip_length, step=self.step)
            
        bboxs = []
        for offset in offsets:
            for i in range(self.clip_length):
                bboxs.append(frames_bbox[offset+i])

        data = {
            'imgs': frames_,
            'imgs_spa_aug': frames,
            'bboxs': bboxs,
            'mask_sample_size': (32, 32),
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        }

        return self.pipeline(data)
    



@DATASETS.register_module()
class VOS_youtube_dataset_mlm(Video_dataset_base):
    def __init__(self, data_prefix, 
                       mask_ratio=0.15,
                       vq_size=32,
                       year='2018',
                       **kwargs
                       ):
        super().__init__(**kwargs)

        self.data_prefix = data_prefix
        self.year = year

        self.vq_res = vq_size
        self.mask_ratio = mask_ratio

        self.load_annotations()

    def load_annotations(self):
        
        self.samples = []
        self.video_dir = osp.join(self.root, self.year, self.data_prefix['RGB'])
        list_path = osp.join(self.list_path, f'youtube{self.year}_{self.split}.json')
        data = mmcv.load(list_path)
    
        
        for vname, frames in data.items():
            sample = dict()
            sample['frames_path'] = []
            for frame in frames:
                sample['frames_path'].append(osp.join(self.video_dir, vname, frame))
                
            sample['num_frames'] = len(sample['frames_path'])
            if sample['num_frames'] < self.clip_length * self.step:
                continue
        
            self.samples.append(sample)
        
        logger = get_root_logger()
        logger.info(" Load dataset with {} videos ".format(len(self.samples)))
    
    

    def prepare_test_data(self, idx):
        
        sample = self.samples[idx]
        frames_path = sample['frames_path']
        num_frames = sample['num_frames']

        offsets = self.temporal_sampling(num_frames, self.num_clips, self.clip_length, self.step, mode=self.temporal_sampling_mode)
        
        if self.data_backend == 'raw':
            frames = self._parser_rgb_rawframe(offsets, frames_path, self.clip_length, step=self.step)
        else:
            frames = self._parser_rgb_lmdb(offsets, frames_path, self.clip_length, step=self.step)


        mask = cv2.resize(mask, (self.vq_res, self.vq_res), cv2.INTER_NEAREST).reshape(-1)
        obj_idxs = np.nonzero(mask)[0]

        if mask.max() > 0:
            sample_idx = np.array(random.sample(obj_idxs.tolist(), 1))
        else:
            sample_idx = np.array(random.sample(range(self.vq_res * self.vq_res), 1))

        assert sample_idx.shape[0] == 1

        data = {
            'imgs': frames,
            'mask_query_idx': sample_idx,
            'modality': 'RGB',
            'num_clips': 1,
            'num_proposals':1,
            'clip_len': self.clip_length
        }

        return self.pipeline(data)

    def prepare_train_data(self, idx):
        
        sample = self.samples[idx]
        frames_path = sample['frames_path']
        num_frames = sample['num_frames']

        offsets = self.temporal_sampling(num_frames, self.num_clips, self.clip_length, self.step, mode=self.temporal_sampling_mode)       
        if self.data_backend == 'raw':
            frames = self._parser_rgb_rawframe(offsets, frames_path, self.clip_length, step=self.step)
        else:
            frames = self._parser_rgb_lmdb(offsets, frames_path, self.clip_length, step=self.step)

        mask_num = int(self.vq_res * self.vq_res * self.mask_ratio)
        mask_query_idx = np.zeros(self.vq_res * self.vq_res)
        sample_idx = np.array(random.sample(range(self.vq_res * self.vq_res), mask_num))
        mask_query_idx[sample_idx] = 1

        assert mask_query_idx.sum() == mask_num

        data = {
            'imgs': frames,
            'mask_query_idx': mask_query_idx,
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        }


        return self.pipeline(data)


@DATASETS.register_module()
class VOS_youtube_dataset_mlm_withbbox_random(VOS_youtube_dataset_mlm):
    def __init__(self, size=None, p=1.0, crop_ratio=0.6, return_first_query=False, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.return_first_query = return_first_query

    def load_annotations(self):
        
        self.samples = []
        self.video_dir = osp.join(self.root, self.year, self.data_prefix['RGB'])
        self.mask_dir = osp.join(self.root, self.year, self.data_prefix['ANNO'])
        self.meta = mmcv.load(osp.join(self.list_path, 'ytvos_s256_flow_raft.json'))[self.split]
        video_idx = 0
        
        for vid in self.meta:
            sample = dict()
            vname = vid["base_path"].split('/')[-1]
            sample['video_name'] = vname
            sample['frames_path'] = []
            sample['frames_bbox'] = []
            sample['num_frames'] = len(vid['frame'])
 
            for frame in vid['frame']:
                sample['frames_path'].append(osp.join(self.video_dir, vname, frame['img_path']))
                sample['frames_bbox'].append(frame['objs'][0]['bbox'])
            
            if sample['num_frames'] < self.clip_length * self.step:
                continue
            
            sample['video_idx'] = video_idx
            video_idx += 1
            self.samples.append(sample)
        
        logger = get_root_logger()
        logger.info(" Load dataset with {} videos ".format(len(self.samples)))

    
    def prepare_train_data(self, idx):
        
        sample = self.samples[idx]
        frames_path = sample['frames_path']
        frames_bbox = sample['frames_bbox']
        num_frames = sample['num_frames']
        video_idx = sample['video_idx']
        
        offsets = self.temporal_sampling(num_frames, self.num_clips, self.clip_length, self.step, mode=self.temporal_sampling_mode)


        if self.data_backend == 'raw':
            frames = self._parser_rgb_rawframe(offsets, frames_path, self.clip_length, step=self.step)
        else:
            frames = self._parser_rgb_lmdb(offsets, frames_path, self.clip_length, step=self.step)
        
        bboxs = []
        for off in offsets:
            for l in range(self.clip_length):
                bboxs.append(frames_bbox[off+l])
        
        if random.random() <= self.p:
            data = {
                'imgs': frames,
                'bboxs': bboxs,
                'video_name':sample['video_name'],
                'mask_ratio': self.mask_ratio,
                'video_idx': video_idx,
                'mask_sample_size': (self.vq_res, self.vq_res),
                'modality': 'RGB',
                'num_clips': self.num_clips,
                'num_proposals':1,
                'clip_len': self.clip_length,
                'return_first_query': self.return_first_query
            }
            return self.pipeline(data)

        else:
            mask_num = int(self.vq_res * self.vq_res * self.mask_ratio)
            mask_query_idx = np.zeros(self.vq_res * self.vq_res).astype(np.uint8)
            sample_idx = np.array(random.sample(range(self.vq_res * self.vq_res), mask_num))
            mask_query_idx[sample_idx] = 1

            data = {
                'imgs': frames,
                'video_name':sample['video_name'],
                'mask_query_idx': mask_query_idx,
                'video_idx': video_idx,
                'modality': 'RGB',
                'num_clips': self.num_clips,
                'num_proposals':1,
                'clip_len': self.clip_length,
                'return_first_query': self.return_first_query
            }

            return self.pipeline(data)



@DATASETS.register_module()
class VOS_youtube_dataset_rgb_flow(VOS_youtube_dataset_rgb):

    def load_annotations(self):
        
        self.samples = []
        self.video_dir = osp.join(self.root, self.year, self.data_prefix['RGB'])
        self.flow_dir = osp.join(self.list_path, self.data_prefix['FLOW'])
        
        list_path = osp.join(self.list_path, f'youtube{self.year}_{self.split}.json')
        data = mmcv.load(list_path)
        
        for vname, frames in data.items():
            sample = dict()
            sample['frames_path'] = []
            sample['flows_path'] = []
            
            for idx, frame in enumerate(frames):
                if idx < len(frames) -1:
                    sample['frames_path'].append(osp.join(self.video_dir, vname, frame))
                    sample['flows_path'].append(osp.join(self.flow_dir, vname, frame))
                
            sample['num_frames'] = len(sample['frames_path'])
            if sample['num_frames'] < self.clip_length * self.step:
                continue
        
            self.samples.append(sample)
        
        logger = get_root_logger()
        logger.info(" Load dataset with {} videos ".format(len(self.samples)))
        
    def prepare_train_data(self, idx):
        
        if self.data_backend == 'lmdb' and self.env == None and self.txn == None:
            self._init_db(self.video_dir)

        sample = self.samples[idx]
        frames_path = sample['frames_path']
        flows_path = sample['flows_path']
        num_frames = sample['num_frames']
        
        offsets = self.temporal_sampling(num_frames, self.num_clips, self.clip_length, self.step, mode=self.temporal_sampling_mode)
        
        # load frame
        if self.data_backend == 'raw':
            frames = self._parser_rgb_rawframe(offsets, frames_path, self.clip_length, step=self.step)
        else:
            frames = self._parser_rgb_lmdb(self.txn, offsets, frames_path, self.clip_length, step=self.step)

        flows = self._parser_rgb_rawframe(offsets, flows_path, self.clip_length, step=self.step)
        flows = [ cv2.resize(flow, frames[0].shape[:2][::-1]) / 255 * 100 - 50 for flow in flows ]
        
        
        data = {
            'imgs': frames,
            'flows': flows,
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        } 

        return self.pipeline(data)