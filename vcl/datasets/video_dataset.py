# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset
from .base_dataset import BaseDataset

import random
import mmcv
import numpy as np
import lmdb
import os
import io

class Video_dataset_base(BaseDataset):
    def __init__(self, root,  
                       list_path, 
                       num_clips=1,
                       clip_length=1,
                       step=1,
                       pipeline=None, 
                       test_mode=False,
                       filename_tmpl='{:05d}.jpg',
                       temporal_sampling_mode='random',
                       data_backend='raw',
                       split='train'
                       ):
        super().__init__(pipeline, test_mode)

        self.clip_length = clip_length
        self.num_clips = num_clips
        self.step = step
        self.list_path = list_path
        self.root = root
        self.filename_tmpl = filename_tmpl
        self.temporal_sampling_mode = temporal_sampling_mode
        self.split = split
        self.data_backend = data_backend
        self.env = None
        self.txn = None
        self.env_anno = None
        self.txn_anno = None

    def temporal_sampling(self, num_frames, num_clips, clip_length, step, mode='random'):
            
        if mode == 'random':
            offsets = [ random.randint(0, num_frames-clip_length * step) for i in range(num_clips) ]
            offsets = sorted(offsets)
        elif mode == 'distant':
            length_ext = num_frames / num_clips
            offsets = np.floor(np.arange(num_clips) * length_ext + np.random.uniform(low=0.0, high=length_ext, size=(num_clips))).astype(np.uint8)
        elif mode =='mast':
            short_term_interval = 2
            offsets_long_term = [0,1]
            short_term_start = random.randint(2, num_frames-clip_length * step - (num_clips-2) * short_term_interval )
            offsets_short_term = list([ short_term_start+i*short_term_interval for i in range(num_clips-2)])
            offsets = offsets_long_term + offsets_short_term
        elif mode == 'mast_v2':
            length_ext = (num_frames - 1) / (num_clips - 1)
            offsets = np.floor(np.arange(num_clips-1) * length_ext + np.random.uniform(low=0.0, high=length_ext, size=(num_clips-1))).astype(np.uint8).tolist()
            offsets.append(offsets[-1]+1)
        return offsets

    def _parser_rgb_rawframe(self, offsets, frames_path, clip_length, step=1, flag='color', backend='cv2'):
        """read frame"""
        frame_list_all = []
        for offset in offsets:
            for idx in range(clip_length):
                frame_path = frames_path[offset + idx * step]
                frame = mmcv.imread(frame_path, backend=backend, flag=flag, channel_order='rgb')
                frame_list_all.append(frame)
        return frame_list_all

    def _parser_rgb_lmdb_deprected(self, offsets, frames_path, clip_length, step=1, flag='color', backend='cv2'):
        """read frame"""
        lmdb_env = lmdb.open(os.path.dirname(frames_path[0]), readonly=True, lock=False)
        frame_list_all = []
        with lmdb_env.begin() as lmdb_txn:
            for offset in offsets:
                for idx in range(clip_length):
                    frame_path = os.path.basename(frames_path[ offset + idx * step])
                    bio = lmdb_txn.get(frame_path.encode())
                    frame = mmcv.imfrombytes(bio, backend=backend, flag=flag, channel_order='rgb')
                    frame_list_all.append(frame)
        return frame_list_all 
    
    def _parser_rgb_lmdb(self, txn, offsets, frames_path, clip_length, step=1, flag='color', backend='cv2'):
        """read frame"""
        frame_list_all = []
        for offset in offsets:
            for idx in range(clip_length):
                frame_path = '/'.join(frames_path[ offset + idx * step].split('/')[-2:])
                bio = txn.get(frame_path.encode())
                frame = mmcv.imfrombytes(bio, backend=backend, flag=flag, channel_order='rgb')
                frame_list_all.append(frame)
        return frame_list_all 
    
    def _init_db(self, db_path, anno=False):
        
        if not anno:
            self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                readonly=True, lock=False,
                readahead=False, meminit=False)
            
            self.txn = self.env.begin(write=False)
        else:
            self.env_anno = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                readonly=True, lock=False,
                readahead=False, meminit=False)
            
            self.txn_anno = self.env_anno.begin(write=False)