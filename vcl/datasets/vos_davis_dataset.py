import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import copy

import torch
import torchvision
from torch.utils import data

import random
import glob
from collections import *
import pdb
from mmcv.utils import print_log
import mmcv
from vcl.utils import *

from davis2017.evaluation import DAVISEvaluation


from .base_dataset import BaseDataset
from .video_dataset import *
from .registry import DATASETS


@DATASETS.register_module()
class VOS_davis_dataset_test(Video_dataset_base):
    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [191, 0, 0], [64, 128, 0], [191, 128, 0], [64, 0, 128],
               [191, 0, 128], [64, 128, 128], [191, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 191, 0], [128, 191, 0], [0, 64, 128],
               [128, 64, 128]]
    
    def __init__(self, 
                       data_prefix, 
                       task='semi-supervised',
                       year='2017',
                       split='val',
                       **kwargs
                       ):
        super().__init__(split=split, **kwargs)

        self.task = task
        self.data_prefix = data_prefix
        self.year = year

        self.load_annotations()

    def load_annotations(self):
        
        self.samples = []
        self.mask_dir = osp.join(self.root, 'Annotations', '480p')
        self.video_dir = osp.join(self.root, 'JPEGImages', '480p')
        list_path = osp.join(self.list_path, f'davis{self.year}_{self.split}_list.json')                
        data = mmcv.load(list_path)
        
        for vname, frames in data.items():
            sample = dict()
            sample['frames_path'] = []
            sample['masks_path'] = []
            for frame in frames:
                sample['frames_path'].append(osp.join(self.video_dir, vname, frame))
                sample['masks_path'].append(osp.join(self.mask_dir, vname, frame.replace('jpg','png')))
            sample['video_path'] = osp.join(self.video_dir, vname)
            sample['num_frames'] = len(sample['frames_path'])
            if sample['num_frames'] < self.clip_length * self.step:
                continue
        
            self.samples.append(sample)
        
        logger = get_root_logger()
        logger.info(" Load dataset with {} videos ".format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def prepare_test_data(self, index):
        if self.data_backend == 'lmdb' and self.env == None and self.txn == None:
            self._init_db(self.video_dir)
            self._init_db(self.mask_dir, anno=True)
        
        sample = self.samples[index]
        num_frames = sample['num_frames']
        masks_path = sample['masks_path']
        frames_path = sample['frames_path']
        
        # load frames and masks
        if self.data_backend != 'lmdb':
            frames = self._parser_rgb_rawframe([0], frames_path, num_frames)
            ref = np.array(self._parser_rgb_rawframe([0], masks_path, 1, flag='unchanged', backend='pillow')[0])
        else:
            frames = self._parser_rgb_lmdb(self.txn, [0], frames_path, num_frames)
            ref = np.array(self._parser_rgb_lmdb(self.txn_anno, [0], masks_path, 1, flag='unchanged', backend='pillow')[0])
        
        original_shape = frames[0].shape[:2]

        data = {
            'imgs': frames,
            'ref_seg_map': ref,
            'video_path': sample['video_path'],
            'original_shape': original_shape,
            'modality': 'RGB',
            'num_clips': 1,
            'clip_len': num_frames
        }

        return self.pipeline(data)


    def davis_evaluate(self, results, output_dir, logger=None):
        dataset_eval = DAVISEvaluation(
            davis_root=self.root, task=self.task, gt_set='val')
        if isinstance(results, str):
            metrics_res = dataset_eval.evaluate(results)
        elif results == None:
            metrics_res = dataset_eval.evaluate(output_dir)
        else:
            assert len(results) == len(self)
            for vid_idx in range(len(self)):
                assert len(results[vid_idx]) == \
                       self.samples[vid_idx]['num_frames'] or \
                       isinstance(results[vid_idx], str)
            if output_dir is None:
                tmp_dir = tempfile.TemporaryDirectory()
                output_dir = tmp_dir.name
            else:
                tmp_dir = None
                mmcv.mkdir_or_exist(output_dir)

            if terminal_is_available():
                prog_bar = mmcv.ProgressBar(len(self))
            for vid_idx in range(len(results)):
                cur_results = results[vid_idx]
                if isinstance(cur_results, str):
                    file_path = cur_results
                    cur_results = np.load(file_path)
                    os.remove(file_path)
                for img_idx in range(
                        self.samples[vid_idx]['num_frames']):
                    result = cur_results[img_idx].astype(np.uint8)
                    img = Image.fromarray(result)
                    img.putpalette(
                        np.asarray(self.PALETTE, dtype=np.uint8).ravel())
                    video_path = self.samples[vid_idx]['video_path']
                    raw_img = Image.open(osp.join(video_path, self.filename_tmpl.format(img_idx)))

                    # blend
                    img_ = copy.deepcopy(img).convert('RGBA')
                    raw_img = raw_img.convert('RGBA')
                    blend_image = Image.blend(raw_img, img_, 0.7)

                    save_path = osp.join(
                    output_dir, osp.relpath(video_path, self.video_dir), 'blend',
                    self.filename_tmpl.format(img_idx).replace('.jpg', '_blend.png'))

                    save_path_mask = osp.join(
                    output_dir, osp.relpath(video_path, self.video_dir),
                    self.filename_tmpl.format(img_idx).replace(
                        'jpg', 'png'))

                    mmcv.mkdir_or_exist(osp.dirname(save_path))
                    mmcv.mkdir_or_exist(osp.dirname(save_path_mask))

                    # blend_image.save(save_path)
                    img.save(save_path_mask)

                if terminal_is_available():
                    prog_bar.update()
            metrics_res = dataset_eval.evaluate(output_dir)
            if tmp_dir is not None:
                tmp_dir.cleanup()

        J, F = metrics_res['J'], metrics_res['F']

        # Generate dataframe for the general results
        g_measures = [
            'J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall',
            'F-Decay'
        ]
        final_mean = (np.mean(J['M']) + np.mean(F['M'])) / 2.
        g_res = np.array([
            final_mean,
            np.mean(J['M']),
            np.mean(J['R']),
            np.mean(J['D']),
            np.mean(F['M']),
            np.mean(F['R']),
            np.mean(F['D'])
        ])
        g_res = np.reshape(g_res, [1, len(g_res)])
        print_log(f'\nGlobal results for {self.split}', logger=logger)
        table_g = pd.DataFrame(data=g_res, columns=g_measures)
        print_log('\n' + table_g.to_string(index=False), logger=logger)

        with open(osp.join(output_dir, 'result.txt'), 'a') as f:
            f.write(table_g.to_string(index=False) + '\n')

        # Generate a dataframe for the per sequence results
        seq_names = list(J['M_per_object'].keys())
        seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
        J_per_object = [J['M_per_object'][x] for x in seq_names]
        F_per_object = [F['M_per_object'][x] for x in seq_names]
        table_seq = pd.DataFrame(
            data=list(zip(seq_names, J_per_object, F_per_object)),
            columns=seq_measures)
        print_log(f'\nPer sequence results for  {self.split}', logger=logger)
        print_log('\n' + table_seq.to_string(index=False), logger=logger)

        with open(osp.join(output_dir, 'result.txt'), 'a') as f:
            f.write(table_seq.to_string(index=False) + '\n')

        eval_results = table_g.to_dict('records')[0]

        return eval_results

    def evaluate(self, results, metrics='davis', output_dir=None, logger=None):
        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['davis']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        eval_results = dict()
        if mmcv.is_seq_of(results, np.ndarray) and results[0].ndim == 4:
            num_feats = results[0].shape[0]
            for feat_idx in range(num_feats):
                cur_results = [result[feat_idx] for result in results]
                eval_results.update(
                    add_prefix(
                        self.davis_evaluate(cur_results, output_dir, logger),
                        prefix=f'feat_{feat_idx}'))
        elif mmcv.is_seq_of(results, list):
            num_feats = len(results[0])
            for feat_idx in range(num_feats):
                cur_results = [result[feat_idx] for result in results]
                eval_results.update(
                    add_prefix(
                        self.davis_evaluate(cur_results, output_dir, logger),
                        prefix=f'feat_{feat_idx}'))
        else:
            eval_results.update(
                self.davis_evaluate(results, output_dir, logger))
        copypaste = []
        for k, v in eval_results.items():
            if 'J&F' in k:
                copypaste.append(f'{float(v)*100:.2f}')
        print_log(f'Results copypaste  {",".join(copypaste)}', logger=logger)
        return eval_results


@DATASETS.register_module()
class VOS_davis_dataset_rgb(Video_dataset_base):
    def __init__(self, data_prefix, 
                       year='2017',
                       per_video=False,
                       temporal_sampling_mode='random',
                       **kwargs
                       ):
        super().__init__(**kwargs)

        self.data_prefix = data_prefix
        self.year = year
        self.per_video = per_video
        self.temporal_sampling_mode = temporal_sampling_mode
        self.load_annotations()

    def __len__(self):
        return len(self.samples)

    def load_annotations(self):
        
        self.samples = []
        self.mask_dir = osp.join(self.root, self.data_prefix['ANNO'])
        self.video_dir = osp.join(self.root, self.data_prefix['RGB'])
        list_path = osp.join(self.list_path, f'davis{self.year}_{self.split}_list.txt')

        with open(list_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                sample = dict()
                vname, num_frames = line.strip('\n').split()
                sample['masks_path'] = sorted(glob.glob(osp.join(self.mask_dir, vname, '*.png')))
                sample['frames_path'] = sorted(glob.glob(osp.join(self.video_dir, vname, '*.jpg')))
                sample['video_path'] = osp.join(self.video_dir, vname)
                sample['num_frames'] = int(num_frames)
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
    
    def prepare_test_data(self, idx):
        sample = self.samples[idx]
        frames_path = sample['frames_path']
        num_frames = sample['num_frames']

        offsets = [num_frames // 2]
        
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
    
    
@DATASETS.register_module()
class VOS_davis_dataset_mlm(Video_dataset_base):
    def __init__(self, data_prefix, 
                       mask_ratio=0.15,
                       vq_size=32,
                       with_mask=False,
                       year='2017',
                       load_to_ram=False,
                       **kwargs
                       ):
        super().__init__(**kwargs)

        self.data_prefix = data_prefix
        self.year = year
        self.load_to_ram = load_to_ram
        self.with_mask = with_mask

        self.vq_res = vq_size
        self.mask_ratio = mask_ratio

        self.load_annotations()

    def load_annotations(self):
        
        self.samples = []
        self.video_dir = osp.join(self.root, self.data_prefix['RGB'])
        self.mask_dir = osp.join(self.root, self.data_prefix['ANNO'])
        list_path = osp.join(self.list_path, f'davis{self.year}_{self.split}_list.txt')

        with open(list_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                sample = dict()
                vname, num_frames = line.strip('\n').split()
                sample['frames_path'] = sorted(glob.glob(osp.join(self.video_dir, vname, '*.jpg')))
                sample['masks_path'] = sorted(glob.glob(osp.join(self.mask_dir, vname, '*.png')))
                sample['num_frames'] = len(sample['frames_path'])
                
                if sample['num_frames'] < self.clip_length * self.step: continue
                
                if self.load_to_ram:
                    sample['frames'] = self._parser_rgb_rawframe([0], sample['frames_path'], sample['num_frames'], step=1)
                self.samples.append(sample)

    def prepare_test_data(self, idx):
        
        sample = self.samples[idx]
        frames_path = sample['frames_path']
        masks_path = sample['masks_path']
        num_frames = sample['num_frames']

        offsets = self.temporal_sampling(num_frames, self.num_clips, self.clip_length, self.step)
        frames = self._parser_rgb_rawframe(offsets, frames_path, self.clip_length, step=1)
        mask = self._parser_rgb_rawframe([offsets[0]], masks_path, 1, flag='unchanged', backend='pillow')[0]

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
        masks_path = sample['masks_path']
        num_frames = sample['num_frames']

        offsets = self.temporal_sampling(num_frames, self.num_clips, self.clip_length, self.step)

        if self.load_to_ram:
            frames = (sample['frames'])[offsets[0]:offsets[0]+self.clip_length]
        else:
            frames = self._parser_rgb_rawframe(offsets, frames_path, self.clip_length, step=1)

        if self.with_mask:
            masks = self._parser_rgb_rawframe(offsets, masks_path, self.clip_length, step=1, backend='pillow', flag='unchanged')

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
        if self.with_mask:
            data['masks'] = masks

        return self.pipeline(data)