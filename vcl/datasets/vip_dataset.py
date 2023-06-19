import copy
import os
import os.path as osp

import mmcv
import numpy as np
import scipy.io as sio
from mmcv.utils import print_log

from vcl.utils import add_prefix, terminal_is_available
from .video_dataset import Video_dataset_base
from .registry import DATASETS
from vcl.utils import *
import glob
import cv2


@DATASETS.register_module()
class vip_dataset_rgb(Video_dataset_base):

    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [191, 0, 0], [64, 128, 0], [191, 128, 0], [64, 0, 128],
               [191, 0, 128], [64, 128, 128], [191, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 191, 0], [128, 191, 0], [0, 64, 128],
               [128, 64, 128]]
    CLASSES = [
        'background', 'hat', 'hair', 'sun-glasses', 'upper-clothes', 'dress',
        'coat', 'socks', 'pants', 'gloves', 'scarf', 'skirt', 'torso-skin',
        'face', 'right-arm', 'left-arm', 'right-leg', 'left-leg', 'right-shoe',
        'left-shoe'
    ]

    def __init__(self,
                 **kwargs
                       ):
        super().__init__(**kwargs)
        self.load_annotations()

    
    def load_annotations(self):
        
        self.samples = []
        list_path = osp.join(self.list_path, 'val_videos.txt')
        self.video_dir = osp.join(self.root, 'Images')

        with open(list_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                # if idx >= 20: break
                sample = dict()
                vname = line.strip('\n')
                sample['frames_path'] = sorted(glob.glob(osp.join(self.root, 'Images' , vname, '*.jpg')))
                sample['num_frames'] = len(sample['frames_path'])
                sample['anno_path'] = sorted(glob.glob(osp.join(self.root, 'Annotations' , vname, '*.png')))
                sample['video_path'] = osp.join(self.root, 'Images', vname)
                
                if sample['num_frames'] < self.clip_length * self.step: continue
                
                self.samples.append(sample)
                
        logger = get_root_logger()
        logger.info(" Load dataset with {} videos ".format(len(self.samples)))

    def prepare_test_data(self, idx):
        sample = self.samples[idx]
        num_frames = sample['num_frames']
        anno_path = sample['anno_path']
        frames_path = sample['frames_path']
        
        # load frames and masks
        frames = self._parser_rgb_rawframe([0], frames_path, num_frames)
        ref = np.array(self._parser_rgb_rawframe([0], anno_path, 1, flag='unchanged', backend='pillow')[0])
        
        original_shape = frames[0].shape[:2]
        assert ref.shape == original_shape

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

    def vip_evaluate(self, results, output_dir, logger=None):
        from terminaltables import AsciiTable
        eval_results = {}
        # assert len(results) == len(self)
        for vid_idx in range(len(results)):
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
        pred_path = []
        gt_path = []
        for vid_idx in range(len(results)):
            cur_results = results[vid_idx]
            frame_dir = self.samples[vid_idx]['video_path']
            ann_frame_dir = frame_dir.replace('Images',
                                              'Annotations')
            frame_list = list(sorted(os.listdir(frame_dir)))
            ann_list = list(sorted(os.listdir(ann_frame_dir)))
            if isinstance(cur_results, str):
                file_path = cur_results
                cur_results = np.load(file_path)
                os.remove(file_path)
            for img_idx in range(self.samples[vid_idx]['num_frames']):
                result = cur_results[img_idx].astype(np.uint8)
                img = Image.fromarray(result)
                img.putpalette(
                    np.asarray(self.PALETTE, dtype=np.uint8).ravel())
                save_path = osp.join(
                    output_dir, osp.relpath(self.samples[vid_idx]['video_path'], self.video_dir),
                    self.filename_tmpl.format(img_idx).replace(
                        'jpg', 'png'))
                mmcv.mkdir_or_exist(osp.dirname(save_path))
                img.save(save_path)
                pred_path.append(save_path)
                gt_path.append(osp.join(ann_frame_dir, ann_list[img_idx]))
            if terminal_is_available():
                prog_bar.update()
        num_classes = len(self.CLASSES)
        class_names = self.CLASSES
        from vcl.core.evaluation.iou import mean_iou
        ret_metrics = mean_iou(
            pred_path, gt_path, num_classes, ignore_index=255)
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        metric = ['mIoU']
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        
        with open(osp.join(output_dir, 'result.txt'), 'a') as f:
            f.write(table.table + '\n')
        
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)
        
        with open(osp.join(output_dir, 'result.txt'), 'a') as f:
            f.write(table.table + '\n')

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i] / 100.0

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results

    def evaluate(self, results, metrics='mIoU', output_dir=None, logger=None):
        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['mIoU']
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
                        self.vip_evaluate(cur_results, output_dir, logger),
                        prefix=f'feat_{feat_idx}'))
        elif mmcv.is_seq_of(results, list):
            num_feats = len(results[0])
            for feat_idx in range(num_feats):
                cur_results = [result[feat_idx] for result in results]
                eval_results.update(
                    add_prefix(
                        self.vip_evaluate(cur_results, output_dir, logger),
                        prefix=f'feat_{feat_idx}'))
        else:
            eval_results.update(self.vip_evaluate(results, output_dir, logger))
        copypaste = []
        for k, v in eval_results.items():
            if 'mIoU' in k:
                copypaste.append(f'{float(v)*100:.2f}')
        print_log(f'Results copypaste  {",".join(copypaste)}', logger=logger)
        return eval_results