import enum
import _init_paths
import mmcv
from vcl.utils import *
import glob
import os.path as osp
from PIL import Image
import tqdm




def blend_for_jhmdb():
    data_root = '/home/lr/dataset/JHMDB/Rename_Images'
    input_dir = '/home/lr/expdir/VCL/group_vqvae_tracker/final_framework_v2_11/pose_eval_outputindices2/Rename_Images'

    video_list = glob.glob(osp.join(input_dir, '*/*/'))
    for v in video_list:
        results = sorted(glob.glob(osp.join(v, '*.png')))
        for idx, r in enumerate(results):
            x = Image.open(r)
            v_path = '/'.join(r.split('/')[-3:-1])
            raw_img = Image.open(osp.join(data_root, v_path, '{:05d}.png'.format(idx+1)))
            out = blend_image(raw_img, x, 0.8)
            output_path = r.replace('Rename_Images', 'Rename_Images_blend')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out.save(output_path)
            
            
def blend_for_vip():
    data_root = '/var/dataset/VIP/VIP_Fine/Images'
    input_dir = '/home/lr/expdir/VCL/group_stsl/res18_d8/eval_outputindices2vip'
    video_list = glob.glob(osp.join(input_dir, '*/'))
    for v in video_list:
        results = sorted(glob.glob(osp.join(v, '*.png')))
        v_paths = sorted(glob.glob(osp.join(data_root, results[0].split('/')[-2],'*.jpg')))
        for idx, r in enumerate(results):
            x = Image.open(r)
            raw_img = Image.open(v_paths[idx])
            out = blend_image(raw_img, x, 0.65)
            output_path = r.replace('eval_outputindices2vip', 'eval_outputindices2vip_blend')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out.save(output_path)
            
def blend_for_vos():
    data_root = '/home/lr/dataset/DAVIS/JPEGImages/480p'
    input_dir = '/home/lr/expdir/VCL/group_vqvae_tracker/dist_nl_l2_layer4_mast_d2/eval_outputindices2'
    video_list = glob.glob(osp.join(input_dir, '*/'))
    for v in tqdm.tqdm(video_list,  total=len(video_list)):
        results = sorted(glob.glob(osp.join(v, '*.png')))
        v_paths = sorted(glob.glob(osp.join(data_root, results[0].split('/')[-2],'*.jpg')))
        for idx, r in enumerate(results):
            x = Image.open(r)
            raw_img = Image.open(v_paths[idx])
            out = blend_image(raw_img, x, 0.65)
            output_path = r.replace('eval_outputindices2', 'eval_outputindices2_blend')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out.save(output_path)
       
# blend_for_jhmdb()
blend_for_vip()
# blend_for_vos()