from PIL import Image
import lmdb
import os
import cv2
import io
import argparse
from tqdm import tqdm
import numpy as np
from glob import glob
from joblib import delayed, Parallel
import pickle
import collections


target = 256


def dumps_data(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)


def create_lmdb_video_dataset_rgb(root_path, dst_path, workers=-1, quality=100, resize=True, save_jpg=False, save_lmdb=True):

    videos = glob(os.path.join(root_path,'*'))
    print('begin')

    def make_video(video_path, dst_path=None, resize=True, save_jpg=False):
        if save_jpg:
            dst_file_jpg = video_path.replace('JPEGImages', f'JPEGImages_s{target}')

        dst_file = dst_path            
        
        if save_lmdb:
            os.makedirs(dst_file, exist_ok=True)
        if save_jpg:
            os.makedirs(dst_file_jpg, exist_ok=True)

        frames = []
        idxs = []
        frame_names = sorted(glob(os.path.join(video_path, '*.jpg')))
        for frame_name in frame_names:
            frame = cv2.imread(frame_name)
            
            if frame is None:
                return 
            
            h, w, c = frame.shape

            if resize:
                if w >= h:
                    size = (int(target * w / h), int(target))
                else:
                    size = (int(target), int(target * h / w))

                frame = cv2.resize(frame, size, cv2.INTER_CUBIC)
            
            if save_jpg:
                file = os.path.join(dst_file_jpg, os.path.basename(frame_name))
                if os.path.exists(file):
                    continue
                cv2.imwrite(file, frame)

            frames.append(frame)
            idxs.append(os.path.basename(frame_name))

        if save_lmdb:
            _, frame_byte = cv2.imencode('.jpg', frames[0],  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            env = lmdb.open(dst_file, frame_byte.nbytes * len(frames) * 50)
            frames_num = len(frames)
            for i in range(frames_num):
                txn = env.begin(write=True)
                key = os.path.basename(frame_names[i])
                frame = frames[i]
                _, frame_byte = cv2.imencode('.jpg', frame,  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                txn.put(key.encode(), frame_byte)
                txn.commit()

            with open(os.path.join(dst_file, 'split.txt'),'a') as f:
                for idx in idxs:
                    f.write(idx + '\n')

    Parallel(n_jobs=workers)(delayed(make_video)(vp, dst_path, resize, save_jpg) for vp in tqdm(videos, total=len(videos)))



def create_lmdb_video_dataset_anno(root_path, dst_path, workers=-1, quality=100, resize=True, save_jpg=False, save_lmdb=True):

    videos = glob(os.path.join(root_path,'*'))
    print('begin')

    def make_video(video_path, dst_path=None, resize=True, save_jpg=False):
        if save_jpg:
            dst_file_jpg = video_path.replace('Annotations', f'Annotations_s{target}')

        dst_file = dst_path
            
        if save_lmdb:
            os.makedirs(dst_file, exist_ok=True)
        if save_jpg:
            os.makedirs(dst_file_jpg, exist_ok=True)

        frames = []
        idxs = []
        frame_names = sorted(glob(os.path.join(video_path, '*.png')))
        for frame_name in frame_names:
            frame = Image.open(frame_name).convert('P')
            w, h = frame.size

            if resize:
                if w >= h:
                    size = (int(target * w / h), int(target))
                else:
                    size = (int(target), int(target * h / w))

                frame = frame.resize(size, Image.NEAREST)
            
            if save_jpg:
                file = os.path.join(dst_file_jpg, os.path.basename(frame_name))
                frame.save(file)

            frames.append(frame)
            idxs.append(os.path.basename(frame_name))

        if save_lmdb:
            env = lmdb.open(dst_file, map_size=1099511627)
            frames_num = len(frames)
            for i in range(frames_num):
                txn = env.begin(write=True)
                key = os.path.basename(frame_names[i])
                frame = frames[i]
                # _, frame_byte = cv2.imencode('.png', frame)
                frame_byte = io.BytesIO()
                frame.save(frame_byte, format='PNG')
                frame_byte = frame_byte.getvalue()
                txn.put(key.encode(), frame_byte)
                txn.commit()

    Parallel(n_jobs=workers)(delayed(make_video)(vp, dst_path, resize, save_jpg) for vp in tqdm(videos, total=len(videos)))


def create_lmdb_video_dataset_rgb_v2(root_path, dst_path, workers=-1, quality=100, resize=True, save_jpg=False, save_lmdb=True):
        
    videos = glob(os.path.join(root_path,'*'))
    print('begin')
    os.makedirs(dst_path, exist_ok=True)
    env = lmdb.open(dst_path, map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    keys = collections.OrderedDict()  

    for idx, vp in enumerate(tqdm(videos, total=len(videos))):
       
        video_name = vp.split('/')[-1]
        frames = []
        idxs = []
        frame_names = sorted(glob(os.path.join(vp, '*.jpg')))
        for frame_name in frame_names:
            try:
                frame = cv2.imread(frame_name)
                h, w, c = frame.shape
                if resize:
                    if w >= h:
                        size = (int(target * w / h), int(target))
                    else:
                        size = (int(target), int(target * h / w))

                    frame = cv2.resize(frame, size, cv2.INTER_CUBIC)
                frames.append(frame)
                idxs.append(os.path.basename(frame_name))
            except Exception as e:
                print(e)
                print(frame_name)
                continue
            
        frames_num = len(frames)
        keys[video_name] = []
        for i in range(frames_num):
            txn = env.begin(write=True)
            key = os.path.basename(frame_names[i])
            key = f'{video_name}/{key}'
            frame = frames[i]
            _, frame_byte = cv2.imencode('.jpg', frame,  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            txn.put(key.encode(), frame_byte)
            txn.commit()
            keys[video_name].append(key.encode())
    
    with env.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_data(keys))
        txn.put(b'__len__', dumps_data(len(keys)))
        # txn.commit()
    
    print("Flushing database ...")
    
    env.sync()
    env.close()
    
def create_lmdb_video_dataset_anno_v2(root_path, dst_path, workers=-1, quality=100, resize=True, save_jpg=False, save_lmdb=True):
    
    videos = glob(os.path.join(root_path,'*'))
    print('begin')
    os.makedirs(dst_path, exist_ok=True)
    env = lmdb.open(dst_path, map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    keys = collections.OrderedDict()               
    for vp in tqdm(videos, total=len(videos)):
        video_name = vp.split('/')[-1]
        frames = []
        idxs = []
        frame_names = sorted(glob(os.path.join(vp, '*.png')))
        for frame_name in frame_names:
            frame = Image.open(frame_name).convert('P')
            w, h = frame.size
            frames.append(frame)
            idxs.append(os.path.basename(frame_name))

        frames_num = len(frames)
        keys[video_name] = []
        for i in range(frames_num):
            txn = env.begin(write=True)
            key = os.path.basename(frame_names[i])
            key = f'{video_name}/{key}'
            frame = frames[i]
            frame_byte = io.BytesIO()
            frame.save(frame_byte, format='PNG')
            frame_byte = frame_byte.getvalue()
            txn.put(key.encode(), frame_byte)
            txn.commit()
            keys[video_name].append(key.encode())
        
        
    with env.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_data(keys))
        txn.put(b'__len__', dumps_data(len(keys)))
        # txn.commit()
    
    print("Flushing database ...")
    
    env.sync()
    env.close()    


def check_data(path):
    env = lmdb.open(path, readonly=True)
    txn = env.begin(write=False)
    
    print('haha')
    

def creat_lmdb_main_ytvos(args):
    for mode in ['train_all_frames',]:
        for resize in [False, True]:
            if not resize:
                root_path = args.root_path.replace('mode', mode)
                dst_path = args.dst_path.replace('mode', mode)
                root_path_anno = args.root_path_anno.replace('mode', mode)
                dst_path_anno = args.dst_path_anno.replace('mode', mode)
                if os.path.exists(root_path_anno):
                    create_lmdb_video_dataset_anno_v2(root_path_anno, dst_path_anno, workers=args.num_workers, resize=resize, save_jpg=args.save_jpg)
                create_lmdb_video_dataset_rgb_v2(root_path, dst_path, workers=args.num_workers, resize=resize, save_jpg=args.save_jpg)
            else:
                root_path = args.root_path.replace('mode', mode)
                dst_path = args.dst_path.replace('mode', mode) + f'_s{target}'  
                create_lmdb_video_dataset_rgb_v2(root_path, dst_path, workers=args.num_workers, resize=resize, save_jpg=args.save_jpg)
                
def creat_lmdb_main_davis(args):
    
    create_lmdb_video_dataset_anno_v2(args.root_path_anno, args.dst_path_anno, workers=args.num_workers, resize=args.resize, save_jpg=args.save_jpg)
    create_lmdb_video_dataset_rgb_v2(args.root_path, args.dst_path, workers=args.num_workers, resize=args.resize, save_jpg=args.save_jpg)
         
                

def parse_option():
    parser = argparse.ArgumentParser('training')

    # dataset
    parser.add_argument('--root-path', type=str, default='/home/lr/dataset/YouTube-VOS/2018/mode/JPEGImages', help='path of original data')
    parser.add_argument('--dst-path', type=str, default='/home/lr/dataset//YouTube-VOS-lmdb-v2/2018/mode/JPEGImages', help='path to store generated data')
    parser.add_argument('--root-path-anno', type=str, default='/home/lr/dataset/YouTube-VOS/2018/mode/Annotations', help='path of original data')
    parser.add_argument('--dst-path-anno', type=str, default='/home/lr/dataset/YouTube-VOS-lmdb-v2/2018/mode/Annotations', help='path to store generated data')
    parser.add_argument('--num-workers', type=int, default=-1, help='num of workers to use')
    parser.add_argument('--resize', type=str, default='', help='path to store generated data')
    parser.add_argument('--save-jpg', type=str, default='', help='path to store generated data')


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args =  parse_option()
    # create_lmdb_video_dataset_rgb(args.root_path, args.dst_path, workers=args.num_workers, resize=args.resize, save_jpg=args.save_jpg)
    # create_lmdb_video_dataset_anno(args.root_path, args.dst_path, workers=args.num_workers, resize=args.resize, save_jpg=args.save_jpg)i
    # create_lmdb_video_dataset_rgb_v2(args.root_path, args.dst_path, workers=args.num_workers, resze=args.resize, save_jpg=args.save_jpg)
    # check_data('/home/lr/dataset/DAVIS-lmdb-v2/JPEGImages/480p')
    creat_lmdb_main_ytvos(args)
    # creat_lmdb_main_davis(args)