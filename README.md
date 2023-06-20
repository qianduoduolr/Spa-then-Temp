## [CVPR 2023] Spatial-then-Temporal Self-Supervised Learning for Video Correspondence



### Introduction

This is the official code for  "**Spatial-then-Temporal Self-Supervised Learning for Video Correspondence**" (CVPR'23).

<!-- ![](figure/framework.png) -->

<div  align="center">    
<img src="figure/framework.png"  height="340px"/> 
</div>


### Citation
If you find this repository useful for your research, please cite our paper:

```latex
@inproceedings{li2023spatial,
  title={Spatial-then-Temporal Self-Supervised Learning for Video Correspondence},
  author={Li, Rui and Liu, Dong},
  booktitle={CVPR},
  pages={2279--2288},
  year={2023}
}
```

### Prerequisites

* Python 3.7.11
* PyTorch 1.8.0
* mmcv-full == 1.3.18
* davis2017-evaluation


To get started, first please clone the repo
```
git clone https://github.com/qianduoduolr/Spa-then-Temp
```
Then, please run the following commands:
```
conda create -n spa_then_temp python=3.7.11
conda activate spa_then_temp

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
pip install -r requirements.txt
pip install future tensorboard

# setup for davis evaluation
git clone https://github.com/davisvideochallenge/davis2017-evaluation.git && cd davis2017-evaluation
python setup.py develop
```

### Model Zoo

|Backbone|Stride|J&F-Mean|mIoU|PCK@0.1| PCK@0.2|   Download   |                                                                     
|----| ---- | ---- | ---- | ----| ------ | ---- |
| ResNet-18 (Temporal) | 8 | 69.2    | 39.8 | 61.7   | 81.5   | [pretrain ckpt](https://drive.google.com/file/d/1roBP0p0n8mIt0HvBsk6Wkszc6vuVUbxE/view?usp=drive_link)
| ResNet-18 | 8 | 70.7     | 40.9   | 63.5   | 83.1 | [pretrain ckpt](https://drive.google.com/file/d/1kksNGOklpPmlai6ljAIu1-dAcxSgMgml/view?usp=drive_link)
| ResNet-50 |8 | 71.2     | 42.2   | 63.6   | 83.8| [pretrain ckpt](https://drive.google.com/file/d/190ND_w4yJC7njdWtmw5bJ-tl5VL8PSJ7/view?usp=drive_link)
| ResNet-18 | 4 |  73.5   |  42.0 |  65.6  | 83.8  |[pretrain ckpt](https://drive.google.com/file/d/1ZJHyWMOpWhfmX6vMX5Qkm_2qkqS0grNM/view?usp=drive_link)
| ResNet-50 |4 |  74.1   |  42.8  |  66.4   | 84.4 |[pretrain ckpt](https://drive.google.com/file/d/1HZYXhfaCDG9IeKX8ESBGHBn-kUrHC8Q8/view?usp=drive_link)

We have rerun part of the main experiments in our paper. The reproduced performance in this repo may be slightly different from the paper.

### Data Preparation
#### ImageNet
Install ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).


#### YouTube-VOS
Please download the zip file `train.zip` from the [official website](https://competitions.codalab.org/competitions/19544#participate-get-data). Then, unzip and place it to`data/ytv`. Besides, please move the `youtube2018_train.json` in `data/data_info/` to `data/ytv`.
#### DAVIS-2017
DAVIS-2017 dataset could be downloaded from the [official website](https://davischallenge.org/davis2017/code.html). We use the 480p validation set for evaluation. Please move the `davis2017_val_list.json` in `data/data_info/` to `data/davis/ImageSets`.

#### JHMDB
Please download the data (`Videos`, `Joint positions`) from [official website](http://jhmdb.is.tue.mpg.de/challenge/JHMDB/datasets), unzip and place them in `data/jhmdb`. Please move the `val_list.txt` in `data/data_info/` to `data/jhmdb`.

#### VIP
Please download the data (only the `VIP_Fine`) from [official website](https://onedrive.live.com/?authkey=%21ALDIzAGeuVz1wyA&id=F04A5473A61552B1%21161&cid=F04A5473A61552B1), unzip and place them in `data/vip`. 


The overall data structure is as followed:

```shell
├── data
│   ├── imagenet
│   │   ├── train
│   │   │   ├── n01440764/
│   │   │   │── ...
│   ├── ytv
│   │   ├── train
│   │   │   ├── JPEGImages
│   │   │   │   ├──003234408d/
│   │   │   │   ├──...
│   │   │   ├── youtube2018_train.json
│   ├── davis
│   │   ├── Annotations
│   │   │   ├── 480p
│   │   │   │   ├── bike-packing/
│   │   │   │   ├── ...
│   │   ├── ImageSets
│   │   │   ├── davis2017_val_list.json
│   │   │   ├── ...
│   │   ├── JPEGImages
│   │   │   ├── 480p
│   │   │   │   ├── bike-packing/
│   │   │   │   ├── ...
│   ├── jhmdb
│   │   ├── Rename_Images
│   │   │   ├── brush_hair/
│   │   │   ├── ...
│   │   ├── joint_positions
│   │   │   ├── brush_hair/
│   │   │   ├── ...
│   │   ├── val_list.txt
│   ├── vip
│   │   ├── VIP_Fine
│   │   │   ├── Annotations/
│   │   │   ├── Images/
│   │   │   ├── lists/
```

### Inference
<p float="left">
<!-- <img src="figure/vos1.gif" width="25%"> -->
<!-- <img src="figure/vos1.gif" width = "230" height = "160"> -->
<img src="figure/vos2.gif" width = "230" height = "160">
<img src="figure/pose1.gif" width = "230" height = "160">
<img src="figure/vip2.gif" width = "230" height = "160">


</p>

The evaluation is conducted on three correspondence-related tasks, including semi-supervised video object segmentation, keypoint tracking, and human part propagation. Here we show some evaluation results.

We follow the prior studies to leverage label propagation for inference, which can be achieved by:
```shell
bash tools/dist_test.sh ${CONFIG}  ${GPUS}  ${TASK}  ${CKPT}
```

Note the `TASK` consists of davis (for VOS), jhmdb (for keypoint tracking), and vip (for human part propagation).

We give a inference cmd example:

```shell
# testing ResNet-18 with a stride of 8 for video object segmentation
bash tools/dist_test.sh configs/test/res18_d8.py 4 davis ckpt/res18_d8_temp.pth
```
The results will be saved to `eval/`. If you have enough memory, you can set the `save_np=False`  and increase the `step` of test_cfg in config for faster inference.

### Tranining
Our training pipeline consists of two seperate steps, including spatial and temporal feature learning. 
#### The first step
In the first step, we learn the spatial-discriminative features with contrastive learning. Please refer to these  official implementations ([MoCo](), [DetCo]()) for image-level self-supervised learning.


#### The second step
Then we perform temporal feature learning on YouTube-VOS:

```shell
bash tools/dist_train.sh configs/train/train.py 4
```
Please modify the `pretrained` for `backbone_t` in `configs/train/train.py`, in order to regard the model learned in the first step as teacher.


### Acknowledgement
The codebase is implemented based on the [MMCV](https://github.com/open-mmlab/mmcv) and [VFS](https://github.com/xvjiarui/VFS). Thanks for these excellent open source repositories.