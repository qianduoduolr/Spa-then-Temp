exp_name = 'train'

# model settings
model = dict(
    type='Framework',
    backbone=dict(type='ResNet',depth=50, strides=(1, 2, 2, 4), out_indices=(1, 2, 3), pool_type='none', pretrained='/path/to/pretrain'), 
    # Put pre-trained model here as teacher. The more strong the teacher is, the better results we can get.
    backbone_t=dict(type='ResNet',depth=50, strides=(1, 2, 2, 2), out_indices=(3,),pretrained='/path/to/pretrain'), 
    loss=dict(type='MSELoss',reduction='mean'),
    feat_size=[64, 32],
    radius=[12, 6],
    downsample_rate=[4, 8],
    detach=True,
    loss_weight = dict(stage0_l1_loss=1, stage1_l1_loss=1, local_corrdis_loss=1000, global_corrdis_loss=50),
    pretrained=None
)

model_test = None

# model training and testing settings
train_cfg = dict(syncbn=True)

test_cfg = dict(
    precede_frames=20,
    topk=10,
    temperature=0.07,
    strides=(1, 2, 2, 1),
    out_indices=(2, ),
    neighbor_range=48,
    with_first=True,
    with_first_neighbor=True,
    output_dir='eval_results')

# dataset settings
train_dataset_type = 'VOS_youtube_dataset_rgb'

val_dataset_type = 'VOS_davis_dataset_test'
test_dataset_type = 'VOS_davis_dataset_test'


# train_pipeline = None
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
img_norm_cfg_lab = dict(mean=[50, 0, 0], std=[50, 127, 127], to_bgr=False)

train_pipeline = [
    dict(type='RandomResizedCrop', area_range=(0.6,1.0), aspect_ratio_range=(1.5, 2.0),),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='RGB2LAB', output_keys='images_lab'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Normalize', **img_norm_cfg_lab, keys='images_lab'),
    dict(type='FormatShape', input_format='NPTCHW'),
    dict(type='FormatShape', input_format='NPTCHW', keys='images_lab'),
    dict(type='Collect', keys=['imgs', 'images_lab'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'images_lab'])
]

val_pipeline = [
    dict(type='Resize', scale=(-1, 480), keep_ratio=True),
    dict(type='Flip', flip_ratio=0),
    dict(type='RGB2LAB'),
    dict(type='Normalize', **img_norm_cfg_lab),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'ref_seg_map'],
        meta_keys=('video_path', 'original_shape')),
    dict(type='ToTensor', keys=['imgs', 'ref_seg_map'])
]

# demo_pipeline = None
data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=16, drop_last=True),  # 4 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
     train=
            dict(
            type=train_dataset_type,
            root='/data/ytv',
            list_path='/data/ytv/train',
            data_prefix=dict(RGB='train/JPEGImages', FLOW=None, ANNO='train/Annotations'),
            clip_length=2,
            pipeline=train_pipeline,
            test_mode=False),

    test =  dict(
            type=test_dataset_type,
            root='/data/davis',
            list_path='/data/davis/ImageSets',
            data_prefix='2017',
            pipeline=val_pipeline,
            test_mode=True
            ),
    
    val =  dict(
            type=val_dataset_type,
            root='/data/davis',
            list_path='/data/davis/ImageSets',
            data_prefix='2017',
            pipeline=val_pipeline,
            test_mode=True
            ),
)
# optimizer
optimizers = dict(
    backbone=dict(type='Adam', lr=0.00005, betas=(0.9, 0.999))
    )

# learning policy
runner_type='epoch'
max_epoch=800
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.001,
    by_epoch=False,
    warmup='linear',
    warmup_iters=20,
    warmup_ratio=0.001,
    warmup_by_epoch=True
    )

checkpoint_config = dict(interval=1600, save_optimizer=True, by_epoch=True)
# remove gpu_collect=True in non distributed training
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False, interval=10),
    ])

visual_config = None


# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./output/{exp_name}'


evaluation = None

eval_config= dict(
                  output_dir=f'{work_dir}/eval_output',
                  checkpoint_path=f'{work_dir}/epoch_{max_epoch}.pth'
                )


load_from = None
resume_from = None
ddp_shuffle = True
workflow = [('train', 1)]