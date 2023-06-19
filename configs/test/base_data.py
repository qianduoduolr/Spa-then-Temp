img_norm_cfg_lab = dict(mean=[50, 0, 0], std=[50, 127, 127], to_bgr=False)

val_pipeline_davis = [
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

val_pipeline_jhmdb = [
    dict(type='Resize', scale=(320, 320), keep_ratio=False),
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

val_pipeline_vip = [
    dict(type='Resize', scale=(560, 560), keep_ratio=False),
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
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),
    
    test_davis =  dict(
            type='VOS_davis_dataset_test',
            root='/data/davis',
            list_path='/data/davis/ImageSets',
            data_prefix='2017',
            pipeline=val_pipeline_davis,
            test_mode=True
            ),

    test_jhmdb =  dict(
            type='jhmdb_dataset_rgb',
            root='/data/',
            list_path='/data/jhmdb',
            split='val',
            pipeline=val_pipeline_jhmdb,
            test_mode=True
            ),
    
    test_vip =  dict(
            type='vip_dataset_rgb',
            root='/data/vip/VIP_Fine',
            list_path='/data/vip/VIP_Fine/lists',
            split='val',
            pipeline=val_pipeline_vip,
            test_mode=True
            ),
    
)