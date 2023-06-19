_base_ = './base_data.py'

exp_name = 'res18_d8'

# model settings
model = dict(
    type='VanillaTracker',
    backbone=dict(type='ResNet',depth=18, strides=(1, 2, 2, 1), out_indices=(2,), pool_type='none'),
)

# model training and testing settings
test_cfg_davis = dict(
    precede_frames=20,
    topk=10,
    temperature=0.07,
    strides=(1, 2, 2, 1),
    out_indices=(2, ),
    neighbor_range=24,
    with_first=True,
    with_first_neighbor=True,
    save_np=True,
    mode='square',
    output_dir='eval_results')

test_cfg_jhmdb = dict(
    precede_frames=20,
    topk=10,
    temperature=0.07,
    strides=(1, 2, 2, 1),
    out_indices=(2, ),
    neighbor_range=10,
    with_first=True,
    save_np=True,
    mode='square',
    with_first_neighbor=True,
    output_dir='eval_results')

test_cfg_vip = dict(
    precede_frames=20,
    topk=10,
    temperature=0.07,
    strides=(1, 2, 2, 1),
    out_indices=(2, ),
    neighbor_range=32,
    with_first=True,
    save_np=True,
    mode='square',
    with_first_neighbor=True,
    output_dir='eval_results')


# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./eval/{exp_name}'

eval_config= dict(
                  output_dir=f'{work_dir}/eval_output',
                checkpoint_path=None
                )