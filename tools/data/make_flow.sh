cd /gdata/lirui/project/vcl
python -m torch.distributed.launch --nproc_per_node=1 tools/data/prepare_youtube_flow.py