
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
#     --nnodes=1 \
#     --nproc_per_node=4 \
#     EAE_main.py ./config/zh/eae/duee.yaml

# ED 
CUDA_VISIBLE_DEVICES=3 python main.py config.yaml

# EAE
# CUDA_VISIBLE_DEVICES=2 python EAE_main.py EAE_config.yaml 