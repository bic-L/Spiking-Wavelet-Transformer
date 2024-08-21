export HOST_NODE_ADDR=127.0.0.1:2945
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_SOCKET_IFNAME=lo

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=$HOST_NODE_ADDR train_imagenet.py -c conf/imagenet/6_512_300E_t4.yml --model swformer --eval-only --haar_vth 0.5 --FL_blocks 2 --resume 6-512-block-2-vth-0.5-bs512-74.9800.pth.tar

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=$HOST_NODE_ADDR train_imagenet.py -c conf/imagenet/6_512_300E_t4.yml --model swformer --eval-only --haar_vth 1.0 --FL_blocks 2 --resume 6-512-block-2-vth-1.0-bs512-74.9499.pth.tar

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=$HOST_NODE_ADDR train_imagenet.py -c conf/imagenet/8_512_300E_t4.yml --model swformer --eval-only --haar_vth 0.5 --FL_blocks 2 --resume 8-512-block-2-vth0.5-bs512-75.1879.pth.tar

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=$HOST_NODE_ADDR train_imagenet.py -c conf/imagenet/8_512_300E_t4.yml --model swformer --eval-only --haar_vth 1.0 --FL_blocks 2 --resume 8-512-block-2-vth-1.-bs512-75.4259.pth.tar