#!/bin/bash

...


# Number of nodes
nnodes=1
# Number of tasks per node
nproc_per_node=8

export HF_HOME="./data/cache"

# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=1200  # Increase timeout to 20 minutes

# # Debugging
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export CUDA_LAUNCH_BLOCKING=1

# Optimizations
export NCCL_SHM_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=2400

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

cd /projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 srun /home/gmongaras/miniconda3/bin/torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29504 \
src/train.py
