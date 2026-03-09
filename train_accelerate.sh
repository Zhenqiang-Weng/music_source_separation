#!/bin/bash
# train_accelerate.sh
trap 'echo "Ctrl+C 清理进程中..."; pkill -f train_accelerate_fixed.py; exit 0' INT

# 如果需要在线同步，请填写 WANDB_KEY 并移除/注释 WANDB_MODE 设置
# export WANDB_MODE="offline"
WANDB_KEY="456ec973dc00b6906905122761925c11a860e679"

export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1          # 若没有 InfiniBand
export NCCL_P2P_DISABLE=1        # 默认即可；若有报错可暂时设1
export NCCL_SOCKET_IFNAME=eth0    # 换成实际网卡，如 ens3
export GLOO_SOCKET_IFNAME=eth0
export MASTER_PORT=$((10000 + RANDOM % 50000))  # 随机一个空闲端口


# accelerate config
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# PYTHONWARNINGS="ignore" accelerate launch --mixed_precision bf16 --num_processes 8 --main_process_port ${MASTER_PORT} train_accelerate_bf16.py \
#     --model_type spk_bs_roformer_exportable \
#     --config_path ckpts/spk_bs_roformer/config_spk.yaml \
#     --start_check_point "results/results_spk_bs_roformer-with-large-window/model_spk_bs_roformer_ep_68_sisdr_11.7275.ckpt" \
#     --results_path results/results_spk_bs_roformer-with-large-window \
#     --data_path "/ailab-train/speech/wengzhenqiang/Music-Source-Separation-Training/dataset/backttracks4all_organized2" \
#     --valid_path "/ailab-train/speech/wengzhenqiang/Music-Source-Separation-Training/dataset/extreme_music_valid" \
#     --dataset_type 4 \
#     --num_workers 4 \
#     --device_ids 0 1 2 3 4 5 6 7 \
#     --wandb_project "msst-accelerate" \
#     --wandb_name "bandroformer-spk-$(date +%Y%m%d_%H%M%S)" \
#     --wandb_key "${WANDB_KEY:-}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
PYTHONWARNINGS="ignore" accelerate launch --mixed_precision bf16 --num_processes 6 --main_process_port ${MASTER_PORT} train_accelerate_bf16.py \
    --model_type spk_bs_roformer_exportable \
    --config_path ckpts/spk_bs_roformer/config_spk.yaml \
    --start_check_point "results/results_spk_bs_roformer-with-large-window/model_spk_bs_roformer_ep_68_sisdr_11.7275.ckpt" \
    --results_path results/results_spk_bs_roformer-with-large-window \
    --data_path "/ailab-train/speech/wengzhenqiang/Music-Source-Separation-Training/dataset/backttracks4all_organized2" \
    --valid_path "/ailab-train/speech/wengzhenqiang/Music-Source-Separation-Training/dataset/extreme_music_valid" \
    --dataset_type 4 \
    --num_workers 4 \
    --device_ids 0 1 2 3 4 5 \
    --wandb_project "msst-accelerate" \
    --wandb_name "bandroformer-spk-$(date +%Y%m%d_%H%M%S)" \
    --wandb_key "${WANDB_KEY:-}"

