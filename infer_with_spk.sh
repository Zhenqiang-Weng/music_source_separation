#!/bin/bash
set -e

# START_TIME=$(date +%s)
# export CUDA_VISIBLE_DEVICES=4
# PYTHONWARNINGS="ignore" python inference_with_spk.py \
#     --use_diffusion \
#     --diffusion_steps 10 \
#     --diffusion_model_path results/results_mdx_diff/last_additional_model_ckpt \
#     --model_type spk_bs_roformer \
#     --config_path ckpt/spk_bs_roformer/config_harmoney.yaml \
#     --start_check_point  results/results_spk_bs_roformer/model_spk_bs_roformer_ep_9_sisdr_9.5943.ckpt \
#     --input_folder dataset/lead_back_vocals  \
#     --store_dir test_sample/lead_back_vocals-spk \
#     --extract_other


# START_TIME=$(date +%s)
# export CUDA_VISIBLE_DEVICES=4
# PYTHONWARNINGS="ignore" python inference_with_spk.py \
#     --model_type bdc_sg_bs_roformer \
#     --config_path ckpt/bdc_sg_bs_roformer/config.yaml \
#     --start_check_point  results/results_bdc_sg_bs_roformer/model_bdc_sg_bs_roformer_ep_41_sisdr_9.5502.ckpt \
#     --input_folder /user-fs/chenzihao/chengongyu/svc/seed-vc/test/moban/november/distortion/raw  \
#     --store_dir test_sample/11poyin-bdc-sg-bs \
#     --extract_other

# START_TIME=$(date +%s)
# export CUDA_VISIBLE_DEVICES=4
# PYTHONWARNINGS="ignore" python inference_with_spk.py \
#     --model_type bdc_sg_bs_roformer \
#     --config_path ckpt/bdc_sg_bs_roformer/config.yaml \
#     --start_check_point  results/results_bdc_sg_bs_roformer/model_bdc_sg_bs_roformer_ep_64_sisdr_9.6411.ckpt \
#     --input_folder dataset/multi_song  \
#     --store_dir test_sample/multi_song_spk_bdc_sg_bs_9.6411 \
#     --extract_other

# START_TIME=$(date +%s)
# export CUDA_VISIBLE_DEVICES=4
# PYTHONWARNINGS="ignore" python inference_with_spk.py \
#     --model_type spk_bs_roformer \
#     --config_path ckpt/spk_bs_roformer/config_harmoney.yaml \
#     --start_check_point  results/results_spk_bs_roformer/model_spk_bs_roformer_ep_5_sisdr_9.8275.ckpt \
#     --input_folder dataset/e_bv_v_25s_s_only_mix  \
#     --store_dir test_sample/e_bv_v_25s_s_only_mix_spk_bs_roformer_ep_5_sisdr_9.8275 \
#     --extract_other

# START_TIME=$(date +%s)
# export CUDA_VISIBLE_DEVICES=4
# PYTHONWARNINGS="ignore" python inference_with_spk.py \
#     --model_type spk_bs_roformer \
#     --config_path ckpt/spk_bs_roformer/config_harmoney.yaml \
#     --start_check_point  results/results_spk_bs_roformer/model_spk_bs_roformer_ep_5_sisdr_9.8275.ckpt \
#     --input_folder test_sample/tmp  \
#     --store_dir test_sample/tmp—inf \
#     --extract_other

# START_TIME=$(date +%s)
# export CUDA_VISIBLE_DEVICES=4
# PYTHONWARNINGS="ignore" python inference_with_spk.py \
#     --model_type spk_bs_roformer \
#     --config_path ckpt/spk_bs_roformer/config_harmoney.yaml \
#     --start_check_point results/results_spk_bs_roformer/model_spk_bs_roformer_ep_5_sisdr_9.8275.ckpt \
#     --input_folder dataset/xianshang1223 \
#     --store_dir test_sample/xianshang1223 \
#     --extract_instrumental

START_TIME=$(date +%s)
export CUDA_VISIBLE_DEVICES=4
PYTHONWARNINGS="ignore" python inference_with_spk.py \
    --model_type bdc_sg_bs_roformer \
    --config_path ckpt/bdc_sg_bs_roformer/config.yaml \
    --start_check_point results/results_bdc_sg_bs_roformer/model_bdc_sg_bs_roformer_ep_49_sisdr_9.4473.ckpt \
    --input_folder dataset/xianshang1223 \
    --store_dir test_sample/xianshang1223-dbc-sg \


# dataset/medley_vox_mix
# dataset/lead_back_vocals
# test_sample/raw
# /user-fs/chenzihao/chengongyu/svc/seed-vc/test/xianshang/11.25raw
# /user-fs/chenzihao/chengongyu/svc/seed-vc/test/moban/november/distortion/raw
# /user-fs/chenzihao/chengongyu/svc/seed-vc/test/xianshang/moban1223

STATUS=$?
END_TIME=$(date +%s)

if [ $STATUS -eq 0 ]; then
    echo "[INFO] Inference completed successfully."
else
    echo "[ERROR] Inference failed!" >&2
fi
echo "[INFO] Total time: $((END_TIME-START_TIME)) seconds."