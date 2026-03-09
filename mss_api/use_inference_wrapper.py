import os
from types import SimpleNamespace # 引入 SimpleNamespace 用于构建配置
# 引入分布式推理函数
from inference_wrapper import AudioSeparator, run_distributed_inference

def main():
    # =======================================================
    # 配置路径变量 (所有场景通用)
    # =======================================================
    model_type = "bdc_sg_bs_roformer"
    config_path = "ckpt/bdc_sg_bs_roformer/config.yaml"
    ckpt_path = "ckpt/bdc_sg_bs_roformer/model_bdc_sg_bs_roformer_ep_20_sisdr_9.5412.ckpt"
    
    # 输出目录
    output_root = "/user-fs/chenzihao/wengzhenqiang/mss-api/test_sample/output_folder_distributed"


    # =======================================================
    # 1. 单进程模型初始化 (场景 1-4 需要这个)
    #    注意：如果你只跑场景 5 (多卡)，这部分初始化其实可以注释掉以节省显存。
    # =======================================================
    print("正在加载单进程模型...")
    separator = AudioSeparator(
        model_type=model_type,
        config_path=config_path,
        model_path=ckpt_path,
        device_ids=[4], 
    )
    print("模型加载完成！")


    # =======================================================
    # 场景 1: 处理单个文件
    # =======================================================
    single_file_path = "test_sample/tmp/1.mp3"
    output_root_1 = "test_sample/output_single"
    if os.path.exists(single_file_path):
        separator.inference_file(single_file_path, output_root_1)


    # =======================================================
    # 场景 2: 处理整个文件夹
    # =======================================================
    # folder_path = "test_sample/tmp"
    # if os.path.exists(folder_path):
    #     separator.inference_folder(folder_path, output_root)


    # =======================================================
    # 场景 3: 直接获取数据不保存
    # =======================================================
    # print(f"\n--- [场景 3] 读取数据到内存示例 ---")
    # if os.path.exists(single_file_path):
    #     data_dict = separator.separate_audio(single_file_path)
    #     print(f"获取到的音轨: {list(data_dict.keys())}")
    #     if 'vocals' in data_dict:
    #         print(f"人声数据形状 (Channels, Samples): {data_dict['vocals'].shape}")
    # else:
    #     print("文件不存在，跳过场景 3")

   

if __name__ == "__main__":
    main()