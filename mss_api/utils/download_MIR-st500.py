import json
import subprocess
import os
import argparse

def batch_download_youtube_audio(json_path, output_dir):
    """
    从JSON文件批量下载YouTube音频并按序号命名为wav格式
    :param json_path: MIR-ST500_link.json的路径
    :param output_dir: 音频保存目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取JSON文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            link_dict = json.load(f)
        print(f"✅ 成功读取JSON文件，共{len(link_dict)}条音频链接")
    except Exception as e:
        print(f"❌ 读取JSON文件失败：{e}")
        return
    
    # 遍历链接批量下载
    total = len(link_dict)
    success_count = 0
    fail_list = []
    
    for idx, (seq_num, url) in enumerate(link_dict.items(), 1):
        # 输出文件名：序号.wav（如1.wav、10.wav、100.wav）
        output_path = os.path.join(output_dir, f"{seq_num}.wav")
        
        # 跳过已下载的文件
        if os.path.exists(output_path):
            print(f"[{idx}/{total}] ⏩ {seq_num}.wav 已存在，跳过")
            success_count += 1
            continue
        
        # 构建yt-dlp命令
        cmd = [
            'yt-dlp',
            '--no-check-certificate',
            '--extractor-args', 'youtube:skip=js',  # 保留跳过JS
            '-x',  # 仅提取音频
            '--audio-format', 'wav',  # 输出格式为wav
            '--audio-quality', '0',  # 最高音质（避免压缩损失）
            '-o', output_path,  # 输出路径
            url  # YouTube链接
        ]
        
        try:
            print(f"[{idx}/{total}] 📥 正在下载：{seq_num}.wav ({url})")
            # 执行下载命令（隐藏冗余输出，仅保留错误信息）
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,  # 屏蔽标准输出
                stderr=subprocess.PIPE,     # 捕获错误输出
                check=True
            )
            print(f"[{idx}/{total}] ✅ 下载完成：{seq_num}.wav")
            success_count += 1
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8', errors='ignore')[:200]  # 截取错误信息
            print(f"[{idx}/{total}] ❌ 下载失败：{seq_num}.wav | 错误：{error_msg}")
            fail_list.append((seq_num, url, error_msg))
        except Exception as e:
            print(f"[{idx}/{total}] ❌ 下载异常：{seq_num}.wav | 错误：{str(e)}")
            fail_list.append((seq_num, url, str(e)))
    
    # 输出下载总结
    print("\n" + "="*50)
    print(f"📊 下载完成总结：")
    print(f"   总数量：{total}")
    print(f"   成功：{success_count}")
    print(f"   失败：{len(fail_list)}")
    if fail_list:
        print(f"\n❌ 失败列表（共{len(fail_list)}条）：")
        for seq_num, url, err in fail_list[:10]:  # 仅显示前10条失败记录
            print(f"   {seq_num}.wav | {url} | 错误：{err}")
        if len(fail_list) > 10:
            print(f"   ... 还有{len(fail_list)-10}条失败记录未显示")

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='批量下载MIR-ST500 YouTube音频')
    parser.add_argument('--json_path', type=str, required=True, 
                        help='MIR-ST500_link.json的路径（如./MIR-ST500_link.json）')
    parser.add_argument('--output_dir', type=str, default='./MIR-ST500_audio', 
                        help='音频保存目录（默认：./MIR-ST500_audio）')
    args = parser.parse_args()
    
    # 执行批量下载
    batch_download_youtube_audio(args.json_path, args.output_dir)

if __name__ == '__main__':
    main()