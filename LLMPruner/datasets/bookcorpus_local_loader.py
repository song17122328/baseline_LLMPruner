#!/usr/bin/env python3
"""
本地加载bookcorpus数据集的辅助函数
"""

import os
import tarfile
import requests
from tqdm import tqdm

BOOKCORPUS_URL = "https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2"
CACHE_DIR = os.path.expanduser("~/.cache/bookcorpus")

def download_bookcorpus():
    """下载bookcorpus数据集"""
    os.makedirs(CACHE_DIR, exist_ok=True)

    tar_path = os.path.join(CACHE_DIR, "bookcorpus.tar.bz2")
    extract_dir = os.path.join(CACHE_DIR, "extracted")

    # 检查是否已经下载
    if os.path.exists(extract_dir) and os.path.isdir(extract_dir):
        txt_files = [f for f in os.listdir(extract_dir) if f.endswith('.txt')]
        if len(txt_files) > 0:
            print(f"✓ Bookcorpus已存在，共 {len(txt_files)} 个文件")
            return extract_dir

    # 下载
    if not os.path.exists(tar_path):
        print(f"正在下载bookcorpus from {BOOKCORPUS_URL}")
        print(f"目标路径: {tar_path}")
        print("这可能需要一些时间...")

        response = requests.get(BOOKCORPUS_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(tar_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"✓ 下载完成")
    else:
        print(f"✓ 压缩包已存在: {tar_path}")

    # 解压
    if not os.path.exists(extract_dir):
        print("正在解压bookcorpus...")
        os.makedirs(extract_dir, exist_ok=True)

        with tarfile.open(tar_path, 'r:bz2') as tar:
            tar.extractall(extract_dir)

        print(f"✓ 解压完成")

    return extract_dir

def load_bookcorpus_texts(max_samples=None):
    """
    加载bookcorpus文本（按行组合成段落）

    Args:
        max_samples: 最多加载多少个文本样本，None表示全部加载

    Returns:
        list of str: 文本列表
    """
    extract_dir = download_bookcorpus()

    # 找到所有txt文件
    txt_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))

    print(f"找到 {len(txt_files)} 个bookcorpus文件")

    # 读取文本，按行读取并组合成段落
    texts = []
    lines_per_sample = 50  # 每个样本包含多少行（约一个段落）

    for txt_file in tqdm(txt_files, desc="加载bookcorpus"):
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                for line in f:
                    line = line.strip()
                    if line:  # 跳过空行
                        lines.append(line)

                        # 每积累lines_per_sample行就组成一个样本
                        if len(lines) >= lines_per_sample:
                            texts.append(' '.join(lines))
                            lines = []

                            # 如果达到了max_samples限制，直接返回
                            if max_samples and len(texts) >= max_samples:
                                print(f"✓ 已达到 {max_samples} 个样本限制")
                                return texts

                # 处理剩余的行
                if lines:
                    texts.append(' '.join(lines))

        except Exception as e:
            print(f"警告: 无法读取 {txt_file}: {e}")
            continue

        # 检查是否已达到样本数限制
        if max_samples and len(texts) >= max_samples:
            break

    print(f"✓ 成功加载 {len(texts)} 个文本段落")
    return texts

if __name__ == '__main__':
    # 测试
    print("测试bookcorpus本地加载器...")
    texts = load_bookcorpus_texts(max_samples=10)

    if texts:
        print(f"\n示例文本 (前200字符):")
        print(texts[0][:200])
        print(f"\n总共加载了 {len(texts)} 个文本")
    else:
        print("✗ 没有加载到任何文本")
