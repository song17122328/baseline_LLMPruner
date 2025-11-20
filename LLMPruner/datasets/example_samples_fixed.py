import random
import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data.dataset import Dataset

def get_c4(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len )
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_bookcorpus(tokenizer, n_samples, seq_len):
    """
    获取bookcorpus数据集的替代方案

    优先级:
    1. 尝试加载bookcorpusopen (社区维护的bookcorpus镜像)
    2. 回退到C4数据集 (大规模通用文本，与bookcorpus类似)
    3. 最后回退到wikitext-2 (不推荐，但保证能运行)
    """
    traindata = None
    dataset_name = "unknown"

    # 方案1: 尝试使用 bookcorpusopen (社区镜像)
    try:
        print("尝试加载 bookcorpusopen 数据集...")
        traindata = load_dataset('bookcorpusopen/bookcorpusopen', split='train', trust_remote_code=True)
        dataset_name = "bookcorpusopen"
        print(f"✓ 成功加载 {dataset_name}")
    except Exception as e:
        print(f"✗ bookcorpusopen 加载失败: {e}")

    # 方案2: 回退到C4 (推荐的替代方案)
    if traindata is None:
        try:
            print("回退到 C4 数据集...")
            traindata = load_dataset(
                'allenai/c4', 'allenai--c4',
                data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
                split='train'
            )
            dataset_name = "C4"
            print(f"✓ 成功加载 {dataset_name}")
        except Exception as e:
            print(f"✗ C4 加载失败: {e}")

    # 方案3: 最后回退到 wikitext-2 (不推荐)
    if traindata is None:
        try:
            print("回退到 wikitext-2 数据集...")
            traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
            dataset_name = "wikitext-2"
            print(f"⚠️  使用 {dataset_name} (可能影响剪枝质量)")
        except Exception as e:
            raise RuntimeError(f"所有数据集加载都失败了: {e}")

    print(f"使用数据集: {dataset_name}, 采样 {n_samples} 个样本, 序列长度 {seq_len}")

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0 )

def get_examples(dataset, tokenizer, n_samples, seq_len = 128):
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len)
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    else:
        raise NotImplementedError
