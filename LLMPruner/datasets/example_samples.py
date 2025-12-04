import random
import numpy as np
import torch

from datasets import load_dataset,load_from_disk
from torch.utils.data.dataset import Dataset

def get_c4(tokenizer, n_samples, seq_len):
    """
    从本地加载C4数据集
    优先使用本地arrow文件: /newdata/DataSets/c4/
    """
    try:
        # 优先从本地arrow文件加载
        print(f"尝试从本地加载 C4 数据集: /newdata/DataSets/c4/")
        traindata = load_dataset(
            'arrow',
            data_files='/newdata/DataSets/c4/data-00000-of-00001.arrow',
            split='train'
        )
        print(f"✓ 成功从本地加载 C4 数据集 ({len(traindata)} 样本)")
    except Exception as e:
        print(f"✗ 本地C4加载失败: {e}")
        print("回退到在线流式加载 C4...")
        # 回退到在线流式加载
        traindata = load_dataset(
            'allenai/c4', 'en', split='train', streaming=True
        )
        # 转换为可索引的列表（只取需要的样本数 * 100以提供足够选择空间）
        traindata = list(traindata.take(n_samples * 100))
        print(f"✓ 在线加载 C4 数据集 ({len(traindata)} 样本)")

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





def get_wikitext2(tokenizer, n_samples, seq_len):
    """
    从本地加载wikitext2数据集
    优先使用本地arrow文件: /newdata/DataSets/wikitext2/
    """
    try:
        print(f"尝试从本地加载 wikitext2 数据集: /newdata/DataSets/wikitext2/")
        traindata = load_from_disk("/newdata/DataSets/wikitext2")['train']
    except Exception as e:
        print(f"✗ 本地 wikitext-2加载失败: {e}")
        print("回退到在线加载 wikitext-2...")
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    dataset_name = "wikitext-2"

    print(f"使用数据集: {dataset_name}, 采样 {n_samples} 个样本, 序列长度 {seq_len}")

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = traindata[i]['text'] if isinstance(traindata[i], dict) else traindata[i]
            tokenized_sample = tokenizer(text, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0 )

def get_examples(dataset, tokenizer, n_samples, seq_len = 128):
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len)
    elif dataset == 'wikitext2':
        return get_wikitext2(tokenizer, n_samples, seq_len)
    else:
        raise NotImplementedError
