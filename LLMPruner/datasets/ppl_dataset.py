'''
Some of the code refer to
https://github.com/IST-DASLab/gptq/blob/main/datautils.py
'''

import random
import numpy as np
import torch

from datasets import load_dataset,load_from_disk
from torch.utils.data.dataset import Dataset

def get_wikitext2(seq_len, tokenizer):
    """
    从本地加载 wikitext2 数据集
    路径: /newdata/DataSets/wikitext2/
    """
    try:
        print("尝试从本地加载 wikitext2 数据集...")
        traindata = load_from_disk('/newdata/DataSets/wikitext2')['train']
        testdata = load_from_disk('/newdata/DataSets/wikitext2')['test']
        print(f"✓ 成功从本地加载 wikitext2")
    except Exception as e:
        print(f"✗ 本地wikitext2加载失败: {e}")
        print("回退到在线加载...")
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        print("✓ 在线加载 wikitext2 成功")
    return traindata, testdata

def get_ptb(seq_len, tokenizer):
    """
    从本地加载 PTB 数据集
    路径: /newdata/DataSets/ptb/
    """
    try:
        print("尝试从本地加载 PTB 数据集...")
        traindata = load_from_disk('/newdata/DataSets/ptb')['train']
        valdata = load_from_disk('/newdata/DataSets/ptb')['validation']
        print(f"✓ 成功从本地加载 PTB")
    except Exception as e:
        print(f"✗ 本地PTB加载失败: {e}")
        print("回退到在线加载...")
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
        valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
        print("✓ 在线加载 PTB 成功")
    return traindata, valdata

class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

def process_data(samples, tokenizer, seq_len, field_name):
    test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)
       

def get_loaders(name, tokenizer, seq_len=2048, batch_size = 8):
    if 'wikitext2' in name:
        train_data, test_data = get_wikitext2(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        train_data, test_data = get_ptb(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_data, test_loader