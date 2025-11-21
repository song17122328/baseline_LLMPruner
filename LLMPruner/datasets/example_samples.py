import random
import numpy as np
import torch

from datasets import load_dataset
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

def get_bookcorpus(tokenizer, n_samples, seq_len):
    """
    获取bookcorpus数据集

    优先级:
    1. 尝试本地下载并加载原始bookcorpus (最佳，与官方一致)
    2. 回退到C4数据集 (大规模通用文本，与bookcorpus类似)
    3. 最后回退到wikitext-2 (不推荐，但保证能运行)
    """
    traindata = None
    dataset_name = "unknown"

    # 方案1: 本地下载并加载原始bookcorpus (推荐)
    try:
        print("尝试本地加载 bookcorpus...")
        from LLMPruner.datasets.bookcorpus_local_loader import load_bookcorpus_texts

        # 加载bookcorpus文本，只需要足够的样本即可
        texts = load_bookcorpus_texts(max_samples=n_samples * 10)

        if texts and len(texts) >= n_samples:
            # 转换为datasets格式
            traindata = [{'text': text} for text in texts]
            dataset_name = "bookcorpus (local)"
            print(f"✓ 成功加载本地 bookcorpus ({len(traindata)} 样本)")
    except Exception as e:
        print(f"✗ 本地bookcorpus加载失败: {e}")
        print(f"  提示: 这会自动下载~2GB的bookcorpus数据集")

    # 方案2: 回退到C4 (推荐的替代方案)
    if traindata is None:
        try:
            print("回退到 C4 数据集...")
            traindata_stream = load_dataset(
                'allenai/c4', 'en', split='train', streaming=True
            )
            # 转换为可索引列表
            traindata = list(traindata_stream.take(n_samples * 100))
            dataset_name = "C4"
            print(f"✓ 成功加载 {dataset_name} ({len(traindata)} 样本)")
        except Exception as e:
            print(f"✗ C4 加载失败: {e}")

    # 方案3: 最后回退到 wikitext-2 (不推荐)
    if traindata is None:
        try:
            print("回退到 wikitext-2 数据集...")
            traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
            traindata = [{'text': item['text']} for item in traindata]
            dataset_name = "wikitext-2"
            print(f"⚠️  使用 {dataset_name} (可能影响剪枝质量)")
        except Exception as e:
            raise RuntimeError(f"所有数据集加载都失败了: {e}")

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


def get_wiki(tokenizer, n_samples, seq_len):
    from datasets import load_from_disk
    traindata = load_from_disk("/newdata/DataSets/wikitext2")['train']
    traindata = [{'text': item['text']} for item in traindata]
    dataset_name = "wikitext-2"

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
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    elif dataset == 'wikitext2':
        return get_wiki(tokenizer, n_samples, seq_len)
    else:
        raise NotImplementedError
