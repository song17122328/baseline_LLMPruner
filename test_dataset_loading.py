#!/usr/bin/env python3
"""
测试各种数据集的加载
"""

from datasets import load_dataset

def test_bookcorpusopen():
    """测试 bookcorpusopen"""
    print("\n" + "="*80)
    print("测试 bookcorpusopen")
    print("="*80)
    try:
        dataset = load_dataset('bookcorpusopen/bookcorpusopen', split='train', trust_remote_code=True)
        print(f"✓ 成功! 数据集大小: {len(dataset)}")
        print(f"  示例文本 (前100字符): {dataset[0]['text'][:100]}")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False

def test_c4():
    """测试 C4"""
    print("\n" + "="*80)
    print("测试 C4 (streaming mode)")
    print("="*80)
    try:
        dataset_stream = load_dataset(
            'allenai/c4', 'en', split='train', streaming=True
        )
        # 取前10个样本测试
        samples = list(dataset_stream.take(10))
        print(f"✓ 成功! 获取了 {len(samples)} 个样本")
        print(f"  示例文本 (前100字符): {samples[0]['text'][:100]}")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False

def test_wikitext():
    """测试 wikitext-2"""
    print("\n" + "="*80)
    print("测试 wikitext-2")
    print("="*80)
    try:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        print(f"✓ 成功! 数据集大小: {len(dataset)}")
        print(f"  示例文本 (前100字符): {dataset[10]['text'][:100]}")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False

if __name__ == '__main__':
    print("测试可用的校准数据集...")
    print("\n推荐优先级: bookcorpusopen > C4 > wikitext-2")

    results = {}
    results['bookcorpusopen'] = test_bookcorpusopen()
    results['c4'] = test_c4()
    results['wikitext'] = test_wikitext()

    print("\n" + "="*80)
    print("总结")
    print("="*80)
    for name, success in results.items():
        status = "✓ 可用" if success else "✗ 不可用"
        print(f"{name}: {status}")

    if results['bookcorpusopen']:
        print("\n建议: 使用 bookcorpusopen (最接近原始bookcorpus)")
    elif results['c4']:
        print("\n建议: 使用 C4 (大规模通用文本，良好的替代方案)")
    else:
        print("\n建议: 使用 wikitext-2 (非最优，但可以运行)")
