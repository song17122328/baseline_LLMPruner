#!/usr/bin/env python3
"""
检查剪枝后K和V的维度是否一致
"""

import torch

def check_kv_consistency(ckpt_path):
    """检查K和V维度是否一致"""
    print(f"加载模型: {ckpt_path}")
    pruned_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = pruned_dict['model']

    print(f"\n{'='*80}")
    print("检查所有层的K/V维度一致性:")
    print(f"{'='*80}\n")

    issues = []

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn

        k_dim = attn.k_proj.weight.data.shape[0]
        v_dim = attn.v_proj.weight.data.shape[0]

        if k_dim != v_dim:
            issues.append({
                'layer': layer_idx,
                'k_dim': k_dim,
                'v_dim': v_dim,
                'diff': abs(k_dim - v_dim)
            })
            print(f"❌ Layer {layer_idx}: K={k_dim}, V={v_dim} (差值={k_dim - v_dim})")

    if not issues:
        print("✅ 所有层的K/V维度都一致")
    else:
        print(f"\n{'='*80}")
        print(f"❌ 发现 {len(issues)} 个层存在K/V维度不一致!")
        print(f"{'='*80}")
        print("\n这就是PPL爆炸的根本原因!")
        print("在GQA架构中,K和V必须共享相同的key-value heads。")
        print("如果K/V维度不一致,attention计算会完全错误。")
        print("\n解决方案:")
        print("1. 在llama3.py的consecutive_groups中同时添加k_proj和v_proj")
        print("2. 或者使用channel_groups将k_proj和v_proj绑定在一起")

    return issues

if __name__ == '__main__':
    import sys
    ckpt_path = 'prune_log/llama3_prune/pytorch_model.bin'
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]

    check_kv_consistency(ckpt_path)
