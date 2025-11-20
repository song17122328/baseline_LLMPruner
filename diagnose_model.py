#!/usr/bin/env python3
"""
诊断剪枝后模型的维度配置
检查attention层的维度是否正确匹配
"""

import torch
import sys

def diagnose_pruned_model(ckpt_path):
    """检查剪枝后模型的维度配置"""
    print(f"加载模型: {ckpt_path}")
    pruned_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = pruned_dict['model']

    print(f"\n{'='*80}")
    print("模型配置:")
    print(f"{'='*80}")
    config = model.config
    print(f"原始配置:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  head_dim: {config.hidden_size // config.num_attention_heads}")

    print(f"\n{'='*80}")
    print("各层实际维度检查 (Layers 0-31):")
    print(f"{'='*80}")

    issues = []

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn

        # 获取实际权重维度
        q_dim = attn.q_proj.weight.data.shape[0]
        k_dim = attn.k_proj.weight.data.shape[0]
        v_dim = attn.v_proj.weight.data.shape[0]
        o_in_dim = attn.o_proj.weight.data.shape[1]

        # 获取配置的维度
        configured_num_heads = attn.num_heads
        configured_num_kv_heads = attn.num_key_value_heads
        configured_head_dim = attn.head_dim

        # 计算期望维度
        expected_q_dim = configured_num_heads * configured_head_dim
        expected_kv_dim = configured_num_kv_heads * configured_head_dim

        # 检查是否匹配
        q_match = (q_dim == expected_q_dim)
        k_match = (k_dim == expected_kv_dim)
        v_match = (v_dim == expected_kv_dim)
        o_match = (o_in_dim == q_dim)

        all_match = q_match and k_match and v_match and o_match

        if not all_match or layer_idx in [0, 1, 2, 3, 4, 15, 29, 30, 31]:
            # 打印有问题的层，或者边界层
            status = "✓" if all_match else "✗"
            print(f"\nLayer {layer_idx} {status}:")
            print(f"  配置: num_heads={configured_num_heads}, num_kv_heads={configured_num_kv_heads}, head_dim={configured_head_dim}")
            print(f"  期望维度: Q={expected_q_dim}, K/V={expected_kv_dim}")
            print(f"  实际维度: Q={q_dim} {'✓' if q_match else '✗'}, K={k_dim} {'✓' if k_match else '✗'}, V={v_dim} {'✓' if v_match else '✗'}, O_in={o_in_dim} {'✓' if o_match else '✗'}")

            if not all_match:
                issues.append({
                    'layer': layer_idx,
                    'q_dim': q_dim,
                    'k_dim': k_dim,
                    'v_dim': v_dim,
                    'o_in_dim': o_in_dim,
                    'expected_q': expected_q_dim,
                    'expected_kv': expected_kv_dim,
                    'configured_heads': configured_num_heads,
                    'configured_kv_heads': configured_num_kv_heads,
                    'head_dim': configured_head_dim
                })

    # MLP维度检查
    print(f"\n{'='*80}")
    print("MLP维度检查 (抽样):")
    print(f"{'='*80}")

    for layer_idx in [0, 4, 15, 29, 31]:
        layer = model.model.layers[layer_idx]
        mlp = layer.mlp

        gate_dim = mlp.gate_proj.weight.data.shape[0]
        up_dim = mlp.up_proj.weight.data.shape[0]
        down_in_dim = mlp.down_proj.weight.data.shape[1]

        mlp_match = (gate_dim == up_dim == down_in_dim)
        status = "✓" if mlp_match else "✗"

        print(f"Layer {layer_idx} {status}: gate={gate_dim}, up={up_dim}, down_in={down_in_dim}")

    # 总结
    print(f"\n{'='*80}")
    print("诊断总结:")
    print(f"{'='*80}")

    if issues:
        print(f"❌ 发现 {len(issues)} 个层存在维度不匹配问题:")
        for issue in issues:
            print(f"\n  Layer {issue['layer']}:")
            print(f"    - 配置: {issue['configured_heads']} heads, {issue['configured_kv_heads']} kv_heads, head_dim={issue['head_dim']}")
            print(f"    - Q维度: 实际={issue['q_dim']}, 期望={issue['expected_q']}, 差值={issue['q_dim'] - issue['expected_q']}")
            print(f"    - K维度: 实际={issue['k_dim']}, 期望={issue['expected_kv']}, 差值={issue['k_dim'] - issue['expected_kv']}")
            print(f"    - V维度: 实际={issue['v_dim']}, 期望={issue['expected_kv']}, 差值={issue['v_dim'] - issue['expected_kv']}")

            # 分析原因
            if issue['q_dim'] % issue['head_dim'] != 0:
                print(f"    ⚠️  Q维度({issue['q_dim']})不是head_dim({issue['head_dim']})的整数倍！")
            if issue['k_dim'] % issue['head_dim'] != 0:
                print(f"    ⚠️  K维度({issue['k_dim']})不是head_dim({issue['head_dim']})的整数倍！")

        print(f"\n🔍 可能的原因:")
        print(f"  1. 剪枝后的维度不满足head_dim的整数倍约束")
        print(f"  2. GQA约束未正确处理 (num_kv_heads必须能被num_heads整除)")
        print(f"  3. llama3.py:164-165的重新计算逻辑有问题")
    else:
        print("✅ 所有层的维度配置都正确匹配")
        print("\n如果PPL仍然很高，可能的其他原因:")
        print("  1. 剪枝策略选择的通道不合适")
        print("  2. 权重合并策略有问题")
        print("  3. 需要进行后续微调才能恢复性能")

    return issues

if __name__ == '__main__':
    ckpt_path = 'prune_log/llama3_prune/pytorch_model.bin'
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]

    issues = diagnose_pruned_model(ckpt_path)
