#!/usr/bin/env python3
"""
对比原始模型和剪枝后模型的权重统计
"""

import torch
from transformers import LlamaForCausalLM

def analyze_weights(model, name):
    """分析模型权重"""
    print(f"\n{'='*80}")
    print(f"{name} 权重统计:")
    print(f"{'='*80}\n")

    # 抽样检查几个关键层
    sample_layers = [0, 4, 15, 29, 31]

    for layer_idx in sample_layers:
        layer = model.model.layers[layer_idx]

        # Attention weights
        q_weight = layer.self_attn.q_proj.weight.data
        k_weight = layer.self_attn.k_proj.weight.data
        v_weight = layer.self_attn.v_proj.weight.data
        o_weight = layer.self_attn.o_proj.weight.data

        # MLP weights
        gate_weight = layer.mlp.gate_proj.weight.data
        up_weight = layer.mlp.up_proj.weight.data
        down_weight = layer.mlp.down_proj.weight.data

        print(f"Layer {layer_idx}:")
        print(f"  Q: mean={q_weight.mean():.6f}, std={q_weight.std():.6f}, max={q_weight.abs().max():.6f}")
        print(f"  K: mean={k_weight.mean():.6f}, std={k_weight.std():.6f}, max={k_weight.abs().max():.6f}")
        print(f"  V: mean={v_weight.mean():.6f}, std={v_weight.std():.6f}, max={v_weight.abs().max():.6f}")
        print(f"  O: mean={o_weight.mean():.6f}, std={o_weight.std():.6f}, max={o_weight.abs().max():.6f}")
        print(f"  Gate: mean={gate_weight.mean():.6f}, std={gate_weight.std():.6f}, max={gate_weight.abs().max():.6f}")
        print(f"  Up: mean={up_weight.mean():.6f}, std={up_weight.std():.6f}, max={up_weight.abs().max():.6f}")
        print(f"  Down: mean={down_weight.mean():.6f}, std={down_weight.std():.6f}, max={down_weight.abs().max():.6f}")
        print()

def main():
    # 加载原始模型
    print("加载原始模型...")
    original_model = LlamaForCausalLM.from_pretrained(
        "/newdata/LLMs/Llama-3-8B",
        torch_dtype=torch.float16
    )

    # 加载剪枝后模型
    print("加载剪枝后模型...")
    pruned_dict = torch.load('prune_log/llama3_prune/pytorch_model.bin', map_location='cpu', weights_only=False)
    pruned_model = pruned_dict['model']

    # 分析权重
    analyze_weights(original_model, "原始模型")
    analyze_weights(pruned_model, "剪枝后模型")

if __name__ == '__main__':
    main()
