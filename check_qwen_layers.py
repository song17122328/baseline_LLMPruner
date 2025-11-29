#!/usr/bin/env python3
"""
检查Qwen模型的层数和结构
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

def check_model_layers(model_path):
    """检查模型的层数"""
    print(f"Loading model from: {model_path}")

    try:
        # 加载模型配置
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)

        print(f"\n=== Model Configuration ===")
        print(f"Model type: {config.model_type}")
        print(f"Number of hidden layers: {config.num_hidden_layers}")
        print(f"Hidden size: {config.hidden_size}")
        print(f"Number of attention heads: {config.num_attention_heads}")

        if hasattr(config, 'num_key_value_heads'):
            print(f"Number of key-value heads (GQA): {config.num_key_value_heads}")
            print(f"GQA ratio: {config.num_attention_heads / config.num_key_value_heads}")

        # 尝试加载模型（轻量级）
        print(f"\n=== Loading Model ===")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )

        # 检查实际层数
        num_layers = len(model.model.layers)
        print(f"Actual number of layers in model.model.layers: {num_layers}")

        # 检查第一层的结构
        print(f"\n=== Layer 0 Structure ===")
        layer_0 = model.model.layers[0]
        print(f"Layer 0 type: {type(layer_0)}")
        print(f"Layer 0 attributes:")
        for attr in dir(layer_0):
            if not attr.startswith('_'):
                print(f"  - {attr}")

        # 检查attention结构
        if hasattr(layer_0, 'self_attn'):
            print(f"\n=== Self Attention Structure ===")
            attn = layer_0.self_attn
            print(f"Self attention type: {type(attn)}")
            print(f"Self attention attributes:")
            for attr in dir(attn):
                if not attr.startswith('_') and hasattr(attn, attr):
                    attr_value = getattr(attn, attr)
                    if isinstance(attr_value, torch.nn.Module):
                        print(f"  - {attr}: {type(attr_value)}")

        # 检查MLP结构
        if hasattr(layer_0, 'mlp'):
            print(f"\n=== MLP Structure ===")
            mlp = layer_0.mlp
            print(f"MLP type: {type(mlp)}")
            print(f"MLP attributes:")
            for attr in dir(mlp):
                if not attr.startswith('_') and hasattr(mlp, attr):
                    attr_value = getattr(mlp, attr)
                    if isinstance(attr_value, torch.nn.Module):
                        print(f"  - {attr}: {type(attr_value)}")

        print(f"\n=== Recommended Settings ===")
        print(f"For block-wise pruning, use:")
        print(f"  --block_attention_layer_start 4")
        print(f"  --block_attention_layer_end {num_layers}")
        print(f"  --block_mlp_layer_start 4")
        print(f"  --block_mlp_layer_end {num_layers}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # 默认使用用户提供的路径
        model_path = "/newdata/LLMs/Qwen2.5-7B"

    check_model_layers(model_path)
