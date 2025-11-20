#!/usr/bin/env python3
"""
完整分析模型所有模块的权重维度
包括 Embedding, Transformer Layers, LayerNorm, LM Head 等所有组件
"""

import torch
import argparse
from collections import OrderedDict


def load_model(model_path, model_type='auto'):
    """加载模型"""
    if model_type == 'pruned' or model_path.endswith('.bin'):
        print(f"加载剪枝模型: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model']
        else:
            model = checkpoint
        return model
    else:
        print(f"加载 HuggingFace 模型: {model_path}")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        return model


def analyze_full_model(model, show_all_layers=False):
    """完整分析模型所有组件"""

    print("=" * 100)
    print("模型完整结构分析")
    print("=" * 100)

    # 1. 模型配置
    print("\n" + "=" * 100)
    print("一、模型配置 (Model Configuration)")
    print("=" * 100)

    if hasattr(model, 'config'):
        config = model.config
        config_items = [
            ('model_type', getattr(config, 'model_type', None)),
            ('hidden_size', getattr(config, 'hidden_size', None)),
            ('intermediate_size', getattr(config, 'intermediate_size', None)),
            ('num_hidden_layers', getattr(config, 'num_hidden_layers', None)),
            ('num_attention_heads', getattr(config, 'num_attention_heads', None)),
            ('num_key_value_heads', getattr(config, 'num_key_value_heads', None)),
            ('head_dim', getattr(config, 'head_dim', None)),
            ('vocab_size', getattr(config, 'vocab_size', None)),
            ('max_position_embeddings', getattr(config, 'max_position_embeddings', None)),
            ('rms_norm_eps', getattr(config, 'rms_norm_eps', None)),
            ('rope_theta', getattr(config, 'rope_theta', None)),
        ]

        for name, value in config_items:
            if value is not None:
                print(f"  {name}: {value}")

        # 计算 head_dim
        if hasattr(config, 'hidden_size') and hasattr(config, 'num_attention_heads'):
            computed_head_dim = config.hidden_size // config.num_attention_heads
            print(f"  computed_head_dim: {computed_head_dim}")

        # GQA 比例
        if hasattr(config, 'num_attention_heads') and hasattr(config, 'num_key_value_heads'):
            if config.num_key_value_heads and config.num_key_value_heads > 0:
                gqa_ratio = config.num_attention_heads // config.num_key_value_heads
                print(f"  GQA_ratio: {gqa_ratio}:1 (num_heads/num_kv_heads)")

    # 2. 输入嵌入层
    print("\n" + "=" * 100)
    print("二、输入嵌入层 (Input Embedding)")
    print("=" * 100)

    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed = model.model.embed_tokens
        weight_shape = list(embed.weight.shape)
        print(f"  model.model.embed_tokens")
        print(f"    weight: {weight_shape}")
        print(f"    vocab_size: {weight_shape[0]}")
        print(f"    embedding_dim: {weight_shape[1]}")
        print(f"    参数量: {embed.weight.numel():,}")

    # 3. Transformer 层
    print("\n" + "=" * 100)
    print("三、Transformer 层 (Transformer Layers)")
    print("=" * 100)

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
        print(f"  总层数: {num_layers}")

        # 决定显示哪些层
        if show_all_layers:
            layers_to_show = list(range(num_layers))
        else:
            # 只显示关键层：第一层、中间层、最后几层
            layers_to_show = [0, 1, 2]
            if num_layers > 6:
                layers_to_show.append(num_layers // 2)
            layers_to_show.extend([num_layers - 3, num_layers - 2, num_layers - 1])
            layers_to_show = sorted(set(layers_to_show))

        for idx in layers_to_show:
            if idx >= num_layers:
                continue

            layer = model.model.layers[idx]
            print(f"\n  Layer {idx}")
            print("  " + "-" * 80)

            # 3.1 Input LayerNorm
            if hasattr(layer, 'input_layernorm'):
                norm = layer.input_layernorm
                print(f"    input_layernorm:")
                print(f"      weight: {list(norm.weight.shape)} (features={norm.weight.shape[0]})")

            # 3.2 Self Attention
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                print(f"    self_attn:")

                # Q projection
                q_shape = list(attn.q_proj.weight.shape)
                print(f"      q_proj.weight: {q_shape} (out={q_shape[0]}, in={q_shape[1]})")

                # K projection
                k_shape = list(attn.k_proj.weight.shape)
                print(f"      k_proj.weight: {k_shape} (out={k_shape[0]}, in={k_shape[1]})")

                # V projection
                v_shape = list(attn.v_proj.weight.shape)
                print(f"      v_proj.weight: {v_shape} (out={v_shape[0]}, in={v_shape[1]})")

                # O projection
                o_shape = list(attn.o_proj.weight.shape)
                print(f"      o_proj.weight: {o_shape} (out={o_shape[0]}, in={o_shape[1]})")

                # Attention 配置
                if hasattr(attn, 'num_heads'):
                    print(f"      num_heads: {attn.num_heads}")
                if hasattr(attn, 'num_key_value_heads'):
                    print(f"      num_key_value_heads: {attn.num_key_value_heads}")
                if hasattr(attn, 'head_dim'):
                    print(f"      head_dim: {attn.head_dim}")

                # 验证维度一致性
                head_dim = getattr(attn, 'head_dim', 128)
                expected_q_heads = q_shape[0] // head_dim
                expected_kv_heads = k_shape[0] // head_dim

                config_q_heads = getattr(attn, 'num_heads', expected_q_heads)
                config_kv_heads = getattr(attn, 'num_key_value_heads', expected_kv_heads)

                q_match = (expected_q_heads == config_q_heads)
                kv_match = (expected_kv_heads == config_kv_heads)

                if not q_match or not kv_match:
                    print(f"      ⚠️  维度不匹配!")
                    print(f"         Q: 期望{config_q_heads}*{head_dim}={config_q_heads*head_dim}, 实际{q_shape[0]}")
                    print(f"         K/V: 期望{config_kv_heads}*{head_dim}={config_kv_heads*head_dim}, 实际{k_shape[0]}")

            # 3.3 Post Attention LayerNorm
            if hasattr(layer, 'post_attention_layernorm'):
                norm = layer.post_attention_layernorm
                print(f"    post_attention_layernorm:")
                print(f"      weight: {list(norm.weight.shape)} (features={norm.weight.shape[0]})")

            # 3.4 MLP
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                print(f"    mlp:")

                # Gate projection
                gate_shape = list(mlp.gate_proj.weight.shape)
                print(f"      gate_proj.weight: {gate_shape} (out={gate_shape[0]}, in={gate_shape[1]})")

                # Up projection
                up_shape = list(mlp.up_proj.weight.shape)
                print(f"      up_proj.weight: {up_shape} (out={up_shape[0]}, in={up_shape[1]})")

                # Down projection
                down_shape = list(mlp.down_proj.weight.shape)
                print(f"      down_proj.weight: {down_shape} (out={down_shape[0]}, in={down_shape[1]})")

                # 验证 MLP 维度一致性
                if gate_shape[0] != up_shape[0] or gate_shape[0] != down_shape[1]:
                    print(f"      ⚠️  MLP 维度不匹配!")
                    print(f"         gate_out={gate_shape[0]}, up_out={up_shape[0]}, down_in={down_shape[1]}")

        if not show_all_layers:
            print(f"\n  ... (使用 --all 参数显示所有 {num_layers} 层)")

    # 4. 最终 LayerNorm
    print("\n" + "=" * 100)
    print("四、最终归一化层 (Final LayerNorm)")
    print("=" * 100)

    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        norm = model.model.norm
        print(f"  model.model.norm")
        print(f"    weight: {list(norm.weight.shape)} (features={norm.weight.shape[0]})")
        print(f"    参数量: {norm.weight.numel():,}")

    # 5. 输出层 (LM Head)
    print("\n" + "=" * 100)
    print("五、输出层 (LM Head)")
    print("=" * 100)

    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
        weight_shape = list(lm_head.weight.shape)
        print(f"  model.lm_head")
        print(f"    weight: {weight_shape} (out={weight_shape[0]}, in={weight_shape[1]})")
        print(f"    vocab_size: {weight_shape[0]}")
        print(f"    hidden_size: {weight_shape[1]}")
        print(f"    参数量: {lm_head.weight.numel():,}")

        if hasattr(lm_head, 'bias') and lm_head.bias is not None:
            print(f"    bias: {list(lm_head.bias.shape)}")

    # 6. 参数统计
    print("\n" + "=" * 100)
    print("六、参数统计 (Parameter Statistics)")
    print("=" * 100)

    total_params = 0
    trainable_params = 0

    param_by_component = OrderedDict()

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

        # 按组件分类
        if 'embed_tokens' in name:
            component = 'Embedding'
        elif 'lm_head' in name:
            component = 'LM Head'
        elif 'model.norm' in name and 'layers' not in name:
            component = 'Final Norm'
        elif 'input_layernorm' in name:
            component = 'Input LayerNorm'
        elif 'post_attention_layernorm' in name:
            component = 'Post Attn LayerNorm'
        elif 'q_proj' in name:
            component = 'Q Projection'
        elif 'k_proj' in name:
            component = 'K Projection'
        elif 'v_proj' in name:
            component = 'V Projection'
        elif 'o_proj' in name:
            component = 'O Projection'
        elif 'gate_proj' in name:
            component = 'Gate Projection'
        elif 'up_proj' in name:
            component = 'Up Projection'
        elif 'down_proj' in name:
            component = 'Down Projection'
        else:
            component = 'Other'

        if component not in param_by_component:
            param_by_component[component] = 0
        param_by_component[component] += param.numel()

    print(f"  总参数量: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  可训练参数: {trainable_params:,} ({trainable_params/1e9:.2f}B)")

    print(f"\n  各组件参数量:")
    for component, count in param_by_component.items():
        percentage = 100 * count / total_params
        print(f"    {component}: {count:,} ({percentage:.1f}%)")

    # 7. 维度一致性检查
    print("\n" + "=" * 100)
    print("七、维度一致性检查 (Dimension Consistency Check)")
    print("=" * 100)

    issues = []

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            mlp = layer.mlp

            # 检查 Attention
            q_dim = attn.q_proj.weight.shape[0]
            k_dim = attn.k_proj.weight.shape[0]
            v_dim = attn.v_proj.weight.shape[0]
            o_in_dim = attn.o_proj.weight.shape[1]

            head_dim = getattr(attn, 'head_dim', 128)

            if q_dim != o_in_dim:
                issues.append(f"Layer {idx}: Q维度({q_dim}) != O输入维度({o_in_dim})")

            if k_dim != v_dim:
                issues.append(f"Layer {idx}: K维度({k_dim}) != V维度({v_dim})")

            if q_dim % head_dim != 0:
                issues.append(f"Layer {idx}: Q维度({q_dim})不是head_dim({head_dim})的整数倍")

            if k_dim % head_dim != 0:
                issues.append(f"Layer {idx}: K维度({k_dim})不是head_dim({head_dim})的整数倍")

            # 检查 GQA 比例
            num_q_heads = q_dim // head_dim
            num_kv_heads = k_dim // head_dim
            if num_kv_heads > 0 and num_q_heads % num_kv_heads != 0:
                issues.append(f"Layer {idx}: Q heads({num_q_heads})不是KV heads({num_kv_heads})的整数倍")

            # 检查 MLP
            gate_dim = mlp.gate_proj.weight.shape[0]
            up_dim = mlp.up_proj.weight.shape[0]
            down_in_dim = mlp.down_proj.weight.shape[1]

            if gate_dim != up_dim:
                issues.append(f"Layer {idx}: gate维度({gate_dim}) != up维度({up_dim})")

            if gate_dim != down_in_dim:
                issues.append(f"Layer {idx}: gate维度({gate_dim}) != down输入维度({down_in_dim})")

    if issues:
        print(f"  ❌ 发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✅ 所有维度检查通过")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description='完整分析模型所有模块的权重维度')
    parser.add_argument('--model', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--model_type', type=str, default='auto',
                        choices=['auto', 'hf', 'pruned'])
    parser.add_argument('--all', action='store_true',
                        help='显示所有 Transformer 层')
    parser.add_argument('--output', type=str, default=None,
                        help='输出到文件')

    args = parser.parse_args()

    # 重定向输出
    if args.output:
        import sys
        original_stdout = sys.stdout
        with open(args.output, 'w') as f:
            sys.stdout = f
            model = load_model(args.model, args.model_type)
            analyze_full_model(model, show_all_layers=args.all)
        sys.stdout = original_stdout
        print(f"结果已保存到: {args.output}")
    else:
        model = load_model(args.model, args.model_type)
        analyze_full_model(model, show_all_layers=args.all)


if __name__ == '__main__':
    main()
