#!/usr/bin/env python3
"""
分析模型每个层的权重通道维度
支持加载 HuggingFace 模型或剪枝后的模型
"""

import torch
import argparse
from collections import defaultdict


def get_layer_dimensions(module, name=""):
    """获取单个模块的权重维度信息"""
    info = {
        'name': name,
        'type': type(module).__name__,
        'params': {}
    }

    for param_name, param in module.named_parameters(recurse=False):
        shape = list(param.shape)
        info['params'][param_name] = {
            'shape': shape,
            'numel': param.numel(),
            'dtype': str(param.dtype)
        }

        # 分析通道维度
        if len(shape) == 2:  # Linear 层
            info['params'][param_name]['out_features'] = shape[0]
            info['params'][param_name]['in_features'] = shape[1]
        elif len(shape) == 1:  # Bias 或 LayerNorm
            info['params'][param_name]['features'] = shape[0]
        elif len(shape) == 4:  # Conv2d
            info['params'][param_name]['out_channels'] = shape[0]
            info['params'][param_name]['in_channels'] = shape[1]
            info['params'][param_name]['kernel_size'] = (shape[2], shape[3])

    return info


def analyze_model(model, verbose=True):
    """分析整个模型的维度结构"""

    results = {
        'config': {},
        'summary': {
            'total_params': 0,
            'layer_counts': defaultdict(int),
        },
        'layers': []
    }

    # 获取模型配置
    if hasattr(model, 'config'):
        config = model.config
        results['config'] = {
            'hidden_size': getattr(config, 'hidden_size', None),
            'num_attention_heads': getattr(config, 'num_attention_heads', None),
            'num_key_value_heads': getattr(config, 'num_key_value_heads', None),
            'intermediate_size': getattr(config, 'intermediate_size', None),
            'num_hidden_layers': getattr(config, 'num_hidden_layers', None),
            'vocab_size': getattr(config, 'vocab_size', None),
        }

        if verbose:
            print("=" * 80)
            print("模型配置")
            print("=" * 80)
            for key, value in results['config'].items():
                if value is not None:
                    print(f"  {key}: {value}")
            print()

    # 遍历所有模块
    if verbose:
        print("=" * 80)
        print("各层权重维度分析")
        print("=" * 80)

    for name, module in model.named_modules():
        # 只分析有参数的叶子模块
        params = list(module.named_parameters(recurse=False))
        if not params:
            continue

        layer_info = get_layer_dimensions(module, name)
        results['layers'].append(layer_info)

        # 统计
        layer_type = layer_info['type']
        results['summary']['layer_counts'][layer_type] += 1

        for param_info in layer_info['params'].values():
            results['summary']['total_params'] += param_info['numel']

        if verbose:
            print_layer_info(layer_info)

    # 打印总结
    if verbose:
        print("\n" + "=" * 80)
        print("统计总结")
        print("=" * 80)
        print(f"总参数量: {results['summary']['total_params']:,} ({results['summary']['total_params']/1e9:.2f}B)")
        print("\n各类型层数量:")
        for layer_type, count in sorted(results['summary']['layer_counts'].items()):
            print(f"  {layer_type}: {count}")

    return results


def print_layer_info(layer_info):
    """打印单层信息"""
    name = layer_info['name']
    layer_type = layer_info['type']

    # 简化显示
    if not layer_info['params']:
        return

    print(f"\n{name} ({layer_type})")
    print("-" * 60)

    for param_name, param_info in layer_info['params'].items():
        shape_str = str(param_info['shape'])

        # 构建详细描述
        details = []
        if 'out_features' in param_info:
            details.append(f"out={param_info['out_features']}, in={param_info['in_features']}")
        elif 'out_channels' in param_info:
            details.append(f"out_ch={param_info['out_channels']}, in_ch={param_info['in_channels']}")
        elif 'features' in param_info:
            details.append(f"features={param_info['features']}")

        detail_str = f" ({', '.join(details)})" if details else ""
        print(f"  {param_name}: {shape_str}{detail_str}")


def analyze_transformer_layers(model, verbose=True):
    """专门分析 Transformer 层的结构（针对 LLaMA 等模型）"""

    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        print("警告: 模型不是标准的 LLaMA 结构")
        return None

    if verbose:
        print("\n" + "=" * 80)
        print("Transformer 层详细分析")
        print("=" * 80)

    layer_dims = []

    for idx, layer in enumerate(model.model.layers):
        layer_info = {
            'layer_idx': idx,
            'attention': {},
            'mlp': {}
        }

        # Attention 层
        attn = layer.self_attn
        layer_info['attention'] = {
            'q_proj': list(attn.q_proj.weight.shape),
            'k_proj': list(attn.k_proj.weight.shape),
            'v_proj': list(attn.v_proj.weight.shape),
            'o_proj': list(attn.o_proj.weight.shape),
            'num_heads': getattr(attn, 'num_heads', None),
            'num_kv_heads': getattr(attn, 'num_key_value_heads', None),
            'head_dim': getattr(attn, 'head_dim', None),
        }

        # MLP 层
        mlp = layer.mlp
        layer_info['mlp'] = {
            'gate_proj': list(mlp.gate_proj.weight.shape),
            'up_proj': list(mlp.up_proj.weight.shape),
            'down_proj': list(mlp.down_proj.weight.shape),
        }

        layer_dims.append(layer_info)

        if verbose:
            print(f"\nLayer {idx}")
            print("-" * 40)

            # Attention
            q_shape = layer_info['attention']['q_proj']
            k_shape = layer_info['attention']['k_proj']
            v_shape = layer_info['attention']['v_proj']
            o_shape = layer_info['attention']['o_proj']

            print(f"  Attention:")
            print(f"    Q: {q_shape} (out={q_shape[0]}, in={q_shape[1]})")
            print(f"    K: {k_shape} (out={k_shape[0]}, in={k_shape[1]})")
            print(f"    V: {v_shape} (out={v_shape[0]}, in={v_shape[1]})")
            print(f"    O: {o_shape} (out={o_shape[0]}, in={o_shape[1]})")

            if layer_info['attention']['num_heads']:
                print(f"    num_heads={layer_info['attention']['num_heads']}, "
                      f"num_kv_heads={layer_info['attention']['num_kv_heads']}, "
                      f"head_dim={layer_info['attention']['head_dim']}")

            # MLP
            gate_shape = layer_info['mlp']['gate_proj']
            up_shape = layer_info['mlp']['up_proj']
            down_shape = layer_info['mlp']['down_proj']

            print(f"  MLP:")
            print(f"    gate: {gate_shape} (out={gate_shape[0]}, in={gate_shape[1]})")
            print(f"    up:   {up_shape} (out={up_shape[0]}, in={up_shape[1]})")
            print(f"    down: {down_shape} (out={down_shape[0]}, in={down_shape[1]})")

    return layer_dims


def load_model(model_path, model_type='auto'):
    """加载模型"""

    if model_type == 'pruned' or model_path.endswith('.bin'):
        # 加载剪枝后的模型
        print(f"加载剪枝模型: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model']
        else:
            model = checkpoint
        return model

    else:
        # 加载 HuggingFace 模型
        print(f"加载 HuggingFace 模型: {model_path}")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        return model


def main():
    parser = argparse.ArgumentParser(description='分析模型权重通道维度')
    parser.add_argument('--model', type=str, required=True,
                        help='模型路径 (HuggingFace 模型名或剪枝后的 .bin 文件)')
    parser.add_argument('--model_type', type=str, default='auto',
                        choices=['auto', 'hf', 'pruned'],
                        help='模型类型: auto(自动检测), hf(HuggingFace), pruned(剪枝模型)')
    parser.add_argument('--detailed', action='store_true',
                        help='显示所有层的详细信息')
    parser.add_argument('--transformer_only', action='store_true',
                        help='只分析 Transformer 层')
    parser.add_argument('--layer', type=int, default=None,
                        help='只分析指定的层')
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果到 JSON 文件')

    args = parser.parse_args()

    # 加载模型
    model = load_model(args.model, args.model_type)

    # 分析模型
    if args.transformer_only:
        analyze_transformer_layers(model, verbose=True)
    elif args.layer is not None:
        # 只分析特定层
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layer = model.model.layers[args.layer]
            print(f"\n分析 Layer {args.layer}")
            analyze_transformer_layers_single(model, args.layer)
        else:
            print("错误: 无法找到指定的层")
    else:
        results = analyze_model(model, verbose=args.detailed)

        if not args.detailed:
            # 默认只显示 Transformer 层分析
            analyze_transformer_layers(model, verbose=True)

        # 保存结果
        if args.output:
            import json
            with open(args.output, 'w') as f:
                # 转换为可序列化格式
                json.dump(results, f, indent=2, default=str)
            print(f"\n结果已保存到: {args.output}")


def analyze_transformer_layers_single(model, layer_idx):
    """分析单个 Transformer 层"""
    layer = model.model.layers[layer_idx]

    print("=" * 60)

    # Attention
    attn = layer.self_attn
    print("\nAttention 层:")
    print(f"  Q projection: {list(attn.q_proj.weight.shape)}")
    print(f"  K projection: {list(attn.k_proj.weight.shape)}")
    print(f"  V projection: {list(attn.v_proj.weight.shape)}")
    print(f"  O projection: {list(attn.o_proj.weight.shape)}")

    if hasattr(attn, 'num_heads'):
        print(f"\n  配置:")
        print(f"    num_heads: {attn.num_heads}")
        print(f"    num_key_value_heads: {getattr(attn, 'num_key_value_heads', 'N/A')}")
        print(f"    head_dim: {attn.head_dim}")

    # MLP
    mlp = layer.mlp
    print("\nMLP 层:")
    print(f"  gate_proj: {list(mlp.gate_proj.weight.shape)}")
    print(f"  up_proj:   {list(mlp.up_proj.weight.shape)}")
    print(f"  down_proj: {list(mlp.down_proj.weight.shape)}")

    # LayerNorm
    print("\nLayerNorm:")
    print(f"  input_layernorm:  {list(layer.input_layernorm.weight.shape)}")
    print(f"  post_attention_layernorm: {list(layer.post_attention_layernorm.weight.shape)}")


if __name__ == '__main__':
    main()
