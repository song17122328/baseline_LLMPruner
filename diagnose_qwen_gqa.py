#!/usr/bin/env python3
"""
诊断Qwen模型的GQA配置和剪枝后的状态
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def diagnose_model_gqa(model_path, is_pruned=False):
    """诊断模型的GQA配置"""

    print(f"\n{'='*80}")
    print(f"诊断模型: {model_path}")
    print(f"是否是剪枝后的模型: {is_pruned}")
    print(f"{'='*80}\n")

    # 加载配置
    config = AutoConfig.from_pretrained(model_path) if not is_pruned else None

    if config:
        print("=== 配置信息 ===")
        print(f"模型类型: {config.model_type}")
        print(f"隐藏层数: {config.num_hidden_layers}")
        print(f"隐藏维度: {config.hidden_size}")
        print(f"Attention heads: {config.num_attention_heads}")
        print(f"Key-value heads: {config.num_key_value_heads}")
        print(f"Head dim: {config.hidden_size // config.num_attention_heads}")
        print(f"GQA ratio: {config.num_attention_heads / config.num_key_value_heads:.1f}:1")
        print()

    # 加载模型
    if is_pruned:
        print("加载剪枝后的模型...")
        model = torch.load(model_path, map_location='cpu')['model']
    else:
        print("加载原始模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )

    # 检查每一层的维度
    print("\n=== 各层详细信息 ===")
    issues = []

    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn

        # 获取实际维度
        q_dim = attn.q_proj.weight.data.shape[0]
        k_dim = attn.k_proj.weight.data.shape[0]
        v_dim = attn.v_proj.weight.data.shape[0]
        o_dim = attn.o_proj.weight.data.shape[1]

        # 获取配置
        num_heads = attn.num_heads if hasattr(attn, 'num_heads') else None
        num_kv_heads = attn.num_key_value_heads if hasattr(attn, 'num_key_value_heads') else None
        head_dim = attn.head_dim if hasattr(attn, 'head_dim') else None

        # 计算实际的heads数量（如果head_dim已知）
        if head_dim:
            actual_q_heads = q_dim // head_dim
            actual_kv_heads = k_dim // head_dim
            actual_gqa_ratio = actual_q_heads / actual_kv_heads if actual_kv_heads > 0 else 0
        else:
            actual_q_heads = num_heads
            actual_kv_heads = num_kv_heads
            actual_gqa_ratio = 0

        # 检查是否有问题
        has_issue = False
        issue_details = []

        # 检查Q、K、V维度是否一致（对于标准attention）
        if k_dim != v_dim:
            has_issue = True
            issue_details.append(f"K维度({k_dim}) != V维度({v_dim})")

        # 检查GQA ratio是否合理
        if head_dim and actual_kv_heads > 0:
            if actual_gqa_ratio not in [1, 2, 4, 7, 8]:  # 常见的GQA ratio
                has_issue = True
                issue_details.append(f"异常的GQA ratio: {actual_gqa_ratio:.2f}:1")

            # 检查维度是否能被head_dim整除
            if q_dim % head_dim != 0:
                has_issue = True
                issue_details.append(f"Q维度({q_dim})不能被head_dim({head_dim})整除")
            if k_dim % head_dim != 0:
                has_issue = True
                issue_details.append(f"K维度({k_dim})不能被head_dim({head_dim})整除")

        if has_issue or i < 3 or i >= len(model.model.layers) - 3:  # 总是显示前3层和后3层
            status = "❌" if has_issue else "✓"
            print(f"\n{status} Layer {i}:")
            print(f"  配置: num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
            print(f"  Q维度: {q_dim} ({actual_q_heads} heads)")
            print(f"  K维度: {k_dim} ({actual_kv_heads} heads)")
            print(f"  V维度: {v_dim}")
            print(f"  O维度: {o_dim}")
            print(f"  实际GQA ratio: {actual_gqa_ratio:.2f}:1")

            if issue_details:
                print(f"  问题:")
                for detail in issue_details:
                    print(f"    - {detail}")

                issues.append({
                    'layer': i,
                    'details': issue_details,
                    'q_dim': q_dim,
                    'k_dim': k_dim,
                    'v_dim': v_dim,
                    'gqa_ratio': actual_gqa_ratio
                })

    # 总结
    print(f"\n{'='*80}")
    if issues:
        print(f"❌ 发现 {len(issues)} 个问题层")
        print("\n建议:")
        print("1. 检查剪枝配置是否正确保持了GQA ratio")
        print("2. 确保consecutive_groups的设置与模型的GQA架构匹配")
        print("3. 对于Qwen2.5-7B (GQA ratio 7:1)，剪枝K/V heads时要特别小心")
    else:
        print("✓ 所有层的GQA配置看起来正常")
    print(f"{'='*80}\n")

    return issues

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        is_pruned = len(sys.argv) > 2 and sys.argv[2].lower() in ['true', '1', 'yes', 'pruned']
    else:
        # 检查原始Qwen模型
        model_path = "/newdata/LLMs/Qwen2.5-7B"
        is_pruned = False

        if os.path.exists(model_path):
            print("检查原始Qwen2.5-7B模型...")
            diagnose_model_gqa(model_path, is_pruned=False)

        # 检查剪枝后的模型（如果存在）
        pruned_path = "prune_log/LLM-Pruner_Qwen_28/pytorch_model.bin"
        if os.path.exists(pruned_path):
            print("\n\n")
            print("检查剪枝后的模型...")
            diagnose_model_gqa(pruned_path, is_pruned=True)
        else:
            print(f"\n剪枝后的模型不存在: {pruned_path}")
            print("提示: 使用 'python diagnose_qwen_gqa.py <pruned_model_path> pruned' 来检查剪枝后的模型")
