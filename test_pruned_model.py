#!/usr/bin/env python3
"""
测试剪枝后模型的基本功能
"""

import torch

def test_pruned_model(ckpt_path):
    """测试剪枝后的模型"""
    print(f"加载模型: {ckpt_path}")
    pruned_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    tokenizer = pruned_dict['tokenizer']
    model = pruned_dict['model']

    model.eval()
    model.to('cuda')

    # 简单的生成测试
    test_prompts = [
        "The capital of France is",
        "1 + 1 =",
        "The meaning of life is"
    ]

    print("\n" + "="*80)
    print("生成测试:")
    print("="*80)

    for prompt in test_prompts:
        print(f"\n输入: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                do_sample=False,  # 贪心解码,更稳定
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"输出: {generated_text}")

    # 测试logits分布
    print("\n" + "="*80)
    print("Logits分布测试:")
    print("="*80)

    test_input = "Hello"
    inputs = tokenizer(test_input, return_tensors="pt").to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        print(f"输入shape: {inputs.input_ids.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Logits统计:")
        print(f"  Mean: {logits.mean().item():.4f}")
        print(f"  Std: {logits.std().item():.4f}")
        print(f"  Min: {logits.min().item():.4f}")
        print(f"  Max: {logits.max().item():.4f}")
        print(f"  是否包含NaN: {torch.isnan(logits).any().item()}")
        print(f"  是否包含Inf: {torch.isinf(logits).any().item()}")

        # Top-5 tokens
        probs = torch.softmax(logits[0, -1], dim=-1)
        top5_probs, top5_indices = torch.topk(probs, 5)

        print(f"\nTop-5预测tokens:")
        for i in range(5):
            token_id = top5_indices[i].item()
            token = tokenizer.decode([token_id])
            prob = top5_probs[i].item()
            print(f"  {i+1}. '{token}' (id={token_id}, prob={prob:.4f})")

if __name__ == '__main__':
    import sys
    ckpt_path = 'prune_log/llama3_prune/pytorch_model.bin'
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]

    test_pruned_model(ckpt_path)
