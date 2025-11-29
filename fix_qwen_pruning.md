# Qwen2.5-7B 剪枝问题解决方案

## 问题分析

您遇到的错误：
```
IndexError: index 28 is out of range
```

**根本原因**：Qwen2.5-7B 模型只有 **28 层**（num_hidden_layers=28），而您在命令中设置了：
- `--block_attention_layer_end 30`
- `--block_mlp_layer_end 30`

这导致代码尝试访问不存在的第28、29层（range(4, 30)会访问索引4到29）。

## 解决方案

### 方案1：调整层数参数（推荐）

将结束层数改为28或更小：

```bash
CUDA_VISIBLE_DEVICES=2 python llama3.py --pruning_ratio 0.28 \
  --device cuda --eval_device cuda \
  --base_model /newdata/LLMs/Qwen2.5-7B \
  --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 28 \
  --block_attention_layer_start 4 --block_attention_layer_end 28 \
  --save_ckpt_log_name LLM-Pruner_Qwen_20 \
  --pruner_type taylor --taylor param_first \
  --max_seq_len 2048 \
  --save_model
```

### 方案2：自动检测层数（更好）

修改 `llama3.py`，添加自动层数检测：

在 `llama3.py` 的第110行之前添加：

```python
# 自动检测模型层数
num_layers = len(model.model.layers)
logger.log(f"Detected {num_layers} layers in the model")

# 自动调整layer end参数
if args.block_attention_layer_end > num_layers:
    logger.log(f"Warning: block_attention_layer_end ({args.block_attention_layer_end}) > num_layers ({num_layers})")
    logger.log(f"Adjusting block_attention_layer_end to {num_layers}")
    args.block_attention_layer_end = num_layers

if args.block_mlp_layer_end > num_layers:
    logger.log(f"Warning: block_mlp_layer_end ({args.block_mlp_layer_end}) > num_layers ({num_layers})")
    logger.log(f"Adjusting block_mlp_layer_end to {num_layers}")
    args.block_mlp_layer_end = num_layers
```

## Qwen 与 Llama 模型的区别

### 1. 层数差异
- **Llama-3-8B**: 32 层
- **Mistral-7B-v0.3**: 32 层
- **Qwen2.5-7B**: **28 层** ⚠️

### 2. GQA配置
- **Llama-3-8B**:
  - num_attention_heads: 32
  - num_key_value_heads: 8
  - GQA ratio: 4

- **Qwen2.5-7B**:
  - num_attention_heads: 28
  - num_key_value_heads: 4
  - GQA ratio: 7 ⚠️

Qwen的GQA比例(7:1)与Llama-3(4:1)不同，这可能影响剪枝策略。

### 3. 模型类型
- Llama/Mistral: `model_type="llama"`
- Qwen: `model_type="qwen2"`

虽然Qwen可以使用LlamaForCausalLM加载（架构相似），但有细微差异。

## 建议的Qwen剪枝配置

### 保守配置（推荐用于首次尝试）
```bash
CUDA_VISIBLE_DEVICES=2 python llama3.py --pruning_ratio 0.28 \
  --device cuda --eval_device cuda \
  --base_model /newdata/LLMs/Qwen2.5-7B \
  --block_wise \
  --block_mlp_layer_start 4 --block_mlp_layer_end 24 \
  --block_attention_layer_start 4 --block_attention_layer_end 24 \
  --save_ckpt_log_name LLM-Pruner_Qwen_28 \
  --pruner_type taylor --taylor param_first \
  --max_seq_len 2048 \
  --save_model
```

### 激进配置（剪枝更多层）
```bash
CUDA_VISIBLE_DEVICES=2 python llama3.py --pruning_ratio 0.28 \
  --device cuda --eval_device cuda \
  --base_model /newdata/LLMs/Qwen2.5-7B \
  --block_wise \
  --block_mlp_layer_start 2 --block_mlp_layer_end 26 \
  --block_attention_layer_start 2 --block_attention_layer_end 26 \
  --save_ckpt_log_name LLM-Pruner_Qwen_28_aggressive \
  --pruner_type taylor --taylor param_first \
  --max_seq_len 2048 \
  --save_model
```

## 验证模型信息的命令

使用我创建的检查脚本：
```bash
python check_qwen_layers.py /newdata/LLMs/Qwen2.5-7B
```

或者使用Python快速检查：
```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("/newdata/LLMs/Qwen2.5-7B")
print(f"num_hidden_layers: {config.num_hidden_layers}")
print(f"num_attention_heads: {config.num_attention_heads}")
print(f"num_key_value_heads: {config.num_key_value_heads}")
```

## 常见问题

### Q: 为什么Qwen只有28层，而Llama-3-8B有32层？
A: 这是模型架构设计选择。Qwen2.5在其他方面可能有补偿（如更大的hidden_size或不同的attention机制）。

### Q: Qwen的GQA比例不同会影响剪枝吗？
A: 可能会。GQA ratio从4变到7意味着key-value heads更少，剪枝时需要更谨慎。建议：
- 使用较小的pruning_ratio（0.25-0.28而不是0.3+）
- 监控剪枝后的性能变化

### Q: 可以直接用Llama-3的剪枝配置吗？
A: 不完全可以。需要调整：
1. layer_end参数：28而不是30+
2. 可能需要调整pruning_ratio
3. 注意GQA差异带来的影响

## 下一步建议

1. 先使用保守配置进行剪枝
2. 评估剪枝后模型的性能
3. 如果效果好，可以逐步尝试更激进的配置
4. 记录不同配置下的性能指标，找到最佳平衡点
