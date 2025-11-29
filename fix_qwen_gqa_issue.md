# Qwen模型GQA剪枝问题分析与解决方案

## 问题现象

剪枝后PPL爆炸：
- wikitext2 PPL: 44494 (正常应该在10以内)
- PTB PPL: 62259 (正常应该在20以内)

## 根本原因

### 1. GQA架构差异

**Llama-3-8B:**
- num_attention_heads: 32
- num_key_value_heads: 8
- **GQA ratio: 4:1**
- head_dim: 128

**Qwen2.5-7B:**
- num_attention_heads: 28
- num_key_value_heads: 4
- **GQA ratio: 7:1** ⚠️
- head_dim: 128

### 2. 剪枝配置问题

当前代码中的`consecutive_groups`设置：
```python
"consecutive_groups": {
    layer.self_attn.k_proj: layer.self_attn.head_dim for layer in model.model.layers
}
```

这告诉剪枝器按照`head_dim=128`的粒度剪枝K projection。

**问题在于：**
- 对于Llama-3 (4:1 ratio)：8个KV heads，每剪枝128维度 = 剪枝1个KV head
- 对于Qwen (7:1 ratio)：**4个KV heads**，剪枝后可能破坏GQA ratio

### 3. 剪枝后的维度计算

剪枝后，代码重新计算heads数量：
```python
layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
layer.self_attn.num_key_value_heads = layer.self_attn.k_proj.weight.data.shape[0] // layer.self_attn.head_dim
```

**问题：**
如果K projection被剪枝了奇数个head_dim维度，那么新的GQA ratio可能变成非整数，导致：
- Attention计算时Q和KV维度不匹配
- 数值不稳定
- PPL爆炸

## 解决方案

### 方案1：禁用Attention层剪枝（最保守）

只剪枝MLP层，不剪枝Attention层：

```bash
CUDA_VISIBLE_DEVICES=2 python llama3.py --pruning_ratio 0.28 \
  --device cuda --eval_device cuda \
  --base_model /newdata/LLMs/Qwen2.5-7B \
  --block_wise \
  --block_mlp_layer_start 4 --block_mlp_layer_end 28 \
  --block_attention_layer_start 28 --block_attention_layer_end 28 \
  --save_ckpt_log_name LLM-Pruner_Qwen_28_mlp_only \
  --pruner_type taylor --taylor param_first \
  --max_seq_len 2048 \
  --save_model
```

注意：`--block_attention_layer_start 28 --block_attention_layer_end 28` 意味着不剪枝任何attention层。

### 方案2：保守的Attention剪枝

剪枝更少的attention层，并减小pruning ratio：

```bash
CUDA_VISIBLE_DEVICES=2 python llama3.py --pruning_ratio 0.20 \
  --device cuda --eval_device cuda \
  --base_model /newdata/LLMs/Qwen2.5-7B \
  --block_wise \
  --block_mlp_layer_start 4 --block_mlp_layer_end 28 \
  --block_attention_layer_start 10 --block_attention_layer_end 24 \
  --save_ckpt_log_name LLM-Pruner_Qwen_20_conservative \
  --pruner_type taylor --taylor param_first \
  --max_seq_len 2048 \
  --save_model
```

### 方案3：修改代码以支持Qwen的GQA ratio（推荐长期方案）

需要修改`llama3.py`中的consecutive_groups设置，使其：
1. 检测GQA ratio
2. 根据ratio调整剪枝粒度
3. 确保剪枝后保持整数ratio

#### 修改步骤：

在`llama3.py`的block_wise部分（第116行附近）添加：

```python
# 检测GQA ratio
config = model.config
gqa_ratio = config.num_attention_heads / config.num_key_value_heads
logger.log(f"Detected GQA ratio: {gqa_ratio}:1")

# 对于高GQA ratio的模型，调整剪枝粒度
if gqa_ratio >= 7:
    logger.log(f"High GQA ratio detected ({gqa_ratio}:1), adjusting pruning strategy")
    # 使用更大的粒度以保持GQA ratio
    consecutive_group_size = int(layer.self_attn.head_dim * gqa_ratio)
    logger.log(f"Using consecutive_group_size: {consecutive_group_size}")
else:
    consecutive_group_size = layer.self_attn.head_dim

"consecutive_groups": {
    layer.self_attn.k_proj: consecutive_group_size for layer in model.model.layers
},
```

## 验证方法

### 1. 使用诊断脚本

```bash
# 检查原始模型
python diagnose_qwen_gqa.py /newdata/LLMs/Qwen2.5-7B

# 检查剪枝后的模型
python diagnose_qwen_gqa.py prune_log/LLM-Pruner_Qwen_28/pytorch_model.bin pruned
```

### 2. 检查PPL

正常的PPL范围：
- wikitext2: 8-15
- PTB: 15-30

如果PPL > 100，说明模型已经被破坏。

### 3. 检查生成质量

```python
python generate.py --model_type pruneLLM --ckpt prune_log/LLM-Pruner_Qwen_28/pytorch_model.bin
```

如果生成的文本是乱码或重复，说明剪枝有问题。

## 为什么Llama-3和Mistral可以工作

1. **更低的GQA ratio (4:1)**
   - 更多的KV heads (8个)
   - 剪枝1-2个KV heads影响较小

2. **更多的层数 (32层)**
   - 可以剪枝更多层
   - 单层影响被分散

3. **Qwen的挑战**
   - 只有4个KV heads
   - 剪枝即使1个KV head都会显著影响GQA ratio (7:1 -> 5.6:1 or 9.3:1)
   - 只有28层，容错空间小

## 建议的实验流程

1. **首先尝试方案1（MLP only）**
   - 验证MLP剪枝对Qwen是否有效
   - 确定基准性能

2. **然后尝试方案2（保守剪枝）**
   - 逐步增加attention层剪枝
   - 监控PPL变化

3. **最后实现方案3（代码修改）**
   - 如果前两个方案效果不好
   - 需要更精细的GQA ratio控制

## 参考资料

- GQA论文: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
- Qwen2.5技术报告: https://qwenlm.github.io/blog/qwen2.5/
- LLM-Pruner论文: "LLM-Pruner: On the Structural Pruning of Large Language Models"
