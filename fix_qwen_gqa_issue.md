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

### 3. LLM-Pruner的GQA比例保持机制

**LLM-Pruner通过importance对齐自动保持GQA ratio**（`hf_llama_pruner.py:332-339`）：

```python
min_imp_size = min([len(imp) for imp in group_imp])  # Qwen: 512 (K/V维度)
for imp in group_imp:
    if len(imp) > min_imp_size and len(imp) % min_imp_size == 0:  # Q的3584 > 512
        imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)  # (7, 512) -> 512
```

**工作原理：**
- Q importance (3584维) reshape成 (7, 512) 然后sum → 512维
- K/V importance 保持512维
- 剪枝1个KV head (128维) → Q自动剪枝7个heads (896维)
- **GQA ratio自动保持7:1不变！**

**那为什么还会PPL爆炸？**
问题不在于ratio被破坏，而在于：
1. **KV heads太少**：Qwen只有4个KV heads，剪枝1个就是25%损失！
2. **GQA ratio高**：7:1意味着每个KV head被7个Q heads共享，影响更大
3. **剪枝过度**：使用了过高的pruning_ratio或剪枝了太多层

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

### 方案3：理解GQA比例保持机制（重要！）

**好消息**：LLM-Pruner已经内置了GQA ratio保持机制！

**关键代码**（`hf_llama_pruner.py:332-339`）会自动：
1. 检测Q、K、V的维度差异
2. 将Q的importance reshape并对齐到K/V的维度
3. 确保剪枝时保持正确的ratio

**因此不需要修改consecutive_groups！**
- 保持 `consecutive_group_size = head_dim = 128`
- LLM-Pruner会自动处理7:1的ratio

**真正的问题**：
- Qwen只有4个KV heads，容错空间极小
- 需要更保守的剪枝策略（方案1或2）
- 或者只剪枝MLP层

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
