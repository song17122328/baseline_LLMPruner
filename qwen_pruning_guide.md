# Qwen2.5-7B 剪枝使用指南

## 问题总结

Qwen2.5-7B的GQA ratio是7:1（28 attention heads, 4 key-value heads），这比Llama-3的4:1更极端。
直接使用Llama-3的剪枝配置会导致PPL爆炸。

## 已修复的问题

1. **自动层数检测**：代码现在会自动检测模型层数并调整参数
2. **GQA ratio感知**：代码现在会检测GQA ratio并调整consecutive_group_size

## 推荐的剪枝策略

### 策略1：仅剪枝MLP层（最稳定）⭐⭐⭐⭐⭐

```bash
CUDA_VISIBLE_DEVICES=2 python llama3.py --pruning_ratio 0.25 \
  --device cuda --eval_device cuda \
  --base_model /newdata/LLMs/Qwen2.5-7B \
  --block_wise \
  --block_mlp_layer_start 4 --block_mlp_layer_end 28 \
  --block_attention_layer_start 28 --block_attention_layer_end 28 \
  --save_ckpt_log_name Qwen_mlp_only_25 \
  --pruner_type taylor --taylor param_first \
  --max_seq_len 2048 \
  --save_model
```

**说明：**
- `--block_attention_layer_start 28 --block_attention_layer_end 28`：不剪枝attention层
- 只剪枝MLP层，避免破坏GQA结构
- 预期可以获得15-20%的参数减少，同时保持良好的PPL

### 策略2：保守的联合剪枝（使用新的GQA感知代码）⭐⭐⭐⭐

```bash
CUDA_VISIBLE_DEVICES=2 python llama3.py --pruning_ratio 0.20 \
  --device cuda --eval_device cuda \
  --base_model /newdata/LLMs/Qwen2.5-7B \
  --block_wise \
  --block_mlp_layer_start 4 --block_mlp_layer_end 28 \
  --block_attention_layer_start 10 --block_attention_layer_end 24 \
  --save_ckpt_log_name Qwen_conservative_20 \
  --pruner_type taylor --taylor param_first \
  --max_seq_len 2048 \
  --save_model
```

**说明：**
- 使用较低的pruning_ratio (0.20 而不是 0.28)
- 只剪枝中间层的attention (10-24，跳过前10层和后4层)
- 新代码会自动使用896 (128*7)的consecutive_group_size

### 策略3：激进剪枝（实验性）⭐⭐⭐

```bash
CUDA_VISIBLE_DEVICES=2 python llama3.py --pruning_ratio 0.25 \
  --device cuda --eval_device cuda \
  --base_model /newdata/LLMs/Qwen2.5-7B \
  --block_wise \
  --block_mlp_layer_start 4 --block_mlp_layer_end 28 \
  --block_attention_layer_start 4 --block_attention_layer_end 28 \
  --save_ckpt_log_name Qwen_aggressive_25 \
  --pruner_type taylor --taylor param_first \
  --max_seq_len 2048 \
  --save_model
```

**警告：**
- 可能导致PPL大幅上升
- 需要仔细监控剪枝后的性能
- 建议先尝试策略1或2

## 验证剪枝效果

### 1. 检查PPL

正常范围：
- **wikitext2: 8-15** （如果>100说明模型已破坏）
- **PTB: 15-30** （如果>100说明模型已破坏）

### 2. 使用诊断脚本

```bash
# 检查剪枝后的模型
python diagnose_qwen_gqa.py prune_log/Qwen_mlp_only_25/pytorch_model.bin pruned
```

应该看到：
- ✓ 所有层的GQA ratio保持一致
- ✓ K和V维度匹配
- ✓ 没有维度整除性错误

### 3. 测试生成质量

```bash
python generate.py --model_type pruneLLM --ckpt prune_log/Qwen_mlp_only_25/pytorch_model.bin
```

生成的文本应该：
- 语义连贯
- 没有重复
- 没有乱码

## 代码修改说明

### 修改1：自动层数检测

```python
num_layers = len(model.model.layers)
if args.block_attention_layer_end > num_layers:
    args.block_attention_layer_end = num_layers
```

**效果：**现在可以使用layer_end=30，代码会自动调整为28

### 修改2：GQA ratio感知

```python
gqa_ratio = config.num_attention_heads / config.num_key_value_heads
if gqa_ratio >= 7:
    consecutive_group_size = head_dim * int(gqa_ratio)
```

**效果：**
- Llama-3 (4:1): consecutive_group_size = 128
- Qwen (7:1): consecutive_group_size = 896
- 确保剪枝时保持正确的GQA ratio

## 性能预期

### 策略1（MLP only）
- 参数减少：15-20%
- PPL增加：< 10%
- 推理速度：加速10-15%

### 策略2（保守联合）
- 参数减少：20-25%
- PPL增加：10-20%
- 推理速度：加速15-20%

### 策略3（激进）
- 参数减少：25-30%
- PPL增加：20-50%（可能不可接受）
- 推理速度：加速20-25%

## 与Llama-3/Mistral的对比

| 模型 | GQA Ratio | KV Heads | 层数 | 剪枝难度 |
|------|-----------|----------|------|----------|
| Llama-3-8B | 4:1 | 8 | 32 | 低 ⭐⭐ |
| Mistral-7B | 4:1 | 8 | 32 | 低 ⭐⭐ |
| **Qwen2.5-7B** | **7:1** | **4** | **28** | **高 ⭐⭐⭐⭐** |

**Qwen更难剪枝的原因：**
1. 更少的KV heads（4 vs 8）- 容错空间小
2. 更高的GQA ratio（7:1 vs 4:1）- 更敏感
3. 更少的层数（28 vs 32）- 影响更集中

## 故障排除

### 问题1：PPL爆炸 (>1000)

**可能原因：**
- attention层剪枝过度
- GQA ratio被破坏

**解决方案：**
- 使用策略1（仅MLP）
- 减小pruning_ratio
- 减少attention层剪枝范围

### 问题2：IndexError: index 28 is out of range

**原因：**使用了旧版本代码

**解决方案：**
```bash
git pull origin main
```

### 问题3：生成重复文本

**可能原因：**
- attention机制被破坏
- 剪枝过度

**解决方案：**
- 重新使用更保守的配置
- 检查GQA ratio是否保持一致

## 建议的实验流程

1. **先尝试策略1（MLP only）**
   ```bash
   CUDA_VISIBLE_DEVICES=2 python llama3.py --pruning_ratio 0.25 ...（策略1配置）
   ```

2. **检查结果**
   ```bash
   python diagnose_qwen_gqa.py prune_log/Qwen_mlp_only_25/pytorch_model.bin pruned
   ```

3. **如果PPL可接受，尝试策略2**
   ```bash
   CUDA_VISIBLE_DEVICES=2 python llama3.py --pruning_ratio 0.20 ...（策略2配置）
   ```

4. **对比性能**
   - 比较PPL
   - 比较生成质量
   - 比较推理速度

5. **选择最佳配置**
   - 平衡性能损失和速度提升
   - 根据应用场景调整

## 联系与反馈

如果遇到问题或发现新的有效配置，欢迎反馈！
