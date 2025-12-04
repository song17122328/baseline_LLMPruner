import os
import gc
import sys
import time
import json
import copy
import random
import argparse
from typing import Tuple

import torch
import numpy as np
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig, AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm

# Try to import Qwen2 RMSNorm for better compatibility
try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    HAS_QWEN2 = True
except ImportError:
    Qwen2RMSNorm = None
    HAS_QWEN2 = False

import LLMPruner.torch_pruning as tp 
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name), 
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # Use AutoModelForCausalLM to automatically detect model type (Llama, Qwen, etc.)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,  # Required for some models like Qwen
    )
    if args.device != "cpu":
        model.half()
        model.to(args.device)

    if args.test_before_train:
        logger.log("\n==================Generation Results before Pruning================\n")
        model = model.to(args.eval_device)
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.eval_device)

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                
                result = tokenizer.decode(generation_output[0])
                logger.log(result)
    
    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.eval_device)
    logger.log("PPL before pruning: {}".format(ppl))

    model.to(args.device)

    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    forward_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ]).to(args.device) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.

    if pruner_type == 'random':
        imp = tp.importance.RandomImportance()
    elif pruner_type == 'l1':
        imp = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        imp = llama_pruner.MagnitudeImportance(p=2)
    elif pruner_type == 'taylor':
        imp = llama_pruner.TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)
    else:
        raise NotImplementedError

    logger.log("Use {} pruner...".format(pruner_type))
    
    model.config.use_cache = False
    
    if args.block_wise:
        # Auto-detect number of layers and adjust if necessary
        num_layers = len(model.model.layers)
        logger.log(f"Detected {num_layers} layers in the model")

        # Adjust layer end parameters if they exceed the actual number of layers
        if args.block_attention_layer_end > num_layers:
            logger.log(f"Warning: block_attention_layer_end ({args.block_attention_layer_end}) > num_layers ({num_layers})")
            logger.log(f"Adjusting block_attention_layer_end to {num_layers}")
            args.block_attention_layer_end = num_layers

        if args.block_mlp_layer_end > num_layers:
            logger.log(f"Warning: block_mlp_layer_end ({args.block_mlp_layer_end}) > num_layers ({num_layers})")
            logger.log(f"Adjusting block_mlp_layer_end to {num_layers}")
            args.block_mlp_layer_end = num_layers

        # Detect and log GQA configuration
        config = model.config
        if hasattr(config, 'num_attention_heads') and hasattr(config, 'num_key_value_heads'):
            gqa_ratio = config.num_attention_heads / config.num_key_value_heads
            logger.log(f"Detected GQA configuration:")
            logger.log(f"  - num_attention_heads: {config.num_attention_heads}")
            logger.log(f"  - num_key_value_heads: {config.num_key_value_heads}")
            logger.log(f"  - GQA ratio: {gqa_ratio}:1")

            # Note: LLM-Pruner's importance alignment mechanism (in hf_llama_pruner.py)
            # automatically maintains GQA ratio by aligning Q/K/V importance tensors
            # We use head_dim for consecutive_groups to ensure complete head pruning
            if gqa_ratio >= 7:
                logger.log(f"⚠️  High GQA ratio detected ({gqa_ratio}:1)")
                logger.log(f"⚠️  Recommendation: Use conservative pruning_ratio (0.20-0.25)")
                logger.log(f"⚠️  Or prune MLP-only to avoid attention sensitivity")

        # Determine whether attention layers can be pruned safely
        # For models with few KV heads (like Qwen with only 4), we need at least head_dim
        # to be pruned to maintain complete head structure
        head_dim = model.model.layers[0].self_attn.head_dim
        kv_proj_dim = model.model.layers[0].self_attn.k_proj.out_features
        min_pruned_for_one_head = int(kv_proj_dim * args.pruning_ratio)

        # Check if we can prune at least one complete head
        can_prune_attention = min_pruned_for_one_head >= head_dim

        if can_prune_attention:
            logger.log(f"  ✓ Can prune attention layers safely")
            logger.log(f"  - Using consecutive_groups: {head_dim} (head_dim)")
            logger.log(f"  - KV proj dim: {kv_proj_dim}, will prune: ~{min_pruned_for_one_head} dims")
            consecutive_groups_dict = {
                layer.self_attn.k_proj: head_dim for layer in model.model.layers
            }
            attention_root_instances = [model.model.layers[i].self_attn.k_proj
                                       for i in range(args.block_attention_layer_start, args.block_attention_layer_end)]
        else:
            logger.log(f"  ⚠️  CANNOT prune attention layers safely with current pruning_ratio!")
            logger.log(f"  - Reason: will prune {min_pruned_for_one_head} dims < head_dim ({head_dim})")
            logger.log(f"  - This would break attention head structure")
            logger.log(f"  ⚠️  AUTO-SWITCHING to MLP-only pruning")
            logger.log(f"  - To prune attention, use pruning_ratio >= {head_dim / kv_proj_dim:.2f}")
            consecutive_groups_dict = {}
            attention_root_instances = []  # Don't prune attention!

        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio,
            "ignored_layers":[],
            "channel_groups": {
            },
            "consecutive_groups": consecutive_groups_dict,
            "customized_pruners": {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
                **({Qwen2RMSNorm: llama_pruner.hf_rmsnorm_pruner} if HAS_QWEN2 else {}),
            },
            "root_module_types": None,
            "root_instances": attention_root_instances +
                              [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]
        }
        if can_prune_attention:
            logger.log("Pruning Attention Layer = {}".format(list(range(args.block_attention_layer_start, args.block_attention_layer_end))))
        else:
            logger.log("Pruning Attention Layer = [] (skipped due to low pruning_ratio)")
        logger.log("Pruning MLP Layer = {}".format(list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))))

        pruner = tp.pruner.MetaPruner(
            model,
            forward_prompts,
            output_transform=lambda x: x.logits,
            **kwargs
        )
        model.zero_grad()

        logger.log("Start Pruning")
        for i in range(args.iterative_steps):

            if pruner_type in ['taylor']:
                example_prompts = get_examples('c4', tokenizer, args.num_examples, seq_len = 64).to(args.device)
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                if args.taylor in ['param_mix', 'param_second']:
                    for j in range(args.num_examples):
                        print(j)
                        batch_input = example_prompts[j].unsqueeze(0)
                        loss = model(batch_input, labels=batch_input).loss
                        logger.log("Loss = {}".format(loss))
                        loss.backward()

                        for module_param in model.parameters():
                            module_param.grad = module_param.grad * module_param.grad / args.num_examples
                            if hasattr(module_param, 'acc_grad'):
                                module_param.acc_grad += module_param.grad
                            else:
                                module_param.acc_grad = copy.deepcopy(module_param.grad)
                        model.zero_grad()
                        del loss.grad
                    
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("Loss = {}".format(loss))
                loss.backward()

            # 1. Consecutive for grouped KV
            # 2. 
            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))
        
            # modify inference-related attributes
            # Update num_heads based on actual pruned dimensions
            for layer_idx, layer in enumerate(model.model.layers):
                pruned_q_dim = layer.self_attn.q_proj.weight.data.shape[0]
                pruned_k_dim = layer.self_attn.k_proj.weight.data.shape[0]

                new_num_heads = pruned_q_dim // layer.self_attn.head_dim
                new_num_kv_heads = pruned_k_dim // layer.self_attn.head_dim

                # Update all possible attribute names for compatibility
                if hasattr(layer.self_attn, 'num_heads'):
                    layer.self_attn.num_heads = new_num_heads
                if hasattr(layer.self_attn, 'num_attention_heads'):
                    layer.self_attn.num_attention_heads = new_num_heads
                if hasattr(layer.self_attn, 'num_key_value_heads'):
                    layer.self_attn.num_key_value_heads = new_num_kv_heads

            # Log summary of pruned heads
            logger.log(f"Updated num_heads after pruning. Example layer 0: num_heads={model.model.layers[0].self_attn.num_heads if hasattr(model.model.layers[0].self_attn, 'num_heads') else 'N/A'}")

            # Update model config to reflect pruned dimensions
            if hasattr(model.config, 'num_attention_heads'):
                first_layer_num_heads = model.model.layers[0].self_attn.q_proj.weight.data.shape[0] // model.model.layers[0].self_attn.head_dim
                model.config.num_attention_heads = first_layer_num_heads
                logger.log(f"Updated model.config.num_attention_heads: {first_layer_num_heads}")

            if hasattr(model.config, 'num_key_value_heads'):
                first_layer_num_kv_heads = model.model.layers[0].self_attn.k_proj.weight.data.shape[0] // model.model.layers[0].self_attn.head_dim
                model.config.num_key_value_heads = first_layer_num_kv_heads
                logger.log(f"Updated model.config.num_key_value_heads: {first_layer_num_kv_heads}")

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None

        del pruner

    elif args.channel_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            "ignored_layers":[],
            #"round_to": model.config.num_attention_heads * 2,
            "channel_groups": {
                #layer.self_attn: layer.self_attn.num_heads for layer in model.model.layers
            },
            "customized_pruners": {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
                **({Qwen2RMSNorm: llama_pruner.hf_rmsnorm_pruner} if HAS_QWEN2 else {}),
                #LlamaAttention: llama_pruner.hf_attention_pruner,
            },
            "root_module_types": [LlamaRMSNorm],
        }

        pruner = tp.pruner.MetaPruner(
            model,
            forward_prompts,
            output_transform=lambda x: x.logits,
            **kwargs
        )
        model.zero_grad()
        
        logger.log("Start Pruning")
        for i in range(args.iterative_steps):

            if pruner_type in ['taylor']:
                example_prompts = get_examples('c4', tokenizer, 10, seq_len = 64)
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("Loss = {}".format(loss))
                loss.backward()

            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None

        # modify inferece-related attributes
        model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
        model.zero_grad()
        
        del pruner
            
    elif args.layer_wise:
        model.model.layers = model.model.layers[:args.layer]
        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    else:
        raise NotImplementedError
    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
    
    gc.collect()
    torch.cuda.empty_cache()

    if args.save_model:
        model.half()
        torch.save({
            'model': model, 
            'tokenizer': tokenizer,
        }, logger.best_checkpoint_path)
    
    if args.eval_device != "cpu":
        model.half()
    model.to(args.eval_device)

    model.config.pad_token_id = tokenizer.pad_token_id = 0 
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if args.test_after_train:
        logger.log("\n==================Generation Results After Pruning================\n")
        
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.eval_device)

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                
                result = tokenizer.decode(generation_output[0])
                logger.log(result)
        
        logger.log("\n==================Finish================\n")
    
    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.eval_device)
    logger.log("PPL after pruning: {}".format(ppl))
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="meta-llama/Llama-2-7b-hf", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')
    parser.add_argument('--pruner_type', type=str, default='l2', help='pruner type')

    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument('--channel_wise', action='store_true', help='channel wise')
    parser.add_argument('--block_wise', action='store_true', help='block wise')
    parser.add_argument('--layer_wise', action='store_true', help='layer wise')
    parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)

    parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping')
    parser.add_argument('--global_pruning', action='store_true', help='whether global pruning')
    parser.add_argument('--taylor', type=str, default='param_first', help='choose from [vectorize, param_second, param_first, param_mix]')
    parser.add_argument('--num_examples', type=int, default=10)

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
