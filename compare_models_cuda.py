import torch
import argparse
import os
import json
from transformers import AutoConfig, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="对比原始HF模型与剪枝后的Bin模型结构")
    parser.add_argument("--original_path", type=str, required=True, 
                        help="原始 HuggingFace 模型的本地路径")
    parser.add_argument("--pruned_path", type=str, required=True, 
                        help="剪枝后的 .bin 模型文件路径")
    parser.add_argument("--device", type=str, default="cuda:7", 
                        help="加载剪枝模型使用的设备")
    parser.add_argument("--output_json", type=str, default="structure_diff.json", 
                        help="结果保存的 JSON 文件路径")
    return parser.parse_args()

def load_pruned_state_dict(bin_path, device):
    print(f"[*] 正在加载剪枝模型文件到 [{device}]: {bin_path} ...")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"找不到文件: {bin_path}")
    
    try:
        # 添加 weights_only=False
        data = torch.load(bin_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"[!] 加载到 {device} 失败，尝试回退到 CPU 加载... 错误信息: {e}")
        data = torch.load(bin_path, map_location="cpu", weights_only=False)
    
    # --- 自动解包逻辑 ---
    if hasattr(data, 'state_dict') and not isinstance(data, dict):
        print("[*] 检测到文件保存的是完整模型对象(Model Object)，正在提取 state_dict...")
        return data.state_dict()

    if isinstance(data, dict):
        if "state_dict" in data:
            print("[*] 检测到 'state_dict' 键，正在提取内部权重...")
            return data["state_dict"]
        elif "model" in data:
            print("[*] 检测到 'model' 键，正在提取内部权重...")
            if isinstance(data["model"], dict):
                return data["model"]
            elif hasattr(data["model"], 'state_dict'):
                return data["model"].state_dict()
            
    return data

def load_original_structure(hf_path):
    print(f"[*] 正在解析原始模型结构: {hf_path} ...")
    try:
        config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True) 
        print("[*] 原始模型结构实例化完成 (Meta Device)。")
        return model
    except Exception as e:
        print(f"[!] 加载原始模型失败: {e}")
        exit(1)

def compare_and_save(original_model, pruned_state_dict, output_json_path):
    print("\n" + "="*100)
    print(f"结构对比分析 (所有层)")
    print("="*100 + "\n")

    if not isinstance(pruned_state_dict, dict):
        print(f"[错误] 提取后的权重不是字典，而是: {type(pruned_state_dict)}")
        return

    # 用于 JSON 的数据结构
    json_output = {
        "summary": {},
        "layers": []
    }
    
    # 用于控制台打印的列表
    console_rows = []
    
    original_params = dict(original_model.named_parameters())
    
    matched_count = 0
    missing_count = 0
    changed_count = 0
    
    # 遍历
    for name, param in original_params.items():
        orig_shape = tuple(param.shape)
        
        if name in pruned_state_dict:
            pruned_tensor = pruned_state_dict[name]
            
            if not torch.is_tensor(pruned_tensor):
                continue

            pruned_shape = tuple(pruned_tensor.shape)
            matched_count += 1
            
            status = "SAME"
            if orig_shape != pruned_shape:
                status = "PRUNED"
                changed_count += 1
            
            # 添加到控制台列表
            console_rows.append([name, str(orig_shape), str(pruned_shape), status])
            
            # 添加到 JSON 列表 (保存为列表格式方便后续程序读取)
            json_output["layers"].append({
                "name": name,
                "original_shape": list(orig_shape),
                "pruned_shape": list(pruned_shape),
                "status": status
            })
            
        else:
            missing_count += 1
            console_rows.append([name, str(orig_shape), "MISSING", "REMOVED"])
            json_output["layers"].append({
                "name": name,
                "original_shape": list(orig_shape),
                "pruned_shape": None,
                "status": "REMOVED"
            })

    # 打印控制台表格
    headers = ["Layer Name", "Original Shape", "Pruned Shape", "Status"]
    fmt_str = "{:<60} | {:<20} | {:<20} | {:<10}"
    
    print(fmt_str.format(*headers))
    print("-" * 120)
    
    # 打印每一行 (不截断)
    for row in console_rows:
        print(fmt_str.format(*row))

    # 统计信息
    total_rows = len(console_rows)
    summary = {
        "total_layers": total_rows,
        "matched_layers": matched_count,
        "pruned_changed_layers": changed_count,
        "missing_layers": missing_count
    }
    json_output["summary"] = summary

    print("\n" + "="*30)
    print("总结报告")
    print("="*30)
    print(f"总参数层数: {total_rows}")
    print(f"匹配层数: {matched_count}")
    print(f"剪枝变化层数: {changed_count}")
    print(f"丢失层数: {missing_count}")
    print("="*30)
    
    # 保存 JSON
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4)
        print(f"\n[+] 完整结构结果已保存至: {output_json_path}")
    except Exception as e:
        print(f"\n[!] 保存 JSON 失败: {e}")

if __name__ == "__main__":
    args = parse_args()
    
    # 1. 加载
    pruned_sd = load_pruned_state_dict(args.pruned_path, args.device)
    
    # 2. 结构
    orig_model = load_original_structure(args.original_path)
    
    # 3. 对比、打印并保存
    compare_and_save(orig_model, pruned_sd, args.output_json)