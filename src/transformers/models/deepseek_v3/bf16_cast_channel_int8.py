import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
# from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download


def weight_quant(tensor: torch.Tensor):
    assert tensor.dim() == 2
    qmax = 127.0
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]  # [rows, 1]
    scale = abs_max / qmax  # [rows, 1]
    assert scale.shape == (tensor.shape[0], 1)
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)


def main(bf16_path, int8_path, model_name="deepseek-ai/DeepSeek-R1"):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(int8_path, exist_ok=True)
    model_index_file = os.path.join(int8_path, "pytorch_model.bin.index.json")
    config_file = os.path.join(int8_path, "config.json")

    if not os.path.exists(model_index_file) or not os.path.exists(config_file):
        snapshot_download(
            repo_id=model_name,
            ignore_patterns=["*.bin"],
            local_dir=int8_path,
            local_dir_use_symlinks=False
        )
        print(f"model index file and config file downloaded to {int8_path}")

        # modify config.json and save it
        config = json.load(open(config_file))
        # delete quantization_config
        config.pop("quantization_config", None)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"config.json modified and saved to {config_file}")

    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    # scale_count = len([key for key in weight_map.keys() if key.endswith("_scale_inv")])

    safetensor_files = list(glob(os.path.join(bf16_path, "*.bin")))
    safetensor_files.sort()
    quant_count = 0
    new_weight_map = {}
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        # state_dict = load_file(safetensor_file, device="cuda")
        state_dict = torch.load(safetensor_file, map_location="cuda")
        new_state_dict = {}
        for weight_name, weight in state_dict.items():
            if ('down_proj' in weight_name or 'gate_proj' in weight_name or
                    'up_proj' in weight_name or 'q_a_proj' in weight_name or
                    'q_b_proj' in weight_name or 'kv_a_proj_with_mqa' in weight_name
                    or 'kv_b_proj' in weight_name or 'o_proj' in weight_name):
                scale_inv_name = f"{weight_name}_scale_inv"
                assert weight.element_size() == 2
                quant_count += 1
                int8_weight, scale_inv = weight_quant(weight)
                new_state_dict[weight_name] = int8_weight
                new_scale_name = scale_inv_name.replace("_scale_inv", "_scale")
                new_state_dict[new_scale_name] = scale_inv

                new_weight_map[weight_name] = file_name
                new_weight_map[new_scale_name] = file_name
            else:
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name
        new_safetensor_file = os.path.join(int8_path, file_name)
        # save_file(new_state_dict, new_safetensor_file)
        torch.save(new_state_dict, new_safetensor_file)
    # assert quant_count == scale_count
    print(f"{quant_count} weights are quantized.")

    # modify model.safetensors.index.json
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    model_index["weight_map"] = new_weight_map
    with open(model_index_file, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"model.safetensors.index.json modified and saved to {model_index_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-bf16-hf-path", type=str, required=True)
    parser.add_argument("--output-int8-hf-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1")

    args = parser.parse_args()
    main(args.input_bf16_hf_path, args.output_int8_hf_path, args.model_name)
    print("done")