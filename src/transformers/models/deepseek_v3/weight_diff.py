import torch
import numpy as np
from glob import glob
import os
from tqdm import tqdm


def compare_layers_by_range(original_path, quantized_path, layer_indices=[0, 1, 60], proj_types=['q_proj', 'o_proj']):
    """
    按层索引范围对比指定投影类型的权重（无可视化，只输出结果）

    Args:
        original_path: BF16原始模型路径
        quantized_path: INT8量化模型路径
        layer_indices: 要对比的层索引列表，如 [0, 1, 2, 60]
        proj_types: 要对比的投影类型，如 ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    Returns:
        results: 包含每个层对比结果的字典
    """

    print("=" * 80)
    print(f"BF16 vs INT8 Quantization Comparison")
    print("=" * 80)
    print(f"Layer indices: {layer_indices}")
    print(f"Projection types: {proj_types}")

    # 加载权重
    print("\n📂 Loading weights...")
    # original_files = sorted(glob(os.path.join(original_path, "*.bin")))
    # quantized_files = sorted(glob(os.path.join(quantized_path, "*.bin")))
    original_files = [f'{original_path}/pytorch_model-00004-of-00119.bin',f'{original_path}/pytorch_model-00001-of-00119.bin']
    quantized_files = [f'{quantized_path}/pytorch_model-00004-of-00119.bin',f'{quantized_path}/pytorch_model-00001-of-00119.bin']
    original_state_dict = {}
    quantized_state_dict = {}

    for f in tqdm(original_files, desc="Loading original"):
        original_state_dict.update(torch.load(f, map_location='cpu'))

    for f in tqdm(quantized_files, desc="Loading quantized"):
        quantized_state_dict.update(torch.load(f, map_location='cpu'))

    print(f"\n✅ Loaded {len(original_state_dict)} original weights")
    print(f"✅ Loaded {len(quantized_state_dict)} quantized weights")

    # 构建要对比的层名列表
    layer_names = []
    for idx in layer_indices:
        for proj in proj_types:
            layer_names.append(f"model.layers.{idx}.mlp.experts.22.{proj}")

    print(f"\n📋 Comparing {len(layer_names)} layers...")

    # 进行对比
    results = {}
    for layer_name in tqdm(layer_names, desc="Comparing"):
        weight_name = f"{layer_name}.weight"
        scale_name = f"{layer_name}.weight_scale"

        if weight_name in original_state_dict and scale_name in quantized_state_dict:
            # 获取原始权重
            original = original_state_dict[weight_name].float()

            # 获取量化权重和scale
            quantized = quantized_state_dict[weight_name]
            scale = quantized_state_dict[scale_name]

            # 处理 scale 形状: [out_features, 1] -> [out_features]
            # if scale.dim() == 2 and scale.shape[1] == 1:
            #     scale = scale.squeeze(-1)

            # 反量化
            dequantized = quantized.float() * scale

            # 计算误差
            error = dequantized - original
            abs_error = error.abs()
            rel_error = abs_error / (original.abs() + 1e-8)

            # 统计指标
            mse = (error ** 2).mean().item()
            cos_sim = torch.cosine_similarity(
                original.flatten(), dequantized.flatten(), dim=0
            ).item()
            max_error = abs_error.max().item()
            mean_error = abs_error.mean().item()
            error_std = error.std().item()

            results[layer_name] = {
                'mse': mse,
                'cos_sim': cos_sim,
                'max_error': max_error,
                'mean_error': mean_error,
                'error_std': error_std,
                'original_shape': tuple(original.shape),
                'quantized_shape': tuple(quantized.shape),
                'scale_shape': tuple(scale.shape),
            }

    # 打印结果表格
    print("\n" + "=" * 100)
    print(f"{'Layer':<40} {'Shape':<20} {'MSE':<12} {'CosSim':<10} {'MaxError':<12} {'MeanError':<12}")
    print("=" * 100)

    for layer_name, metrics in results.items():
        shape_str = f"{metrics['original_shape'][0]}x{metrics['original_shape'][1]}"
        print(
            f"{layer_name:<40} {shape_str:<20} {metrics['mse']:<12.6f} {metrics['cos_sim']:<10.6f} {metrics['max_error']:<12.6f} {metrics['mean_error']:<12.6f}")

    # 汇总统计
    if results:
        mses = [r['mse'] for r in results.values()]
        cos_sims = [r['cos_sim'] for r in results.values()]
        max_errors = [r['max_error'] for r in results.values()]

        print("\n" + "=" * 100)
        print("📊 Summary Statistics")
        print("=" * 100)
        print(f"Total layers compared: {len(results)}")
        print(
            f"MSE - Mean: {np.mean(mses):.6f}, Std: {np.std(mses):.6f}, Min: {np.min(mses):.6f}, Max: {np.max(mses):.6f}")
        print(f"CosSim - Mean: {np.mean(cos_sims):.6f}, Min: {np.min(cos_sims):.6f}, Max: {np.max(cos_sims):.6f}")
        print(f"MaxError - Mean: {np.mean(max_errors):.6f}")

    return results


# 使用示例
if __name__ == "__main__":
    original_path = "/data/Jiutian-236B-32k-chat-V0.3.0-260211"
    quantized_path = "/data/Jiutian-236B-32k-chat-V0.3.0-260211-int8"

    # 对比第0、1、60层的 q_proj 和 o_proj
    results = compare_layers_by_range(
        original_path,
        quantized_path,
        # layer_indices=[0, 1, 2],
        layer_indices=[1],
        proj_types=['down_proj', 'gate_proj', 'up_proj']
    )

    # 也可以对比更多层和更多投影类型
    # results = compare_layers_by_range(
    #     original_path,
    #     quantized_path,
    #     layer_indices=[0, 1, 2, 3, 58, 59, 60],
    #     proj_types=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj']
    # )