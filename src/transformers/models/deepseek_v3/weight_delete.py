#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化版脚本：读取model.safetensors.index.json中的weight_map，
筛选出层数 <= 1 的权重（即model.layers.0和model.layers.1），
并删除当前目录下对应的.safetensors文件
"""

import json
import os
import re


def parse_layer_number(weight_name):
    """从权重名称中解析层号"""
    match = re.search(r"model\.layers\.(\d+)\.", weight_name)
    return int(match.group(1)) if match else None


def main():
    current_dir = os.getcwd()
    # todo 需要修改
    index_file = os.path.join(current_dir, "index.json")

    if not os.path.exists(index_file):
        print(f"错误：找不到 {index_file}")
        return

    # 读取索引文件
    with open(index_file, 'r', encoding='utf-8') as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    print(f"总共有 {len(weight_map)} 个权重条目")

    # 找出需要保留的权重（层号 <= 1）
    weights_to_keep = {}
    files_to_keep = set()

    for weight_name, file_name in weight_map.items():
        layer_num = parse_layer_number(weight_name)
        # todo 需要修改，保留几层
        if layer_num is None or layer_num <= 1:
            weights_to_keep[weight_name] = file_name
            files_to_keep.add(file_name)

    print(f"\n需要保留的层: 0 和 1")
    print(f"需要保留的权重数量: {len(weights_to_keep)}")
    print(f"需要保留的文件: {sorted(files_to_keep)}")

    # 找出要删除的文件
    files_to_delete = set(weight_map.values()) - files_to_keep
    print(f"\n需要删除的文件数量: {len(files_to_delete)}")
    if files_to_delete:
        print("需要删除的文件:")
        for f in sorted(files_to_delete):
            print(f"  - {f}")

    # 确认操作
    print("\n" + "=" * 50)
    print("警告：此操作将永久删除文件！")
    print("=" * 50)

    response = input(f"\n是否确定删除上述 {len(files_to_delete)} 个文件？(yes/no): ")

    if response.lower() == 'yes':
        # 删除文件
        for file_name in files_to_delete:
            file_path = os.path.join(current_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"已删除: {file_name}")
            else:
                print(f"文件不存在: {file_name}")

        # 备份原索引文件
        backup_file = index_file + ".backup"
        os.rename(index_file, backup_file)
        print(f"\n已备份原索引文件到: {backup_file}")

        # 创建新索引文件
        new_index_data = {
            "metadata": index_data.get("metadata", {}),
            "weight_map": weights_to_keep
        }

        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(new_index_data, f, indent=2, ensure_ascii=False)

        print(f"已更新索引文件，保留 {len(weights_to_keep)} 个权重条目")
        print("操作完成！")
    else:
        print("操作已取消")


if __name__ == "__main__":
    main()