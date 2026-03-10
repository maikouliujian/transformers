import json

# 读取 index.json 文件
with open("./index.json", "r") as f:
    index_data = json.load(f)

# 获取 metadata 和 weight_map
# metadata = index_data["metadata"]
weight_map = index_data["weight_map"]
a = []
b = []
for weight_name in weight_map:
    if '_scale_inv' in weight_name:
        a.append(weight_name) # 45808 45808
        continue
    if ('down_proj' in weight_name or 'gate_proj' in weight_name or
            'up_proj' in weight_name or 'q_a_proj' in weight_name or
            'q_b_proj' in weight_name or 'kv_a_proj_with_mqa' in weight_name
            or 'kv_b_proj' in weight_name or 'o_proj' in weight_name):
        b.append(weight_name)

print(len(a))
print(len(b))
# with open("a.txt", "w") as f:
#     for item in a:
#         f.write(f"{item}\n")
#
# with open("b.txt", "w") as f:
#     for item in b:
#         f.write(f"{item}\n")

if __name__ == '__main__':
    pass