from transformers import AutoModelForCausalLM,AutoTokenizer

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    "./",
    trust_remote_code=True,
    device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(
    "./",
    trust_remote_code=True
)
for name, param in model.named_parameters():
    print(f"参数名: {name}")
    print(f"形状: {param.shape}")
    print(f"值:\n{param.data}")
    print("-" * 50)
# 生成文本
inputs = tokenizer("AI will", return_tensors="pt").to("cuda")
# breakpoint()
outputs = model.generate(**inputs, max_length=20)
print(tokenizer.decode(outputs[0]))

if __name__ == '__main__':
    pass