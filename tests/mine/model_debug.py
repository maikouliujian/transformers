# from transformers.testing_utils import (
#     TOKEN,
#     CaptureLogger,
#     LoggingLevel,
#     TemporaryHubRepo,
#     TestCasePlus,
#     hub_retry,
#     is_staging_test,
#     require_accelerate,
#     require_flax,
#     require_read_token,
#     require_safetensors,
#     require_tf,
#     require_torch,
#     require_torch_accelerator,
#     require_torch_gpu,
#     require_torch_multi_accelerator,
#     require_usr_bin_time,
#     slow,
#     torch_device,
# )

from transformers import AutoModelForCausalLM,AutoTokenizer

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(
    "/Users/lj/mine/klx/modes/Matroyshka-ReRanker-document",
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "/Users/lj/mine/klx/modes/Matroyshka-ReRanker-document",
    trust_remote_code=True,
    device_map="auto")
# for name, param in model.named_parameters():
#     print(f"参数名: {name}")
#     print(f"形状: {param.shape}")
#     print(f"值:\n{param.data}")
#     print("-" * 50)
# 生成文本
inputs = tokenizer("AI will", return_tensors="pt").to("cuda")
# breakpoint()
outputs = model.generate(**inputs, max_length=20)
print(tokenizer.decode(outputs[0]))

if __name__ == '__main__':
    pass