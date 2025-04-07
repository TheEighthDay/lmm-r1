import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from peft import PeftModel

# 加载基础模型和分词器
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"  # 替换为实际的基础模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 加载 LoRA 模型
lora_model_path = "/data/phd/tiankaibin/lmm-r1/finetune/video-llm-output"  # 替换为实际的 LoRA 模型路径
lora_model = PeftModel.from_pretrained(base_model, lora_model_path)

# 合并 LoRA 参数到基础模型
merged_model = lora_model.merge_and_unload()

# 保存合并后的模型
output_model_path = "/data/phd/tiankaibin/lmm-r1/finetune/lora_merged_model"
merged_model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)