import json
import PIL
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration,BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import gc
import time
from peft import LoraConfig, get_peft_model
from trl import SFTConfig,SFTTrainer

def clear_memory():
    # Delete variables if they exist in the current global scope
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "trainer" in globals():
        del globals()["trainer"]
    if "peft_model" in globals():
        del globals()["peft_model"]
    if "bnb_config" in globals():
        del globals()["bnb_config"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


clear_memory()


dataset = [ json.loads(line) for line in open("/data/phd/tiankaibin/dataset/data/qwen_format_sft_train.jsonl", "r").readlines() ]
for data in dataset:
    data['messages'][1]['content'][1]['image'] = PIL.Image.open(data['messages'][1]['content'][1]['image'])
train_dataset = [data['messages'] for data in dataset]

dataset = [ json.loads(line) for line in open("/data/phd/tiankaibin/dataset/data/qwen_format_sft_test.jsonl", "r").readlines() ]
for data in dataset:
    data['messages'][1]['content'][1]['image'] = PIL.Image.open(data['messages'][1]['content'][1]['image'])
eval_dataset = [data['messages'] for data in dataset]


processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True,device_map="auto",torch_dtype=torch.bfloat16)



# def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
#     print(sample)
#     # Prepare the text input by applying the chat template
#     text_input = processor.apply_chat_template(
#         sample[1:2], tokenize=False, add_generation_prompt=True  # Use the sample without the system message
#     )

#     # Process the visual input from the sample
#     image_inputs, _ = process_vision_info(sample)

#     # Prepare the inputs for the model
#     model_inputs = processor(
#         text=[text_input],
#         images=image_inputs,
#         return_tensors="pt",
#     ).to(
#         device
#     )  # Move inputs to the specified device

#     # Generate text with the model
#     generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

#     # Trim the generated ids to remove the input ids
#     trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

#     # Decode the output text
#     output_text = processor.batch_decode(
#         trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )

#     return output_text[0]  # Return the first decoded output text

# output = generate_text_from_sample(model, processor, train_dataset[0])
# # print(output)




training_args = SFTConfig(
    output_dir="qwen2-7b-instruct-trl-sft",  # Directory to save the model
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,  # Batch size for evaluation
    gradient_accumulation_steps=8,  # Steps to accumulate gradients
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=5e-6,  # Learning rate for training
    lr_scheduler_type="cosine",  # Type of learning rate scheduler
    # Logging and evaluation
    logging_steps=10,  # Steps interval for logging
    eval_steps=10,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    save_strategy="steps",  # Strategy for saving the model
    save_steps=20,  # Steps interval for saving
    metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    greater_is_better=False,  # Whether higher metric values are better
    load_best_model_at_end=True,  # Load the best model after training
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    tf32=True,  # Use TensorFloat-32 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    # Hub and reporting
    push_to_hub=False,  # Whether to push model to Hugging Face Hub
    report_to="wandb",  # Reporting tool for tracking metrics
    # Gradient checkpointing settings
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    # max_seq_length=1024  # Maximum sequence length for input
)
training_args.remove_unused_columns = False  # Keep unused columns in dataset


# Create a data collator to encode text and image pairs
def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch



trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,
)

trainer.train()