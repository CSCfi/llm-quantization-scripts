import os
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from llmcompressor.entrypoints.oneshot import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

model_name = "tiiuae/falcon-rw-1b"
dataset_name = "HuggingFaceH4/ultrachat_200k"
dataset_split = "train_sft"
num_calibration_samples = 256
max_seq_length = 512
prompt = "The future of AI is"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
dispatch_for_generation(model) # prepare model for efficient text generation

# Measure model inference time and generate sample output for a given prompt
def benchmark(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warm-up run (to remove cold start effects)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

    elapsed_time = end - start
    decoded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return decoded_text, elapsed_time

# Run benchmark on full model
initial_output, initial_time = benchmark(model, tokenizer, prompt)

# Save full model before quantization
save_dir_full = model_name.split("/")[-1] + "-full"
model.save_pretrained(save_dir_full, safe_serialization=True)
tokenizer.save_pretrained(save_dir_full)

# Load and preprocess dataset
dataset = load_dataset(dataset_name, split=f"{dataset_split}[:{num_calibration_samples}]")
dataset = dataset.shuffle(seed=42)

# Flatten each example's list of messages into a single text string with roles
def preprocess(example):
    text = ""
    for msg in example["messages"]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        text += f"{role}: {content}\n"
    return {"text": text}

dataset = dataset.map(preprocess, batched=False)

# Tokenize dataset
def tokenize_sample(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False
    )

dataset = dataset.map(tokenize_sample, remove_columns=dataset.column_names, batched=False)

# Set quantization recipe
recipe = AWQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

# Quantize model
oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=max_seq_length,
    num_calibration_samples=num_calibration_samples
)

# Save quantized model
save_dir_quant = model_name.split("/")[-1] + "-awq"
model.save_pretrained(save_dir_quant, safe_serialization=True)
tokenizer.save_pretrained(save_dir_quant)

# Reload quantized model for inference
quant_model = AutoModelForCausalLM.from_pretrained(save_dir_quant, device_map="auto")
quant_tokenizer = AutoTokenizer.from_pretrained(save_dir_quant)
dispatch_for_generation(quant_model) # prepare model for efficient text generation

# Run benchmark on quantized model
quant_output, quant_time = benchmark(quant_model, quant_tokenizer, prompt)

# Compare model sizes
def get_folder_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)  # MB

initial_size = get_folder_size(save_dir_full)
quant_size = get_folder_size(save_dir_quant)

# Print results
print("=== Full Model ===")
print(f" Output: {initial_output}")
print(f" Size: {initial_size:.2f} MB")
print(f" Inference time: {initial_time:.4f} s")

print("\n=== Quantized Model ===")
print(f" Output: {quant_output}")
print(f" Size: {quant_size:.2f} MB")
print(f" Inference time: {quant_time:.4f} s")
