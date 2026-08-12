import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset

model_name = "facebook/opt-125m"
prompt = "The future of AI is"


# Measure model inference time and generate sample output for a given prompt
def benchmark(model, tokenizer, prompt, max_new_tokens=50):
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
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

    elapsed_time = end - start
    decoded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded_text, elapsed_time


def get_folder_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)  # MB


# ---------------------------------------------------------------------------
# 1. Load base model and tokenizer, run benchmark, save full model
# ---------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

initial_output, initial_time = benchmark(model, tokenizer, prompt)

save_dir_full = model_name.split("/")[-1] + "-full"
model.save_pretrained(save_dir_full, safe_serialization=True)
tokenizer.save_pretrained(save_dir_full)

# Free memory before quantization
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 2. Prepare calibration dataset for GPTQ
# ---------------------------------------------------------------------------
calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train",
).select(range(256))["text"]

# ---------------------------------------------------------------------------
# 3. Quantize the model with gptqmodel (no optimum/transformers.GPTQConfig)
# ---------------------------------------------------------------------------
quantize_config = QuantizeConfig(
    bits=4,
    group_size=128,
)

quant_model = GPTQModel.load(model_name, quantize_config)
quant_model.quantize(calibration_dataset, batch_size=2)

save_dir_quant = model_name.split("/")[-1] + "-gptq-config"
quant_model.save(save_dir_quant)
tokenizer.save_pretrained(save_dir_quant)

del quant_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 4. Reload quantized model and benchmark
# ---------------------------------------------------------------------------
reloaded_quant_model = GPTQModel.load(save_dir_quant)
quant_tokenizer = AutoTokenizer.from_pretrained(save_dir_quant)

quant_output, quant_time = benchmark(reloaded_quant_model, quant_tokenizer, prompt)

# ---------------------------------------------------------------------------
# 5. Compare model sizes and print results
# ---------------------------------------------------------------------------
initial_size = get_folder_size(save_dir_full)
quant_size = get_folder_size(save_dir_quant)

print("=== Full Model ===")
print(f" Output: {initial_output}")
print(f" Size: {initial_size:.2f} MB")
print(f" Inference time: {initial_time:.4f} s")

print("\n=== Quantized Model ===")
print(f" Output: {quant_output}")
print(f" Size: {quant_size:.2f} MB")
print(f" Inference time: {quant_time:.4f} s")
