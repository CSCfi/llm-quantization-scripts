import os
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

model_name =  "facebook/opt-125m"
prompt = "The future of AI is"

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

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Run benchmark on full model
initial_output, initial_time = benchmark(model, tokenizer, prompt)

# Set BitsAndBytes Config
bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.bfloat16,
   bnb_4bit_use_double_quant=True,
   bnb_4bit_quant_storage=torch.bfloat16,
)

# Quantize the model with bitsandbytes
quant_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto")

# Run benchmark on quantized model
quant_output, quant_time = benchmark(quant_model, tokenizer, prompt)

# Print results
print("=== Full Model ===")
print(f" Output: {initial_output}")
print(f" Inference time: {initial_time:.4f} s")

print("\n=== Quantized Model ===")
print(f" Output: {quant_output}")
print(f" Inference time: {quant_time:.4f} s")
