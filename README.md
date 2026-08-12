# Quantization

This repository explores quantization methods for Large Language Models (LLMs).  
Quantization is a key technique for reducing model size and inference cost, enabling LLMs to run efficiently on consumer hardware or limited GPU memory.

We provide examples and experiments for:

- **[Bitsandbytes (bnb)](https://github.com/CSCfi/llm-quantization-scripts/tree/main/BitsAndBytes)** – nf4 quantization using the Hugging Face integration.
- **[AWQ (Activation-aware Weight Quantization)](https://github.com/CSCfi/llm-quantization-scripts/tree/main/AWQ)** – a method that preserves accuracy by considering activation statistics.
- **[GPTQ (Gradient Post-training Quantization)](https://github.com/CSCfi/llm-quantization-scripts/tree/main/GPTQ)** – post-training quantization optimized for autoregressive transformers.

---

## 📖 What to Expect in This Repo

1. **Implementation Examples**  
   - Scripts for loading, quantizing, and saving models.  
   - Examples include small models (for example; `facebook/opt-125m`) so you can try things quickly, and notes for scaling to larger models.

2. **Benchmarks**  
   - Inference time comparisons before and after quantization.  
   - Model size reduction (disk footprint in MB).  

3. **Guides & Utilities**  
   - Helper functions for measuring model size, timing inference, and testing the quantized model.  
   - Notes on how to run the examples on LUMI and Roihu.
     
---
