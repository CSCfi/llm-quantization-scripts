# Quantization Examples with Transformers & LLM Compressor

This repository contains two practical examples of applying GPTQ quantization to LLMs.  
Both examples currently use the small **OPT-125M** model for demonstration, but the code is written so you can swap in larger models.

1. **GPTQConfig** — Uses Hugging Face `transformers` and [`GPTQConfig`](https://huggingface.co/docs/transformers/en/quantization/gptq) to quantize the **OPT-125M** model.
2. **GPTQModifier** — Uses [LLM Compressor](https://github.com/vllm-project/llm-compressor) with a GPTQ recipe to quantize the **OPT-125M** model. 

---

## Installations

The CSC preinstalled PyTorch module covers most of the libraries needed to run these examples
(torch, transformers, datasets, accelerate). The rest can be installed on top of the module in a virtual environment.

### Load the module
```bash
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.7
```
### Create and activate a virtual environment using system packages
```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
```
### Install packages
```bash
pip install optimum --cache-dir ./.pip-cache
# tested with optimum==1.27.0
```
The flag --cache-dir points the pip cache to the current (scratch) folder instead of the default (home directory), to avoid filling up home directory quota. 

The GPTQmodel library is needed for the **gptq-config** example. To install it on Puhti or Mahti, you need to use a GPU interactively when installing, or set the following environment variable:

- For Puhti:`export TORCH_CUDA_ARCH_LIST="7.0"`
- For Mahti: `export TORCH_CUDA_ARCH_LIST="8.0"`

Then install with: 
```bash
pip install gptqmodel==4.0.0 --no-build-isolation --cache-dir ./.pip-cache
```

This version of gptqmodel is compatible with PyTorch 2.7.

Note: When quantizing models that use Rotary Positional Embeddings (RoPE), such as LlaMA, you might encounter runtime errors related to rotary dimensions. The current fix is to downgrade transformers to version 4.51.3.

For the **gptq-modifier** example, you need to install the llmcompressor library.

```bash
pip install llmcompressor --cache-dir ./.pip-cache
# tested with llmcompressor==0.7.1
```
## Usage

The launch scripts for gptq-config are: 

- `run-gptq-config-lumi.sh` - quantizes model on LUMI with 1 GPU 
- `run-gptq-config-mahti.sh` - quantizes model on Mahti with 1 GPU
- `run-gptq-config-puhti.sh` - quantizes model on Puhti with 1 GPU

Similarly, for gptq-modifier:

- `run-gptq-modifier-lumi.sh` - quantizes model on LUMI with 1 GPU 
- `run-gptq-modifier-mahti.sh` - quantizes model on Mahti with 1 GPU
- `run-gptq-modifier-puhti.sh` - quantizes model on Puhti with 1 GPU

**Note:** the scripts are made to be run on `gputest` or `dev-g` partition with a 30 minutes time-limit. You have to select the proper partition for longer jobs for your real runs. Additionally, change the `--account` parameter to your own project code. 

For example to run on LUMI, you would run the command:

```bash
sbatch run-gptq-config-lumi.sh
```
You can also increase the memory and number of GPUs if you decide to run quantization on larger models.

## `gptq-config.py`
- Uses Hugging Face 🤗 `transformers` with [`GPTQConfig`](https://huggingface.co/docs/transformers/en/quantization/gptq).
- Saves both the full-precision and quantized models. 
- Compares outputs, inference latency, and model size.

## `gptq-modifier.py`
- Uses [LLM Compressor](https://github.com/vllm-project/llm-compressor) with a `GPTQModifier` recipe.
- Runs explicit **calibration** on a subset of the [Ultrachat-200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) dataset.
- Saves both the full-precision and quantized models.
- Compares outputs, inference latency, and model size.
- Provides finer control over quantization schemes (e.g. `W4A16`, `ignore=["lm_head"]`).

## Output Includes
- Generated text before and after quantization.
- Inference time comparison. 
- Model size (MB) before and after quantization.
- Note that the effect of quantization on inference might not be noticeable for smaller models.

## Notes
- The current scripts use **OPT-125M** for fast experimentation. You can replace `model_name` with a larger model. In this case, you might want to disable saving the models.
- For large models, `device_map="auto"` lets 🤗 Accelerate handle placement across GPUs.
- The `GPTQConfig` path is simpler and integrates directly with Hugging Face pipelines, while the `GPTQModifier` path gives you more flexibility for research and custom recipes.
- Feel free to experiment with different values for `num_calibration_samples` and `max_seq_lenght`.
