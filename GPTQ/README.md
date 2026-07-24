# Quantization Examples with Transformers & LLM Compressor

This repository contains two practical examples of applying GPTQ quantization to LLMs.  
Both examples currently use the small **OPT-125M** model for demonstration, but the code is written so you can swap in larger models.

1. **GPTQModel** — Uses [GPTQModel](https://github.com/modelcloud/gptqmodel) with `QuantizeConfig` to quantize the **OPT-125M** model to 4-bit precision.
2. **GPTQModifier** — Uses [LLM Compressor](https://github.com/vllm-project/llm-compressor) with a GPTQ recipe to quantize the **OPT-125M** model to mixed precision W4A16.

---

## LUMI

To run gptq scripts on LUMI, you have to setup a Python environment using `optimum`, `gptqmodel`, and `llmcompressor`, built on the top of LUMI's AI Singularity framework. 

Load the `Singularity` container environment and set the contianer image path. Later, create virtual environment inside the contianer and install the packages. 

```bash
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif

singularity shell "$SIF"

Apptainer> python -m venv venv --system-site-packages
Apptainer> source venv/bin/activate
(venv) Apptainer> pip install gptqmodel==7.1.0 dataset --no-build-isolation --cache-dir ./.pip-cache
(venv) Apptainer> pip install llmcompressor==0.10.0 --cache-dir ./.pip-cache

```
The flag --cache-dir points the pip cache to the current (scratch) folder instead of the default (home directory), to avoid filling up home directory quota.

---
## Roihu

The CSC preinstalled PyTorch module covers most of the libraries needed to run these examples
(torch, transformers, datasets, accelerate). The rest can be installed on top of the module in a virtual environment.

### Load the module
```bash
module purge
module load python-pytorch/2.10
```
### Create and activate a virtual environment using system packages
```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
```
### Install packages
```bash
(venv)> pip install gptqmodel==7.1.0 dataset --no-build-isolation --cache-dir ./.pip-cache
```
This version of gptqmodel is compatible with python-pytorch/2.10.

For the **gptq-modifier** example, you need to install the llmcompressor library.

```bash
(venv)> pip install llmcompressor==0.12.0 --cache-dir ./.pip-cache
``` 

---

## Usage

The launch scripts for gptq-config are: 

- `run-gptq-config-lumi.sh` - quantizes model on LUMI with 1 GPU
- `run-gptq-config-roihu.sh` - quantizes model on Roihu with 1 GPU

Similarly, for gptq-modifier:

- `run-gptq-modifier-lumi.sh` - quantizes model on LUMI with 1 GPU
- `run-gptq-modifier-roihu.sh` - quantizes model on Roihu with 1 GPU

**Note:** the scripts are made to be run on `gputest` or `dev-g` partition with a 30 minute time-limit. You have to select the proper partition for longer jobs for your real runs. Additionally, change the `--account` parameter to your own project code. 

For example to run on LUMI, you would run the command:

```bash
sbatch run-gptq-config-lumi.sh
```
You can also increase the memory if you decide to run quantization on larger models. Setting `device_map="auto"` automatically offloads the model to a CPU to help fit the model in memory, and allow the model modules to be moved between the CPU and GPU for quantization.

## `gptq-config.py`
- Uses [`gptqmodel`](https://github.com/modelcloud/gptqmodel)(`GPTQModel.load`, `QuantizeConfig`) with `QuantizeConfig` recipe.
- Runs explicit **calibration** on a subset of the [aleenai/c4](https://huggingface.co/datasets/allenai/c4) dataset.
- This example quantizes the model to 4-bit precision, supported precisions are 2-bit, 3-bit*, 4-bit and 8-bit. 
- Saves both the full-precision and quantized models. 
- Compares outputs, inference latency, and model size.

## `gptq-modifier.py`
- Uses [LLM Compressor](https://github.com/vllm-project/llm-compressor) with a `GPTQModifier` recipe.
- Runs explicit **calibration** on a subset of the [Ultrachat-200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) dataset.
- This example quantizes the model to mixed precision W4A16 (INT4 weights and FP16 activations). Supported formats include W8A8 (INT8 and FP8), W4A16, W8A16, and NVFP4 (with W4A4 and W4A16 support).
- Saves both the full-precision and quantized models.
- Compares outputs, inference latency, and model size.
- Provides finer control over quantization schemes (e.g. `W4A16`, `ignore=["lm_head"]`).

*3-bit quantization is not currently supported on LUMI with PyTorch 2.7, if you wish to use 3-bit quantization you can use the PyTorch 2.5 module.

## Output Includes
- Generated text before and after quantization.
- Inference time comparison. 
- Model size (MB) before and after quantization.
- Note that the effect of quantization on inference might not be noticeable for smaller models.

## Notes
- The current scripts use **OPT-125M** for fast experimentation. You can replace `model_name` with a larger model. In this case, you might want to disable saving the models.
- For large models, `device_map="auto"` allows the model modules to be moved between the CPU and GPU for quantization.
- The `GPTQConfig` path is simpler and integrates directly with Hugging Face pipelines, while the `GPTQModifier` path gives you more flexibility for research and custom recipes.
- Feel free to experiment with different values for `num_calibration_samples` and `max_seq_lenght`.
