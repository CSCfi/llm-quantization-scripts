# AWQ Quantization with LLM Compressor

This example script shows how to quantize a model using **Activation-Aware Weight Quantization (AWQ)** with the [LLM Compressor](https://github.com/vllm-project/llm-compressor) toolkit. AWQ protects ~1% of the most important weight channels to reduce quantization error and improve performance when compared to uniform quantization.

In order to target weight and activation scaling locations within the model, the AWQModifier must be provided an AWQ mapping. The model used in the example already has these mappings provided, but in case the model you want to quantize does not, you need to add your own mappings via the mappings argument with instantiating the AWQModifier. You can check existing mappings (and contribute your own) [here.](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/awq/mappings.py)

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
pip install optimum llmcompressor --cache-dir ./.pip-cache
# tested with optimum==1.27.0 llmcompressor==0.7.1
```
The flag --cache-dir points the pip cache to the current (scratch) folder instead of the default (home directory), to avoid filling up home directory quota. 

## Usage

The launch scripts are: 

- `run-awq-modifier-lumi.sh` - quantizes model on LUMI with 1 GPU 
- `run-awq-modifier-mahti.sh` - quantizes model on Mahti with 1 GPU
- `run-awq-modifier-puhti.sh` - quantizes model on Puhti with 1 GPU

**Note:** the scripts are made to be run on `gputest` or `dev-g` partition with a 30 minutes time-limit. You have to select the proper partition for longer jobs for your real runs. Additionally, change the `--account` parameter to your own project code. 

For example to run on LUMI, you would run the command: 

```bash
sbatch run-awq-modifier-lumi.sh
```
You can also increase the memory and number of GPUs if you decide to run quantization on larger models. 

The script will quantize the **Falcon-RW-1B** model using the following recipe:
```bash
recipe = AWQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
```
Meaning the script quantizes the model’s linear layers using a mixed-precision approach: weights are reduced to 4-bit while activations remain at 16-bit to retain higher accuracy. The lm_head layer (the model’s output projection) is excluded to preserve output quality. You can check the output in the output file.

## Output Includes
- Generated text before and after quantization.
- Inference time comparison. Note that the effect of quantization on inference might not be noticeable for smaller models.
- Model size (MB) before and after quantization.

## Notes
- The current scripts use **Falcon-RW-1B** for fast experimentation. You can replace `model_name` with a larger model. In this case, you might want to disable saving the full model.
- For large models, `device_map="auto"` lets 🤗 Accelerate handle placement across GPUs.
- Feel free to experiment with different values for `num_calibration_samples` and `max_seq_lenght` and to modify the quantization recipe.
