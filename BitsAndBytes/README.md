# Bitsandbytes Quantization with Transformers

This example demonstrates quantizing the **OPT-125M** model using the [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) 4-bit quantization API in the 🤗 `transformers` library. Bitsandbytes uses runtime quantization, which compresses weights to 4-bit on-the-fly during inference to save memory, but these quantized weights cannot be saved to disk. This example quantizes to nf4, but bitsandbytes also supports standard floating point 4-bit and 8-bit quantization.

## Running the script

All of the libraries needed to run this example (transformers, bitsandbytes, accelerate) are covered by the CSC preinstalled PyTorch module.

Each script showing how to quantize a model on CSC's supercomputers. 

The script `bnb-quantization.py` will quantize the OPT-125M model to nf4 or NormalFloat 4-bit, introduced to use with QLoRA technique, a parameter efficient fine-tuning technique. It can be used with QLoRA for fine-tuning, or without just for reducing model size. 

The launch scripts are: 

- `run-bnb-quantization-lumi.sh` - quantizes model on LUMI with 1 GPU 
- `run-bnb-quantization-mahti.sh` - quantizes model on Mahti with 1 GPU
- `run-bnb-quantization-puhti.sh` - quantizes model on Puhti with 1 GPU

**Note:** the scripts are made to be run on `gputest` or `dev-g` partition with a 30 minutes time-limit. You have to select the proper partition for longer jobs for your real runs. Additionally, change the `--account` parameter to your own project code. 

For example to run on LUMI, you would run the command: 

```bash
sbatch run-bnb-quantization-lumi.sh
```
You can also increase the memory and number of GPUs if you decide to run quantization on larger models.

## Output Includes

- Generated text before and after quantization.
- Inference time comparison. Note that the effect of quantization on inference might not be noticeable for smaller models.

## Notes

- The current scripts use OPT-125M for fast experimentation. You can replace `model_name` with a larger model.
- For large models, `device_map="auto"` lets 🤗 Accelerate handle placement across GPUs.
- You can easily change the quantization datatype in the `BitsAndBytesConfig` inside the script.
