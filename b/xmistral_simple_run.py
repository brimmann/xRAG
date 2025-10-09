import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig
)

from src.model import XMistralForCausalLM

model_name = "Hannibal046/xrag-7b"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side = 'left',
    add_eos_token=False, ## import to include this!
    use_fast=False,
)

config = AutoConfig.from_pretrained(model_name)

 ## load llm
config = AutoConfig.from_pretrained(model_name)
MODEL_CLASS = eval(config.architectures[0])
model = MODEL_CLASS.from_pretrained(
    model_name,
    torch_dtype = torch.bfloat16,
    low_cpu_mem_usage = True,
    device_map='auto',
    offload_folder="./offload"
)

