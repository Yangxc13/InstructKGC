import os
import sys

import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

device = 'cuda:0'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["WANDB_DISABLED"] = "true"


def gen_zhixi_model(
    # load_8bit: bool = False,
    base_model: str = 'model_hub/zhixi-13b',
    lora_weights: str = 'model_hub/zhixi-13b-lora',
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # if load_8bit:
    #     config = LoraConfig.from_pretrained(lora_weights)
    #     model = get_peft_model(model, config) 
    #     adapters_weights = torch.load(os.path.join(lora_weights, WEIGHTS_NAME), map_location=model.device)
    #     set_peft_model_state_dict(model, adapters_weights)
    if lora_weights:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"": device},
        )
    # elif device == "mps":
    #     model = LlamaForCausalLM.from_pretrained(
    #         base_model,
    #         device_map={"": device},
    #         torch_dtype=torch.float16,
    #     )
    #     model = PeftModel.from_pretrained(
    #         model,
    #         lora_weights,
    #         device_map={"": device},
    #         torch_dtype=torch.float16,
    #     )
    # else:
    #     model = LlamaForCausalLM.from_pretrained(
    #         base_model, device_map={"": device}, low_cpu_mem_usage=True
    #     )
    #     model = PeftModel.from_pretrained(
    #         model,
    #         lora_weights,
    #         device_map={"": device},
    #     )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = tokenizer.bos_token_id = 1
    model.config.eos_token_id = tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"

    # if not load_8bit:
    model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return model, tokenizer
