import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
# sys.path.append('/root/miniconda3/envs/RIE/lib/python3.10/site-packages/')
# print(sys.executable)
# print("Current Working Directory:", os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Showo, MAGVITv2
from training.utils import get_config
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from training.prompting_utils import UniversalPrompting
# from llava.llava import conversation as conversation_lib
from peft import LoraConfig, get_peft_model


# process images into discrete tokens
def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

def main():
    config = get_config()
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"config.model.showo.llm_model_path is {config.model.showo.llm_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")
    
    # VQ model for processing image into discrete tokens
    vq_model = get_vq_model_class(config.model.vq_model.type)
    if config.model.vq_model.get("pretrained_model_path", None):
        vq_model = vq_model()
        state_dict = torch.load(config.model.vq_model.pretrained_model_path)['model']
        vq_model.load_state_dict(state_dict)
    else:
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name)
    vq_model.eval()
    vq_model.requires_grad_(False)

    model = Showo.from_pretrained(config.model.showo.pretrained_model_path)

    # model.eval()
    print(model)

    lora_r = config.training.lora_r

    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            # print(f"In function: lora_target_modules are {lora_target_modules}")
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                # print(f"name={name}, module={module}")
                if (
                    isinstance(module, cls) and
                    any(x in name for x in lora_target_modules)  # Include only specific layers
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))


        lora_alpha = config.training.lora_alpha
        lora_dropout = config.training.lora_dropout
        lora_target_modules = find_linear_layers(
            model, config.training.lora_target_modules.split(",")
        )
        # print(f"lora_target_modules are {lora_target_modules}")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            # task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        for name, param in model.named_parameters():
            # print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
            if 'lora' not in name:
                param.requires_grad = False
        model.print_trainable_parameters()

        state_dict = torch.load(config.merge_lora_weight.weight, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

        model = model.merge_and_unload()
        state_dict = {}
        for k, v in model.state_dict().items():
            state_dict[k] = v
        
        model.save_pretrained(config.merge_lora_weight.save_path, state_dict=state_dict)
        # tokenizer.save_pretrained(config.merge_lora_weight.save_path)
        print("Successfully saved!")

if __name__ == "__main__":
    main()       