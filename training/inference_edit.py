import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

from models import Showo, MAGVITv2, get_mask_chedule, RGenieModel
from training.edit_dataset import EditingDataset
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")
    
if __name__ == '__main__':

    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)
    
    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    # model = Showo.from_pretrained(config.model.showo.model_path_for_inference).to(device)
    model = RGenieModel.from_pretrained(config.model.RGenie.model_path_for_inference).to(device)
    model.eval()
    mask_token_id = model.config.mask_token_id

    test_dataset = EditingDataset(data_path=config.testing.test_data_path, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    for batch_idx, (img_name, img_id, instruction, image, target) in enumerate(tqdm(test_dataloader, desc="Testing")):
        prompt = instruction
        # image = Image.open(config.image_path).convert("RGB")
        images = image.to(device)
        image_tokens = vq_model.get_code(images) + len(uni_prompting.text_tokenizer)
        input_ids, _ = uni_prompting((prompt, image_tokens), 't2i_gen')

        if config.training.guidance_scale > 0:
            uncond_input_ids, _ = uni_prompting(([''] * len(prompt), image_tokens), 't2i_gen')
            attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True)
        else:
            attention_mask = create_attention_mask_predict_next(input_ids,
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True)
            uncond_input_ids = None
        
        if config.get("mask_schedule", None) is not None:
            schedule = config.mask_schedule.schedule
            args = config.mask_schedule.get("params", {})
            mask_schedule = get_mask_chedule(schedule, **args)
        else:
            mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

        with torch.no_grad():
            gen_token_ids = model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                guidance_scale=config.training.guidance_scale,
                temperature=config.training.get("generation_temperature", 1.0),
                timesteps=config.training.generation_timesteps,
                noise_schedule=mask_schedule,
                noise_type=config.training.get("noise_type", "mask"),
                seq_len=config.model.showo.num_vq_tokens,
                uni_prompting=uni_prompting,
                config=config,
            )
        gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
        
        images = vq_model.decode_code(gen_token_ids)
        # print(images.size())

        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        # print(images.shape)
        # print(images[0].shape)
        # print(type(images))
        image = Image.fromarray(images[0]) 
        # print(image.size)
        print(f"Saving to {os.path.join(config.testing.save_dir, img_name[0])}")
        image.save(os.path.join(config.testing.save_dir, img_name[0]))
    # logits = model(input_ids=input_ids, attention_mask=attention_mask, batch_size_t2i=config.training.batch_size)
    # print(f"model output size is {logits.size()}")
    # logits = logits[0]

    # logits.permute(1, 2, 0)
    # logits_min, logits_max = logits.min(), logits.max()
    # normalized = (logits - logits_min) / (logits_max - logits_min)  # 归一化到 [0, 1]
    # image_array = (normalized * 255).byte().numpy()  # 转为 uint8 类型

    # image = Image.fromarray(image_array)
    # image.save("edited_image.png")