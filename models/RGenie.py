import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from .modeling_utils import ConfigMixin, ModelMixin
from .sampling import cosine_schedule, mask_by_random_topk
from .modeling_showo import Showo
from .rgenie_components import ReasoningVisualModulator
import os


def _resolve_rgenie_config(config, **kwargs):
    if config is None:
        config = kwargs.get("config")
    if isinstance(config, str):
        return OmegaConf.load(config)
    if isinstance(config, dict) and "model" not in config and "config" in config:
        return OmegaConf.load(config["config"])
    if hasattr(config, "config") and not hasattr(config, "model"):
        return OmegaConf.load(config.config)
    return config


def _has_model_weights(path):
    if not path or not os.path.isdir(path):
        return False
    weight_names = (
        "pytorch_model.bin",
        "diffusion_pytorch_model.bin",
        "model.safetensors",
        "diffusion_pytorch_model.safetensors",
    )
    return any(os.path.exists(os.path.join(path, name)) for name in weight_names)


class RGenieModel(ModelMixin, ConfigMixin):

    def __init__(self, config=None, **kwargs):
        super().__init__()
        config = _resolve_rgenie_config(config, **kwargs)
        if config is None or not hasattr(config, "model"):
            raise ValueError("RGenieModel requires a training yaml config or a config object with a model section.")
        self.rgenie_config = config

        if _has_model_weights(config.model.showo.pretrained_model_path):
            self.model = Showo.from_pretrained(config.model.showo.pretrained_model_path)
        else:
            self.model = Showo(**config.model.showo)
        self.llm = self.model.showo
        self.output_size = self.model.output_size
        self.vocab_size = self.model.vocab_size
        self.mask_token_id = self.model.config.mask_token_id
        self.register_to_config(mask_token_id=self.mask_token_id)

        hidden_size = self.llm.config.hidden_size
        self.reasoning_modulator = ReasoningVisualModulator(
            hidden_size=hidden_size,
            num_heads=config.model.RGenie.get("num_attention_heads", 8),
            num_hrm_layers=config.model.RGenie.get("num_hrm_layers", 2),
        )
    
    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def _build_reasoning_condition(self, hidden_states, image_tokens, max_seq_length):
        if image_tokens is None:
            return hidden_states

        image_tokens = image_tokens.clamp(min=0, max=self.vocab_size - 1)
        visual_embeddings = self.llm.get_input_embeddings()(image_tokens)
        return self.reasoning_modulator(hidden_states, visual_embeddings, max_seq_length)
    
    def forward(
        self,
        input_ids,
        image_tokens=None,
        input_embeddings=None,
        attention_mask=None,
        labels=None,
        label_smoothing=0.0,
        batch_size_t2i=0,
        batch_size_lm=0,
        batch_size_mmu=0,
        max_seq_length=128,
        labels_mask_text=None,
        labels_mask_image=None,
        **kwargs,
    ):
        llm_kwargs = {
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if input_embeddings is None:
            llm_kwargs["input_ids"] = input_ids
        else:
            llm_kwargs["inputs_embeds"] = input_embeddings
        outputs = self.llm(**llm_kwargs)
        hidden_states = outputs.hidden_states[-1]
        hidden_states = self._build_reasoning_condition(hidden_states, image_tokens, max_seq_length)
        logits = self.llm.lm_head(hidden_states).float()
        
        if labels is not None:
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            )
            return logits, loss_t2i

        return logits
    
    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """
        # begin with all image token ids masked
        mask_token_id = self.mask_token_id
        num_vq_tokens = config.model.showo.num_vq_tokens
        num_new_special_tokens = config.model.showo.num_new_special_tokens

        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                                    mask_token_id,
                                                    input_ids_minus_lm_vocab_size - config.model.showo.llm_vocab_size - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                cond_logits, uncond_logits = self(model_input, attention_mask=attention_mask).chunk(2)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]
            else:
                logits = self(input_ids, attention_mask=attention_mask)
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]

            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))

            # Add by hlf
            sampled = torch.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
            sampled = torch.clamp(sampled, min=0.0)
            sampled_sum = sampled.sum(dim=-1, keepdim=True)
            sampled = sampled / (sampled_sum + 1e-8)

            # Handle all-zero rows by replacing with uniform distribution
            mask = (sampled_sum.squeeze(-1) == 0)
            if mask.any():
                uniform_dist = torch.ones_like(sampled) / sampled.size(-1)
                sampled[mask] = uniform_dist[mask]

            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + config.model.showo.llm_vocab_size
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids

    @torch.no_grad()
    def mmu_generate(self, idx=None, input_embeddings=None, attention_mask=None, max_new_tokens=100, temperature=1.0, top_k=None, eot_token=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        try:
            device = idx.device
        except AttributeError:
            device = input_embeddings.device

        result = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            # logits, _ = self(idx_cond)
            logits = self(idx, input_embeddings=input_embeddings, attention_mask=attention_mask)

            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze()
            attention_mask_a = torch.hstack(
                [
                    attention_mask,  # L, L
                    torch.zeros((L, 1)).to(device) + torch.finfo(logits.dtype).min,
                ]
            )
            attention_mask_b = torch.vstack(
                [
                    attention_mask_a,  # L, L+1
                    torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
                ]
            )
            attention_mask = attention_mask_b

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            result.append(idx_next[0][0])
            # append sampled index to the running sequence and continue
            if getattr(self.model, "w_clip_vit", getattr(self.model.config, "w_clip_vit", False)):
                idx_next_embeddings = self.llm.model.embed_tokens(idx_next)
                input_embeddings = torch.cat([input_embeddings, idx_next_embeddings], dim=1)
            else:
                idx = torch.cat((idx, idx_next), dim=1)

            if eot_token is not None and idx_next.cpu() == eot_token:
                break

        return result
