import torch
import torch.nn.functional as F
# import torch.nn as nn
from torch import nn, Tensor
from transformers import AutoConfig
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .sampling import cosine_schedule, mask_by_random_topk
# from .phi import PhiForCausalLM
from .modeling_showo import Showo
import math

class RGenieModel(ModelMixin, ConfigMixin):

    # @register_to_config
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.register_to_config(mask_token_id=config.model.showo.vocab_size - 1)
        self.model = Showo.from_pretrained(config.model.showo.pretrained_model_path)
        self.llm = self.model.showo

        self.proj1 = nn.Linear(self.model.output_size, self.model.output_size, bias=False)
        self.proj2 = nn.Linear(1024, self.model.output_size, bias=False)
        num_heads = 1
        self.self_attn = Attention(self.model.output_size, num_heads)
        # self.norm = nn.LayerNorm(self.model.output_size)
    
    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True
    
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
        # Aquire the expected visual tokens after modifications
        hidden_states = self.llm(input_ids=input_ids, attention_mask=attention_mask)['logits'] # logits: torch.Size([B, 1155, 58498]) image_tokens: torch.Size([B, 1024])
        #### Reasoning-Attention Bridge
        h_edit = self.proj1(hidden_states.to(torch.bfloat16)) # h_edit: torch.Size([B, 1155, 58498])
        V = image_tokens
        visual_feature = V.unsqueeze(1).repeat(1, hidden_states.shape[1], 1).to(torch.bfloat16)
        visual_feature = self.proj2(visual_feature) # visual_feature: torch.Size([B, 1155, 58498])
        h_edit = self.self_attn(h_edit, visual_feature, visual_feature)
        # h_edit = self.norm(h_edit)
        h_edit = h_edit + hidden_states

        #### Hierarchical Reasoning Module
        h_reason = nn.AdaptiveAvgPool2d((1155, 1))(visual_feature) # h_reason.shape: torch.Size([B, 1155, 1])
        V_global = nn.AdaptiveAvgPool2d((1155, 1))(visual_feature) # V_global.shape: torch.Size([B, 1155, 1])
        h_reason = self.self_attn(V_global, h_reason, h_reason)
        # h_reason = self.norm(h_reason)
        h_reason = h_reason + hidden_states # torch.Size([B, 1155, 58498])

        hidden_states = h_edit + h_reason

        logits = hidden_states

        if labels is not None:
            # 1. Mask token prediction (discrete diffusion) for image generation
            # Note that, max_seq_length indicates the maximum number of text tokens, maybe a bit confused.
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.model.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            )
        
        if labels is not None:
            # 1. Mask token prediction (discrete diffusion) for image generation
            # Note that, max_seq_length indicates the maximum number of text tokens, maybe a bit confused.
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.model.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            )

            # # 2. Next token prediction for language modeling
            # loss_lm = F.cross_entropy(
            #     logits[batch_size_t2i:batch_size_t2i + batch_size_lm, :-1].contiguous().view(-1, self.output_size),
            #     labels[batch_size_t2i:batch_size_t2i + batch_size_lm, 1:].contiguous().view(-1), ignore_index=-100,
            # )

            # # 3. Next token prediction for captioning/multimodal understanding
            # loss_mmu = F.cross_entropy(
            #     logits[-batch_size_mmu:, :-1].contiguous().view(-1, self.output_size),
            #     labels[-batch_size_mmu:, 1:].contiguous().view(-1), ignore_index=-100,
            # )

            # return logits, loss_t2i, loss_lm, loss_mmu
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
        mask_token_id = self.config.mask_token_id
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
        except:
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
            if self.config.w_clip_vit:
                idx_next_embeddings = self.showo.model.embed_tokens(idx_next)
                input_embeddings = torch.cat([input_embeddings, idx_next_embeddings], dim=1)
            else:
                idx = torch.cat((idx, idx_next), dim=1)

            if eot_token is not None and idx_next.cpu() == eot_token:
                break

        return result


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (self.internal_dim % num_heads == 0), "num_heads must divide embedding_dim."

        # self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        # self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        # self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        # self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        # q = self.q_proj(q)
        # k = self.k_proj(k)
        # v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        # out = self.out_proj(out)

        return out

