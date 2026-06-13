import torch
from torch import nn


class ReasoningVisualModulator(nn.Module):
    """Implements the HRM and RAB stage described in the R-Genie paper."""

    def __init__(self, hidden_size, num_heads=8, num_hrm_layers=2):
        super().__init__()
        if hidden_size % num_heads != 0:
            num_heads = 1

        self.visual_proj = nn.Linear(hidden_size, hidden_size)
        self.rab_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.hrm_attn = nn.ModuleList(
            [nn.MultiheadAttention(hidden_size, num_heads, batch_first=True) for _ in range(num_hrm_layers)]
        )
        self.hrm_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_hrm_layers)])
        self.rab_norm = nn.LayerNorm(hidden_size)
        self.condition_norm = nn.LayerNorm(hidden_size)
        self.condition_gate = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden_states, visual_token_embeddings, max_text_length):
        batch_size, seq_len, hidden_size = hidden_states.shape
        visual_features = self.visual_proj(visual_token_embeddings.to(hidden_states.dtype))

        text_end = min(max_text_length + 1, seq_len)
        visual_start = min(text_end + 1, seq_len)
        visual_len = min(visual_features.shape[1], max(seq_len - visual_start - 1, 0))

        h_edit = hidden_states[:, :text_end, :]
        h_reason = h_edit.mean(dim=1, keepdim=True)
        i_global = visual_features.mean(dim=1, keepdim=True)

        for attn, norm in zip(self.hrm_attn, self.hrm_norm):
            reason_delta, _ = attn(query=h_reason, key=i_global, value=i_global, need_weights=False)
            h_reason = norm(h_reason + reason_delta)

        local_delta, _ = self.rab_attn(query=h_edit, key=visual_features, value=visual_features, need_weights=False)
        h_edit = self.rab_norm(h_edit + local_delta)

        local_condition = torch.zeros(
            batch_size,
            seq_len,
            hidden_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        local_condition[:, :text_end, :] = h_edit
        if visual_len > 0:
            local_condition[:, visual_start:visual_start + visual_len, :] = visual_features[:, :visual_len, :]

        global_condition = h_reason.expand(-1, seq_len, -1)
        condition = self.condition_norm(global_condition + local_condition)
        gate = torch.sigmoid(self.condition_gate(torch.cat([hidden_states, condition], dim=-1)))
        return hidden_states + gate * condition

