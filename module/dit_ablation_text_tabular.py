

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class AdaLN(nn.Module):
    
    def __init__(self, hidden_size, cond_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Linear(cond_size, hidden_size * 2, bias=True)
        nn.init.constant_(self.mlp.weight, 0)
        nn.init.constant_(self.mlp.bias, 0)

    def forward(self, x, cond):
        shift, scale = self.mlp(cond).chunk(2, dim=-1)
        if cond.dim() != 2:
            raise RuntimeError(f"[AdaLN] cond must be 2D (B, D), got {cond.shape}.")
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class GatedCrossAttention(nn.Module):
    
    def __init__(self, hidden_size, num_heads, context_dim, cond_size=512, dropout=0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            kdim=context_dim, vdim=context_dim,
            dropout=dropout, batch_first=True
        )
        self.gate_proj = nn.Linear(cond_size, hidden_size, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, x, context, cond=None, mask=None, need_weights=False):
        key_padding_mask = (mask == 0) if mask is not None else None
        attn_out, attn_weights = self.cross_attn(
            query=x, key=context, value=context,
            key_padding_mask=key_padding_mask, need_weights=need_weights
        )
        if cond is not None:
            gate = self.gate_proj(cond).unsqueeze(1)  # [B, 1, D]
        else:
            gate = torch.zeros(1, device=x.device, dtype=x.dtype)
        return gate * attn_out, attn_weights


class TabularProjector(nn.Module):
    
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )

    def forward(self, age, gender, hr):
        age = age.view(-1)
        gender = gender.view(-1)
        hr = hr.view(-1)
        x = torch.stack([age, gender, hr], dim=-1)
        return self.net(x)



class TextTabularDiTBlock(nn.Module):
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cond_size=512,
                 text_embed_dim=768, dropout=0.0):
        super().__init__()
        self.adaln1 = AdaLN(hidden_size, cond_size)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        self.adaln2 = AdaLN(hidden_size, cond_size)
        self.cross_attn_text = GatedCrossAttention(
            hidden_size, num_heads, text_embed_dim, cond_size, dropout
        )

        self.adaln3 = AdaLN(hidden_size, cond_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, global_cond, text_embeds):
        h = self.adaln1(x, global_cond)
        attn_out, _ = self.self_attn(h, h, h)
        x = x + attn_out

        h = self.adaln2(x, global_cond)
        text_delta, _ = self.cross_attn_text(h, text_embeds, cond=global_cond)
        x = x + text_delta

        h = self.adaln3(x, global_cond)
        x = x + self.mlp(h)

        return x



class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_size, hidden_size * 2, bias=True)
        )

    def forward(self, x, cond):
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)



class DiT_TextTabular_ECG(nn.Module):
    
    def __init__(
        self,
        in_channels=4,
        seq_length=128,
        hidden_size=512,
        depth=12,
        num_heads=8,
        text_embed_dim=768,
        mlp_ratio=4.0,
        dropout=0.0,
        use_rope=False,
        cfg_dropout_joint=0.70,
        cfg_dropout_text=0.10,
        cfg_dropout_tab=0.10,
        cfg_dropout_uncond=0.10,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cfg_dropout_joint = cfg_dropout_joint
        self.cfg_dropout_text = cfg_dropout_text
        self.cfg_dropout_tab = cfg_dropout_tab
        self.cfg_dropout_uncond = cfg_dropout_uncond

        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.tabular_projector = TabularProjector(hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_length, hidden_size))

        self.register_buffer('null_text_embed',
                             torch.zeros(1, 1, text_embed_dim))
        self.register_buffer('null_tabular_embed',
                             torch.zeros(1, hidden_size))

        self.blocks = nn.ModuleList([
            TextTabularDiTBlock(
                hidden_size, num_heads, mlp_ratio, hidden_size,
                text_embed_dim, dropout
            )
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, in_channels, hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.pos_embed, std=0.02)

    def apply_cfg_masks(self, text_embeds, tab_vector):
        
        if not self.training:
            return text_embeds, tab_vector

        B = text_embeds.shape[0]
        rand = torch.rand(B, device=text_embeds.device)

        mask_text = torch.ones(B, dtype=torch.bool, device=text_embeds.device)
        mask_tab = torch.ones(B, dtype=torch.bool, device=text_embeds.device)

        p_uncond = self.cfg_dropout_uncond                                    # [0, 0.10)
        p_tab = p_uncond + self.cfg_dropout_tab                               # [0.10, 0.20)
        p_text = p_tab + self.cfg_dropout_text                                # [0.20, 0.30)

        m_un = rand < p_uncond
        mask_text[m_un] = False
        mask_tab[m_un] = False

        m_tab = (rand >= p_uncond) & (rand < p_tab)
        mask_text[m_tab] = False

        m_text = (rand >= p_tab) & (rand < p_text)
        mask_tab[m_text] = False


        text_embeds = text_embeds.clone()
        tab_vector = tab_vector.clone()

        if (~mask_text).any():
            text_embeds[~mask_text] = 0.0

        if (~mask_tab).any():
            tab_vector[~mask_tab] = 0.0

        return text_embeds, tab_vector

    def forward(self, x, t, text_embeds, age, gender, hr, return_dict=False,
                force_uncond_text=False, force_uncond_tab=False):
        
        B = x.size(0)

        tab_vector = self.tabular_projector(age, gender, hr)

        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(1)

        text_embeds, tab_vector = self.apply_cfg_masks(text_embeds, tab_vector)
        #xiufu
        if not self.training:
            uncond_mask = (age.view(-1) == 99999.0) | (hr.view(-1) == 99999.0)
            if uncond_mask.any():
                tab_vector[uncond_mask] = 0.0

        x = self.x_embedder(x.transpose(1, 2)) + self.pos_embed  # (B, L, D)

        global_cond = self.t_embedder(t) + tab_vector  # (B, D)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x, global_cond, text_embeds)

        x = self.final_layer(x, global_cond).transpose(1, 2)

        if return_dict:
            return {"noise_pred": x}
        return x
