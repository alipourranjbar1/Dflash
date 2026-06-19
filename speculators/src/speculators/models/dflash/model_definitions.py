from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3RMSNorm,
    eager_attention_forward,
)
from typing_extensions import Unpack

if TYPE_CHECKING:
    from collections.abc import Callable


# Local copy of rotate_half to avoid dependency on internal transformers functions
def _rotate_half(x):
    """Rotates half the hidden dims of the input (local implementation)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _debug_kv_tensor(name: str, tensor: torch.Tensor | None) -> None:
    """Print None status and whether a K/V tensor is all-zero."""
    if tensor is None:
        print(f"[KV-DEBUG] {name}: is_none=True", flush=True)
        return
    with torch.no_grad():
        x = tensor.detach().float()
        max_abs = x.abs().max().item() if x.numel() else 0.0
        mean_abs = x.abs().mean().item() if x.numel() else 0.0
        zero_frac = (x == 0).float().mean().item() if x.numel() else 1.0
        print(
            f"[KV-DEBUG] {name}: is_none=False shape={tuple(tensor.shape)} "
            f"all_zero={max_abs == 0.0} max_abs={max_abs:.6g} "
            f"mean_abs={mean_abs:.6g} zero_frac={zero_frac:.4f}",
            flush=True,
        )


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids=None,  # noqa: ARG001
    unsqueeze_dim=1,
):
    """Apply rotary position embeddings (local implementation)."""

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (_rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Implements the custom attention which injects the target models
    # hidden states into the kv cache.
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,  # type: ignore[operator]
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads  # type: ignore[operator]
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_attention_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.k_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_key_value_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.v_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_key_value_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,  # type: ignore[operator]
            config.hidden_size,  # type: ignore[arg-type]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.sliding_window = (
            config.sliding_window
            if hasattr(config, "layer_types")
            and config.layer_types is not None
            and config.layer_types[layer_idx] == "sliding_attention"  # type: ignore[index]
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: torch.Tensor | None = None,
        target_k: torch.Tensor | None = None,  # [bsz, ctx_len, nkv*hd] post-norm+RoPE
        target_v: torch.Tensor | None = None,  # [bsz, ctx_len, nkv*hd] from target cache
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = hidden_states.shape[:-1]

        # Q always comes from the drafter's own noise tokens
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)

        # K/V noise always from the drafter's own tokens
        k_noise = self.k_proj(hidden_states)
        v_noise = self.v_proj(hidden_states)

        cos, sin = position_embeddings

        if self.layer_idx == 0:
            _debug_kv_tensor("target_k", target_k)
            _debug_kv_tensor("target_v", target_v)
            _debug_kv_tensor("target_hidden", target_hidden)

        if target_k is not None and target_v is not None:
            # Target K/V from the connector are post-k_norm + post-RoPE (paged cache).
            # Use context K/V as-is; apply draft k_norm + RoPE only to noise tokens.
            # Matches inference kv_mode="copy" (skip_norm_and_rope on context).
            ctx_len = target_k.shape[1]
            k_ctx = target_k.view(bsz, ctx_len, -1, self.head_dim).transpose(1, 2)
            v_ctx = target_v.view(bsz, ctx_len, -1, self.head_dim).transpose(1, 2)

            k_noise_view = k_noise.view(bsz, q_len, -1, self.head_dim)
            v_noise_view = v_noise.view(bsz, q_len, -1, self.head_dim)
            k_noise = self.k_norm(k_noise_view).transpose(1, 2)
            v_noise = v_noise_view.transpose(1, 2)

            cos_u = cos.unsqueeze(1)
            sin_u = sin.unsqueeze(1)
            q = (q * cos_u[..., -q_len:, :]) + (_rotate_half(q) * sin_u[..., -q_len:, :])
            k_noise = (k_noise * cos_u[..., ctx_len : ctx_len + q_len, :]) + (
                _rotate_half(k_noise) * sin_u[..., ctx_len : ctx_len + q_len, :]
            )

            k = torch.cat([k_ctx, k_noise], dim=2)
            v = torch.cat([v_ctx, v_noise], dim=2)
            if self.layer_idx == 0:
                _debug_kv_tensor("k_ctx", k_ctx)
                _debug_kv_tensor("v_ctx", v_ctx)
               
              
        else:
            # Original path: project context K/V from target hidden states, then
            # apply k_norm + RoPE to the full concatenated sequence.
            assert target_hidden is not None
            ctx_len = target_hidden.shape[1]
            _debug_kv_tensor("target_hidden", target_hidden)
            k_ctx = self.k_proj(target_hidden).view(bsz, ctx_len, -1, self.head_dim)
            v_ctx = self.v_proj(target_hidden).view(bsz, ctx_len, -1, self.head_dim)

            k = torch.cat([k_ctx, k_noise.view(bsz, q_len, -1, self.head_dim)], dim=1)
            v = torch.cat([v_ctx, v_noise.view(bsz, q_len, -1, self.head_dim)], dim=1)

            k = self.k_norm(k).transpose(1, 2)
            v = v.transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if (
            self.config._attn_implementation is not None  # noqa: SLF001
            and self.config._attn_implementation != "eager"  # noqa: SLF001
        ):
            attn_fn = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation  # noqa: SLF001
            ]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,  # type: ignore[arg-type]
        )

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        target_k: torch.Tensor | None = None,
        target_v: torch.Tensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        assert hidden_states is not None  # noqa: S101
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            target_k=target_k,
            target_v=target_v,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states  # type: ignore[operator]
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states  # type: ignore[operator,return-value]
