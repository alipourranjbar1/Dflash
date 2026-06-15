# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.utils import (
    get_dflash_causal,
    load_dflash_model,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id

logger = init_logger(__name__)


class _KVMode(str, Enum):
    """Controls how DFlash populates the draft model's context KV cache.

    Select via ``dflash_config.kv_mode`` in the drafter's ``config.json``.
    ``hidden_states`` is the default and requires no extra configuration.
    All other modes also require ``dflash_config.kv_target_layer_ids``.

    Modes
    -----
    hidden_states  — (default) Target hidden states are projected through the
                     draft's own k/v projection weights, then the **draft's**
                     k-norm and RoPE are applied before writing to the draft
                     KV cache.  No hooks on the target model.

    raw_copy       — Hook on ``self_attn.qkv_proj`` captures K and V
                     **before** k-norm and RoPE.  The **draft's own** k-norm
                     and RoPE are then applied before writing to the draft KV
                     cache.  Useful when the draft has different norm/RoPE
                     parameters than the target.

    copy           — Hook on ``self_attn.attn`` captures K and V **after**
                     k-norm and RoPE — the exact values the target writes to
                     its paged cache.  Written directly to the draft KV cache
                     with no further transformation.  The draft cache is a
                     separate GPU allocation.

    alias          — Same hook as ``copy``, but the draft attention layers'
                     ``kv_cache`` tensors are **aliased** to the corresponding
                     target layers' ``kv_cache`` tensors after ``set_attn``.
                     The precompute write step is skipped entirely: the target
                     already wrote the right K/V into the shared tensor.  The
                     draft reads context K/V from the target's physical pages
                     and writes speculative K/V to its own allocated blocks.
                     Prerequisites: same num_kv_heads and head_dim; target's
                     block manager must pre-allocate speculative slots.
    """

    HIDDEN_STATES = "hidden_states"
    RAW_COPY = "raw_copy"
    COPY = "copy"
    ALIAS = "alias"


class _KVCaptureHooks:
    """Captures K/V tensors from target attention layers via PyTorch forward hooks.

    Supports two hook placements controlled by ``raw``:

    * ``raw=True``  — hook on ``self_attn.qkv_proj``: captures K and V
      **before** k-norm and RoPE (raw projection outputs).  Used by
      ``raw_copy`` mode so the draft can apply its own k-norm + RoPE.

    * ``raw=False`` (default) — hook on ``self_attn.attn``: captures K and V
      **after** k-norm and RoPE — identical to what the target stores in its
      paged KV cache.  Used by ``copy`` and ``alias`` modes.

    Usage::

        hooks = _KVCaptureHooks.register(
            target_model, kv_layer_ids, draft_num_layers, raw=False
        )
        # ... target model forward runs, hooks fire automatically ...
        kv_list = hooks.pop()  # list[(k, v)] per draft layer, or None
    """

    def __init__(self) -> None:
        self._captured: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    # ------------------------------------------------------------------
    @staticmethod
    def _make_post_norm_hook(buf: "_KVCaptureHooks"):
        """Hook for self_attn.attn — inputs are post-k_norm + post-RoPE."""
        def _hook(module, inp, output):
            # inp = (query, key, value, ...) fed into Attention.forward
            k = inp[1].detach()  # [num_tokens, kv_size]
            v = inp[2].detach()  # [num_tokens, kv_size]
            buf._captured.append((k, v))
        return _hook

    @staticmethod
    def _make_pre_norm_hook(buf: "_KVCaptureHooks", q_size: int, kv_size: int):
        """Hook for self_attn.qkv_proj — output is raw QKV (pre-k_norm, pre-RoPE)."""
        def _hook(module, inp, output):
            qkv = output[0] if isinstance(output, tuple) else output
            # qkv: [num_tokens, q_size + 2 * kv_size]
            k = qkv[..., q_size : q_size + kv_size].detach()
            v = qkv[..., q_size + kv_size :].detach()
            buf._captured.append((k, v))
        return _hook

    # ------------------------------------------------------------------
    @classmethod
    def register(
        cls,
        target_model: nn.Module,
        target_layer_ids: list[int],
        draft_num_layers: int,
        raw: bool = False,
    ) -> "_KVCaptureHooks":
        """Register hooks and return the capture buffer.

        Parameters
        ----------
        raw:
            ``True``  → hook ``self_attn.qkv_proj`` (pre-k_norm, pre-RoPE).
            ``False`` → hook ``self_attn.attn``     (post-k_norm, post-RoPE).
        """
        buf = cls()

        if len(target_layer_ids) != draft_num_layers:
            logger.warning(
                "DFlash KV capture: kv_target_layer_ids length (%d) != "
                "draft num_hidden_layers (%d). Falling back to hidden_states.",
                len(target_layer_ids),
                draft_num_layers,
            )
            return buf

        target_lm = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        layers = getattr(getattr(target_lm, "model", target_lm), "layers", None)
        if layers is None:
            logger.warning(
                "DFlash KV capture: cannot find target model layers; "
                "falling back to hidden_states."
            )
            return buf

        placement = "qkv_proj (pre-norm/RoPE)" if raw else "attn (post-norm/RoPE)"
        for lid in target_layer_ids:
            if lid >= len(layers):
                logger.warning(
                    "DFlash KV capture: layer %d out of range (%d layers); skipping.",
                    lid, len(layers),
                )
                continue
            attn_mod = layers[lid].self_attn
            if raw:
                handle = attn_mod.qkv_proj.register_forward_hook(
                    cls._make_pre_norm_hook(buf, attn_mod.q_size, attn_mod.kv_size)
                )
            else:
                handle = attn_mod.attn.register_forward_hook(
                    cls._make_post_norm_hook(buf)
                )
            buf._hooks.append(handle)

        logger.info(
            "DFlash KV capture: hooked target layers %s via %s.",
            target_layer_ids, placement,
        )
        return buf

    # ------------------------------------------------------------------
    def pop(self) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        """Return and clear the accumulated (k, v) list, or ``None`` if empty."""
        if not self._captured:
            return None
        result = self._captured
        self._captured = []
        return result

    # ------------------------------------------------------------------
    def remove(self) -> None:
        """Unregister all hooks (call on teardown)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    def __bool__(self) -> bool:
        """True when hooks are actually registered."""
        return bool(self._hooks)


class DFlashSpeculator(DraftModelSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )

        # Multimodal inputs not currently supported.
        self.supports_mm_inputs = False

        # Each request emits exactly (bonus + N mask) query tokens per step.
        self.num_query_per_req = 1 + self.num_speculative_steps

        self.parallel_drafting_token_id = get_parallel_drafting_token_id(
            self.draft_model_config.hf_config
        )

        self.dflash_causal = get_dflash_causal(self.draft_model_config)

        # Buffers for context K/V precomputation. Populated by prepare_dflash_inputs,
        # and processed by the model's precompute_and_store_context_kv method.
        # NOT captured by CUDA graphs.
        self.context_positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self.context_slot_mapping = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        # Per-mask-token sampling buffers. Flattened from (num_reqs, num_spec_tokens).
        max_num_sampled_tokens = self.max_num_reqs * self.num_speculative_steps
        self.sample_indices = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_pos = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_idx_mapping = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int32, device=device
        )
        # [0, 1, ..., N-1, 0, 1, ..., N-1, ...] -> the per-token column index into
        # draft_logits[req, step, :].
        self.sample_col = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        ).repeat(self.max_num_reqs)

        self.query_cudagraph_manager: DFlashCudaGraphManager | None = None
        self.draft_kv_cache_group_id: int = -1

        # KV sharing mode and capture hooks.
        # Populated by load_draft_model from dflash_config.
        self._kv_mode: _KVMode = _KVMode.HIDDEN_STATES
        self._kv_target_layer_ids: list[int] = []
        self._kv_capture: _KVCaptureHooks | None = None
        # Reference to target model kept for alias-mode post-set_attn wiring.
        self._target_model_ref: nn.Module | None = None

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # PIECEWISE cudagraphs are not supported for dflash
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        self.query_cudagraph_manager = DFlashCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.num_query_per_req,
            causal=self.dflash_causal,
        )

    def capture(self, attn_states: dict | None = None) -> None:
        logger.info("Capturing model for DFlash speculator...")
        # Reset sampling indices to zero to prevent stale values from prior
        # dummy runs from being baked into the captured graph.
        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.zero_()
        assert self.query_cudagraph_manager is not None
        self.query_cudagraph_manager.capture(
            self._generate_draft,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            progress_bar_desc="Capturing dflash CUDA graphs",
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        dflash_model = load_dflash_model(target_model, self.vllm_config)

        drafter_cfg = (
            getattr(self.draft_model_config.hf_config, "dflash_config", None) or {}
        )
        kv_layer_ids: list[int] | None = drafter_cfg.get("kv_target_layer_ids", None)
        raw_mode: str = drafter_cfg.get("kv_mode", _KVMode.HIDDEN_STATES.value)
        try:
            self._kv_mode = _KVMode(raw_mode)
        except ValueError:
            logger.warning(
                "DFlash: unknown kv_mode %r — falling back to 'hidden_states'.", raw_mode
            )
            self._kv_mode = _KVMode.HIDDEN_STATES

        if self._kv_mode == _KVMode.HIDDEN_STATES:
            logger.info("DFlash KV mode: hidden_states (original projection path).")
            return dflash_model

        # All other modes require kv_target_layer_ids.
        draft_num_layers = self.draft_model_config.hf_config.num_hidden_layers
        if kv_layer_ids is None:
            logger.warning(
                "DFlash kv_mode=%r requires kv_target_layer_ids in dflash_config; "
                "falling back to 'hidden_states'.",
                self._kv_mode.value,
            )
            self._kv_mode = _KVMode.HIDDEN_STATES
            return dflash_model

        self._kv_target_layer_ids = kv_layer_ids

        # raw_copy hooks qkv_proj (pre-norm, pre-RoPE).
        # copy and alias hook self_attn.attn (post-norm, post-RoPE).
        use_raw = self._kv_mode == _KVMode.RAW_COPY
        self._kv_capture = _KVCaptureHooks.register(
            target_model, kv_layer_ids, draft_num_layers, raw=use_raw
        )

        if self._kv_mode == _KVMode.ALIAS:
            self._target_model_ref = target_model
            logger.info(
                "DFlash KV mode: alias — will alias draft kv_cache tensors "
                "to target layers %s after set_attn.",
                kv_layer_ids,
            )
        else:
            logger.info(
                "DFlash KV mode: %s — K/V from target layers %s.",
                self._kv_mode.value, kv_layer_ids,
            )

        return dflash_model

    def _alias_kv_caches(self) -> None:
        """Alias each draft attention layer's kv_cache to the corresponding
        target layer's kv_cache (Mode ALIAS only).

        After this call the two models share the same physical KV cache
        memory for the mapped layers.  The draft attention will read context
        K/V from the target's pages (at the target's slot positions) and
        write speculative K/V to its own separately allocated blocks within
        the same physical tensor.

        Must be called after set_attn so that kv_cache tensors are allocated.
        """
        target = self._target_model_ref
        if target is None or not self._kv_target_layer_ids:
            return
        target_lm = (
            target.get_language_model() if hasattr(target, "get_language_model") else target
        )
        target_layers = getattr(getattr(target_lm, "model", target_lm), "layers", None)
        draft_layers = getattr(
            getattr(self.model, "model", self.model), "layers", None
        )
        if target_layers is None or draft_layers is None:
            logger.warning("DFlash alias: could not locate layer lists; skipping alias.")
            return

        aliased = 0
        for draft_idx, target_idx in enumerate(self._kv_target_layer_ids):
            if draft_idx >= len(draft_layers) or target_idx >= len(target_layers):
                continue
            d_attn = draft_layers[draft_idx].self_attn.attn
            t_attn = target_layers[target_idx].self_attn.attn
            if not hasattr(d_attn, "kv_cache") or not hasattr(t_attn, "kv_cache"):
                continue
            d_attn.kv_cache = t_attn.kv_cache
            aliased += 1

        logger.info(
            "DFlash KV mode: alias — aliased %d draft layer(s) to target kv_cache. "
            "Draft attention now reads context K/V directly from target pages.",
            aliased,
        )
        # Drop the reference; no longer needed.
        self._target_model_ref = None

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        super().set_attn(model_state, kv_cache_config, block_tables)

        # DFlash precomputes context K/V with a single block_size; mixing
        # kv-cache groups would silently corrupt the cache for the non-matching group.
        draft_groups = [gid for gid, g in enumerate(self.attn_groups) if g]
        assert len(draft_groups) == 1, (
            "DFlash currently requires all draft attention layers to share "
            "a single kv-cache group."
        )
        self.draft_kv_cache_group_id = draft_groups[0]
        self.draft_block_size = self.block_tables.block_sizes[
            self.draft_kv_cache_group_id
        ]

        # Alias mode: wire up shared kv_cache tensors now that they exist.
        if self._kv_mode == _KVMode.ALIAS:
            self._alias_kv_caches()

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            last_hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
            )
        return last_hidden_states

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )

        num_sample = num_reqs * self.num_speculative_steps
        sample_hidden_states = last_hidden_states[self.sample_indices[:num_sample]]
        draft_tokens = self.sample_draft(
            sample_hidden_states,
            self.sample_pos[:num_sample],
            self.sample_idx_mapping[:num_sample],
            self.temperature,
            self.seeds,
            self.sample_col[:num_sample],
            self.draft_logits,
        )
        self.draft_tokens[:num_reqs] = draft_tokens.view(
            num_reqs, self.num_speculative_steps
        )

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        num_query_per_req: int | None = None,
        causal: bool = False,
    ) -> dict[str, Any] | None:
        if not self.draft_attn_layer_names:
            return None
        assert num_query_per_req is None  # Omitted for DFlash, read from self instead
        return super()._build_draft_attn_metadata(
            num_reqs,
            num_reqs_padded,
            num_tokens_padded,
            num_query_per_req=self.num_query_per_req,
            causal=causal,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        # Optional: per-draft-layer K/V from the target model.
        # Each element is (k, v) with shape [num_tokens, num_kv_heads, head_dim].
        # When provided (or when self._kv_capture fires), context K/V are taken
        # directly from the target instead of being re-projected from hidden states.
        aux_kv_states: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_query_per_req, self.max_model_len
        )

        # NOTE: To avoid CPU-GPU synchronization without CPU knowing the
        # number of rejected tokens, we maintain the size of input_ids and
        # hidden_states the same as the target model's. This means, we pad each
        # request's query length to include any rejected positions.
        if aux_hidden_states:
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            hidden_states = last_hidden_states
        self.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])

        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        if dummy_run and skip_attn_for_dummy_run:
            # Memory profiling path: block_tables / kv_cache_config are not initialized.
            # Since DFlash needs to build its own attention metadata, we must skip the
            # preparation in this path and run a minimal forward pass.
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:num_target_tokens],
                self.context_positions[:num_target_tokens],
            )
            self._generate_draft(
                num_reqs,
                num_query_tokens,
                attn_metadata=None,
                slot_mappings=None,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            return self.draft_tokens[:num_reqs]

        # The query slot mapping is written into the shared BlockTables slot_mappings.
        # That buffer's address is what the captured CUDA graph reads from at replay.
        assert self.draft_kv_cache_group_id >= 0
        query_slot_mapping = self.block_tables.slot_mappings[
            self.draft_kv_cache_group_id
        ]
        prepare_dflash_inputs(
            self.input_buffers,
            query_slot_mapping,
            self.context_positions,
            self.context_slot_mapping,
            self.sample_indices,
            self.sample_pos,
            self.sample_idx_mapping,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.block_tables.input_block_tables[self.draft_kv_cache_group_id],
            self.draft_block_size,
            self.parallel_drafting_token_id,
            self.num_query_per_req,
            self.num_speculative_steps,
            self.max_num_reqs,
            self.max_num_tokens,
        )

        # Pre-insert context K/V into the draft cache.
        # Runs eagerly outside the captured graph because context shape varies.
        # During dummy runs block tables are placeholders — skip the cache write.
        _ctx_slot_mapping = (
            None if dummy_run else self.context_slot_mapping[:num_target_tokens]
        )

        # --- Context K/V population — mode-driven ---
        #
        # HIDDEN_STATES : project from target hidden states via draft k/v weights,
        #                 then apply draft k-norm + RoPE.
        # RAW_COPY      : hook captured pre-norm, pre-RoPE K/V from qkv_proj;
        #                 apply DRAFT's k-norm + RoPE before writing to draft cache.
        # COPY          : hook captured post-norm+RoPE K/V from self_attn (exact
        #                 target values); write directly — no further transforms.
        # ALIAS         : draft kv_cache IS the target kv_cache (aliased in
        #                 set_attn); target already wrote values — skip write step.

        if self._kv_mode == _KVMode.ALIAS:
            # Nothing to precompute: the aliased kv_cache already holds the
            # target's K/V written during its own forward pass.
            pass

        else:
            # Read hooked K/V for RAW_COPY / COPY modes.
            if self._kv_capture:
                hooked = self._kv_capture.pop()
                if hooked is not None:
                    aux_kv_states = hooked

            if aux_kv_states is not None:
                target_k_layers = [kv[0][:num_target_tokens] for kv in aux_kv_states]
                target_v_layers = [kv[1][:num_target_tokens] for kv in aux_kv_states]

                if self._kv_mode == _KVMode.RAW_COPY:
                    # Pre-norm, pre-RoPE: apply draft's own k-norm + RoPE.
                    self.model.precompute_and_store_context_kv_from_target(
                        target_k_layers,
                        target_v_layers,
                        self.context_positions[:num_target_tokens],
                        context_slot_mapping=_ctx_slot_mapping,
                        skip_norm_and_rope=False,
                    )
                else:
                    # COPY: post-norm+RoPE — write directly, no transforms.
                    self.model.precompute_and_store_context_kv_from_target(
                        target_k_layers,
                        target_v_layers,
                        self.context_positions[:num_target_tokens],
                        context_slot_mapping=_ctx_slot_mapping,
                        skip_norm_and_rope=True,
                    )
            else:
                # HIDDEN_STATES: project context K/V from target hidden states.
                self.model.precompute_and_store_context_kv(
                    self.hidden_states[:num_target_tokens],
                    self.context_positions[:num_target_tokens],
                    context_slot_mapping=_ctx_slot_mapping,
                )

        # Every DFlash step has exactly num_query_per_req tokens, so we can use FULL CGs
        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            num_reqs,
            num_query_tokens,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        num_reqs_padded = batch_desc.num_reqs or num_reqs
        num_tokens_padded = batch_desc.num_tokens

        # Rebuild the draft attention metadata even when replaying the FULL
        # graph so that any attention metadata builder state is updated.
        draft_attn_metadata = self._build_draft_attn_metadata(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens_padded=num_tokens_padded,
            causal=self.dflash_causal,
        )
        draft_slot_mappings_by_layer = build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :num_tokens_padded],
            self.kv_cache_config,
        )

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.query_cudagraph_manager is not None
            self.query_cudagraph_manager.run_fullgraph(batch_desc)
        else:
            self._generate_draft(
                num_reqs_padded,
                num_tokens_padded,
                draft_attn_metadata,
                draft_slot_mappings_by_layer,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=batch_desc.cg_mode,
            )

        return self.draft_tokens[:num_reqs]


@triton.jit
def _prepare_dflash_inputs_kernel(
    # Outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Block table for slot mapping lookup.
    block_table_ptr,
    block_table_stride,
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling: splice in the next prefill token.
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_query = (j >= num_ctx) & (j < num_ctx + num_query_per_req)
    query_off = j - num_ctx

    # --- Context positions / slots ---
    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_ctx, other=0)
    ctx_block_num = ctx_pos // block_size
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_ctx,
        other=0,
    ).to(tl.int64)
    ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)
    tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)
    tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)

    # --- Query positions / input_ids / slots ---
    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)

    q_block_num = query_pos // block_size
    q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=is_query,
        other=0,
    ).to(tl.int64)
    q_slot = q_block_id * block_size + (query_pos % block_size)

    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
    tl.store(out_query_positions_ptr + query_idx, query_pos, mask=is_query)
    tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)

    # --- Sample indices / positions / idx_mapping (mask tokens only) ---
    is_sample = is_query & (query_off > 0)
    sample_idx = req_idx * num_speculative_steps + (query_off - 1)
    tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_sample)
    tl.store(out_sample_pos_ptr + sample_idx, query_pos, mask=is_sample)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_sample)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        # seq_lens is the absolute sequence length the draft attention
        # reads up to (context + query), not just the count of accepted
        # tokens this step.
        tl.store(out_seq_lens_ptr + req_idx, last_valid_pos + 1 + num_query_per_req)
        if req_idx == num_reqs - 1:
            # Pad per-request buffers to max_num_reqs for CUDA graph safety.
            last_query_end = num_reqs * num_query_per_req
            for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs + 1
                tl.store(out_query_start_loc_ptr + block, last_query_end, mask=mask)
            for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs
                tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            # Padded sample slots point at query index 0 (a valid row in
            # last_hidden_states) so CG replay never reads OOB.
            pad_start = num_reqs * num_speculative_steps
            pad_end = max_num_reqs * num_speculative_steps
            for i in range(pad_start, pad_end, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < pad_end
                tl.store(out_sample_indices_ptr + block, 0, mask=mask)
                tl.store(out_sample_pos_ptr + block, 0, mask=mask)
                tl.store(out_sample_idx_mapping_ptr + block, 0, mask=mask)
            # Pad query slot mappings past num_query_tokens with PAD so the
            # captured CG sees PAD slots (no K/V write) for replay sizes
            # larger than the current request count.
            q_pad_start = num_reqs * num_query_per_req
            for i in range(q_pad_start, max_num_tokens, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_tokens
                tl.store(out_query_slot_mapping_ptr + block, PAD_SLOT_ID, mask=mask)


def prepare_dflash_inputs(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    # [max_num_reqs, max_num_blocks]
    block_table: torch.Tensor,
    block_size: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
) -> None:
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0
    # Cover the longest possible per-request span (ctx + query). Use the max
    # per-request query length, not the total token count across the batch.
    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    BLOCK_SIZE = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
    _prepare_dflash_inputs_kernel[(num_reqs, num_blocks)](
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        block_table,
        block_table.stride(0),
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        PAD_SLOT_ID=PAD_SLOT_ID,
        BLOCK_SIZE=BLOCK_SIZE,
    )
