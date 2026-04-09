"""
Full attention model with standard causal attention.
"""

import math
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from torch.nn.attention.flex_attention import flex_attention, create_block_mask, and_masks
from modeling.masking import causal_mask, document_mask_factory_method, left_padding_mask_factory_method, get_mask_mod_w_offset, cached_tokens_padding_mask_factory_method

# Import shared components from model.py
from modeling.models.model import (
    RMSNorm,
    Block,
    CausalSelfAttention,
    MLP,
    precompute_freqs_cis,
    apply_rotary_emb,
    compute_left_padded_position_ids,
    generate_left_padded_document_idx,
    infer_is_real_tokens,
    ModelConfig as BaseModelConfig,
    IGNORE_INDEX,
    validate_left_padded_tokens,
)
from modeling.models.utils.sampling import sample_next_token

try:
    from modeling.models.attention.triton_keybias_flash_attention import keybias_attention as triton_keybias_attention
except Exception:
    triton_keybias_attention = None



@dataclass
class ModelConfig(BaseModelConfig):
    """Configuration for full attention models."""
    use_triton_full_attention: bool = False
    warp_specialize: bool = False


class TritonFullAttention(CausalSelfAttention):
    """Standard causal attention backed by the DMS Triton key-bias kernel."""

    def __init__(self, config):
        super().__init__(config)
        self.warp_specialize = getattr(config, "warp_specialize", False)

    def _triton_key_bias(self, *, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.n_head, seq_len), device=device, dtype=torch.bfloat16)

    def _triton_key_bias_window(self) -> int:
        return 1

    def forward(
        self,
        x,
        freqs_cis: torch.Tensor,
        attn_block_mask: Optional[torch.Tensor] = None,
        past_key_values: Tuple[torch.Tensor, torch.Tensor] = None,
        documents_idx_BxT: Optional[torch.Tensor] = None,
    ):
        if (
            triton_keybias_attention is None
            or not x.is_cuda
            or attn_block_mask is not None
            or past_key_values is not None
            or documents_idx_BxT is None
        ):
            return super().forward(
                x,
                freqs_cis,
                attn_block_mask=attn_block_mask,
                past_key_values=past_key_values,
            )

        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.hidden_size, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        q = q.transpose(1, 2).to(torch.bfloat16)
        k = k.transpose(1, 2).to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        key_bias_BxHxT = self._triton_key_bias(batch_size=B, seq_len=T, device=x.device)
        y = triton_keybias_attention(
            q,
            k,
            v,
            1.0 / math.sqrt(q.shape[-1]),
            key_bias_BxHxT,
            key_bias_window=self._triton_key_bias_window(),
            warp_specialize=self.warp_specialize,
            documents_idx_BxT=documents_idx_BxT.contiguous(),
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = y.to(self.c_proj.weight.dtype)
        y = self.resid_dropout(self.c_proj(y))
        return y, (k, v)


class TritonFullAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config)
        self.attn = TritonFullAttention(config)
        self.mlp_norm = RMSNorm(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x,
        freqs_cis: torch.Tensor,
        attn_block_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        documents_idx_BxT: Optional[torch.Tensor] = None,
    ):
        attn_output, past_key_values = self.attn(
            self.attention_norm(x),
            freqs_cis,
            attn_block_mask=attn_block_mask,
            past_key_values=past_key_values,
            documents_idx_BxT=documents_idx_BxT,
        )
        x = x + attn_output
        x = x + self.mlp(self.mlp_norm(x))
        return x, past_key_values

class Model(nn.Module):
    """
    Full attention model with standard causal attention.
    This is the concrete implementation of the transformer model.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.pad_token_id is not None, "pad_token_id must be provided in config"
        self.config = config
        self.use_triton_full_attention = bool(
            getattr(config, "use_triton_full_attention", False)
        )
        block_cls = self._get_block_cls()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.hidden_size),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([block_cls(config) for _ in range(config.n_layer)]),
            output_norm = RMSNorm(config)
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                self.config.hidden_size // self.config.n_head, self.config.block_size 
            ),
            persistent=False,
        )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        self.last_beacon_idx = None

    def _get_block_cls(self):
        return TritonFullAttentionBlock if self.use_triton_full_attention else Block

    def generate_document_idx(self, idx_BxT: Tensor) -> Tensor:
        """
        Generate document indices for each token based on EOS token positions.
        """
        return generate_left_padded_document_idx(
            idx_BxT,
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
        )

    def get_prefilling_mask_function(
        self,
        idx_BxT: Tensor,
        documents_idx_BxT: Optional[Tensor] = None,
    ) -> Callable:
        """
        Get the attention mask function for full-sequence training/evaluation without padding.
        Padding is applied by create_attention_mask.
        Can be overridden by subclasses to add model-specific masks (e.g., sliding window).
        """
        mask_fn = causal_mask

        if documents_idx_BxT is None:
            documents_idx_BxT = self.generate_document_idx(idx_BxT)
        mask_fn = and_masks(mask_fn, document_mask_factory_method(documents_idx_BxT))

        return mask_fn

    def apply_padding_masks(
        self,
        mask_fn: Callable,
        idx_BxT: Tensor,
        is_real_BxT: Tensor,
        cache_lengths: Optional[Tensor] = None,
    ) -> Callable:
        """
        Apply padding masks to a base mask function.
        Used during padded training/evaluation to ensure consistent padding handling.
        """
        device = idx_BxT.device

        padding_offsets = is_real_BxT.long().argmax(dim=1)
        mask_fn = and_masks(mask_fn, left_padding_mask_factory_method(padding_offsets))

        if cache_lengths is not None and (cache_lengths != 0).any():
            batched_kv_len = cache_lengths.max()
            lengths_tensor = cache_lengths.to(device)
            cached_pad_mask_fn = cached_tokens_padding_mask_factory_method(lengths_tensor, past_max_len=batched_kv_len)
            mask_fn = and_masks(mask_fn, cached_pad_mask_fn)

            if batched_kv_len == 0:
                seq_lengths = is_real_BxT.sum(dim=1)
                max_seq_len = seq_lengths.max()
                if max_seq_len < idx_BxT.shape[1]:
                    right_pad_mask_fn = cached_tokens_padding_mask_factory_method(
                        seq_lengths.to(device), past_max_len=idx_BxT.shape[1]
                    )
                    mask_fn = and_masks(mask_fn, right_pad_mask_fn)

            mask_fn = get_mask_mod_w_offset(mask_fn, batched_kv_len)

        return mask_fn

    def create_attention_mask(
        self,
        idx_BxT: Tensor,
        cache_lengths: Tensor,
        is_real_BxT: Optional[Tensor] = None,
        documents_idx_BxT: Optional[Tensor] = None,
    ) -> Optional[Callable]:
        """
        Model-level hook for building attention masks.
        """
        if is_real_BxT is None:
            is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        batched_kv_len = cache_lengths.max()
        prefilling_mode = batched_kv_len == 0

        if prefilling_mode:
            mask_fn = self.get_prefilling_mask_function(idx_BxT, documents_idx_BxT=documents_idx_BxT)
            if not bool(is_real_BxT.all()):
                padding_offsets = is_real_BxT.long().argmax(dim=1)
                mask_fn = and_masks(mask_fn, left_padding_mask_factory_method(padding_offsets))
        else:
            mask_fn = self.apply_padding_masks(causal_mask, idx_BxT, is_real_BxT, cache_lengths)

        return mask_fn

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _can_use_triton_prefill(
        self,
        idx_BxT: Tensor,
        is_real_BxT: Tensor,
    ) -> bool:
        return (
            self.use_triton_full_attention
            and triton_keybias_attention is not None
            and idx_BxT.is_cuda
            and bool(is_real_BxT.all())
        )

    def _forward_block(
        self,
        block: nn.Module,
        x: Tensor,
        freqs_cis: Tensor,
        *,
        attn_block_mask: Optional[torch.Tensor],
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
        documents_idx_BxT: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if self.use_triton_full_attention:
            return block(
                x,
                freqs_cis,
                attn_block_mask=attn_block_mask,
                past_key_values=past_key_values,
                documents_idx_BxT=documents_idx_BxT,
            )
        return block(
            x,
            freqs_cis,
            attn_block_mask=attn_block_mask,
            past_key_values=past_key_values,
        )


    def forward_hidden_states(
        self, idx_BxT: Tensor, targets_BxT: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass for training when inputs already include beacons/padding.
        """
        device = idx_BxT.device
        b, t = idx_BxT.size()

        is_not_pad_mask = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(
            is_not_pad_mask,
            allow_all_pad=True,
            context="forward inputs",
        )
        use_triton_prefill = self._can_use_triton_prefill(idx_BxT, is_not_pad_mask)

        if use_triton_prefill:
            position_ids = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)
        else:
            position_ids = compute_left_padded_position_ids(is_not_pad_mask)

        assert t <= self.freqs_cis.shape[0], f"Cannot forward sequence of length {t}, block size is only {self.freqs_cis.shape[0]}"

        all_freqs_cis = self.freqs_cis.to(device)
        freqs_cis = all_freqs_cis[position_ids]

        documents_idx_BxT = self.generate_document_idx(idx_BxT)
        if use_triton_prefill:
            attn_block_mask = None
        else:
            cache_lengths = torch.zeros(b, dtype=torch.long, device=device)
            attn_mask_function = self.create_attention_mask(
                idx_BxT=idx_BxT,
                cache_lengths=cache_lengths,
                is_real_BxT=is_not_pad_mask,
                documents_idx_BxT=documents_idx_BxT,
            )
            attn_block_mask = create_block_mask(
                attn_mask_function,
                B=b,
                H=None,
                Q_LEN=t,
                KV_LEN=t,
                device=device
            )

        tok_emb = self.transformer.wte(idx_BxT)
        x = self.transformer.drop(tok_emb)

        past_key_values_Lx2 = [None for _ in range(self.config.n_layer)]
        for layer_idx, block in enumerate(self.transformer.h):
            x, _ = self._forward_block(
                block,
                x,
                freqs_cis,
                attn_block_mask=attn_block_mask,
                past_key_values=past_key_values_Lx2[layer_idx],
                documents_idx_BxT=documents_idx_BxT if use_triton_prefill else None,
            )

        x = self.transformer.output_norm(x)

        return x, targets_BxT, is_not_pad_mask

    def forward(self, idx_BxT: Tensor, targets_BxT: Tensor):
        """
        Forward pass for training. Computes logits and loss.
        """
        device = idx_BxT.device

        x, targets_BxT, is_not_pad_mask = self.forward_hidden_states(idx_BxT, targets_BxT)
        original_t = idx_BxT.size(1)

        b, t = x.size(0), x.size(1)

        logits = self.lm_head(x)

        # Return logits without beacons (if any)
        original_logits = logits[targets_BxT != IGNORE_INDEX].view(b, original_t, logits.size(-1))

        # Mask out padded positions and EOS positions in targets
        targets_BxT[~is_not_pad_mask] = IGNORE_INDEX
        targets_BxT[idx_BxT == self.config.eos_token_id] = IGNORE_INDEX

        # Compute loss, ignoring padded positions
        token_count = (targets_BxT != IGNORE_INDEX).sum()
        token_nll_sum = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets_BxT.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        loss = token_nll_sum / token_count.clamp(min=1)

        stats = {
            "token_nll_sum": token_nll_sum.detach(),
            "token_nll_count": token_count.detach(),
            "token_count": token_count.detach(),
            "beacon_loss_sum": torch.zeros((), device=device, dtype=token_nll_sum.dtype),
            "beacon_loss_count": torch.zeros((), device=device, dtype=torch.long),
            "beacon_count": torch.zeros((), device=device, dtype=torch.long),
        }

        return original_logits, loss, stats

    @contextmanager
    def _temporary_disable_generation_triton(self):
        previous_use_triton = self.use_triton_full_attention
        previous_config_flag = getattr(self.config, "use_triton_full_attention", previous_use_triton)
        self.use_triton_full_attention = False
        if hasattr(self.config, "use_triton_full_attention"):
            self.config.use_triton_full_attention = False
        try:
            yield
        finally:
            self.use_triton_full_attention = previous_use_triton
            if hasattr(self.config, "use_triton_full_attention"):
                self.config.use_triton_full_attention = previous_config_flag

    def _sample_next_token(
        self,
        logits_BxV: Tensor,
        active_mask_B: Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        forbidden_token_ids: Optional[Tensor] = None,
    ) -> Tensor:
        return sample_next_token(
            logits_BxV,
            active_mask_B,
            pad_token_id=self.config.pad_token_id,
            suppressed_token_ids=(self.config.pad_token_id,),
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            forbidden_token_ids=forbidden_token_ids,
        )

    def _forward_generation_hidden_states(
        self,
        idx_BxT: Tensor,
        *,
        past_key_values_Lx2: Optional[list[Optional[Tuple[Tensor, Tensor]]]] = None,
        cache_lengths_B: Optional[Tensor] = None,
    ) -> tuple[Tensor, list[Tuple[Tensor, Tensor]]]:
        device = idx_BxT.device
        b, t = idx_BxT.size()
        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(
            is_real_BxT,
            allow_all_pad=False,
            context="generation inputs",
        )

        if cache_lengths_B is None:
            cache_lengths_B = torch.zeros(b, dtype=torch.long, device=device)
        batched_kv_len = int(cache_lengths_B.max().item())

        position_ids_BxT = compute_left_padded_position_ids(is_real_BxT) + cache_lengths_B.unsqueeze(1)
        max_position = int(position_ids_BxT[is_real_BxT].max().item())
        if max_position >= self.freqs_cis.shape[0]:
            raise ValueError(
                f"Cannot generate position {max_position} when block size is {self.freqs_cis.shape[0]}"
            )

        all_freqs_cis = self.freqs_cis.to(device)
        freqs_cis = all_freqs_cis[position_ids_BxT]

        documents_idx_BxT = None
        if batched_kv_len == 0:
            documents_idx_BxT = self.generate_document_idx(idx_BxT)
        attn_mask_function = self.create_attention_mask(
            idx_BxT=idx_BxT,
            cache_lengths=cache_lengths_B,
            is_real_BxT=is_real_BxT,
            documents_idx_BxT=documents_idx_BxT,
        )
        attn_block_mask = create_block_mask(
            attn_mask_function,
            B=b,
            H=None,
            Q_LEN=t,
            KV_LEN=t + batched_kv_len,
            device=device,
        )

        x = self.transformer.drop(self.transformer.wte(idx_BxT))
        if past_key_values_Lx2 is None:
            past_key_values_Lx2 = [None for _ in range(self.config.n_layer)]
        next_past_key_values_Lx2 = []
        for layer_idx, block in enumerate(self.transformer.h):
            x, layer_past_key_values = self._forward_block(
                block,
                x,
                freqs_cis,
                attn_block_mask=attn_block_mask,
                past_key_values=past_key_values_Lx2[layer_idx],
                documents_idx_BxT=documents_idx_BxT if self.use_triton_full_attention else None,
            )
            next_past_key_values_Lx2.append(layer_past_key_values)
        x = self.transformer.output_norm(x)
        return x, next_past_key_values_Lx2

    def _generate_single_unpadded(
        self,
        prompt_T: Tensor,
        max_new_tokens: int,
        *,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        stop_on_eos: bool,
        forbidden_token_ids: Optional[Tensor],
    ) -> Tensor:
        device = prompt_T.device
        prompt_1xT = prompt_T.unsqueeze(0)
        current_len = prompt_1xT.size(1)
        hidden_states_BxTxC, past_key_values_Lx2 = self._forward_generation_hidden_states(prompt_1xT)
        next_logits_BxV = self.lm_head(hidden_states_BxTxC)[:, -1, : self.config.vocab_size]

        generated_T = torch.full(
            (max_new_tokens,),
            self.config.pad_token_id,
            device=device,
            dtype=prompt_T.dtype,
        )
        active_mask_B = torch.ones((1,), device=device, dtype=torch.bool)

        for step in range(max_new_tokens):
            next_token_B = self._sample_next_token(
                next_logits_BxV,
                active_mask_B,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                forbidden_token_ids=forbidden_token_ids,
            ).to(prompt_T.dtype)
            generated_T[step] = next_token_B[0]
            if stop_on_eos and int(next_token_B[0].item()) == self.config.eos_token_id:
                active_mask_B[0] = False
                break

            hidden_states_BxTxC, past_key_values_Lx2 = self._forward_generation_hidden_states(
                next_token_B.view(1, 1),
                past_key_values_Lx2=past_key_values_Lx2,
                cache_lengths_B=torch.tensor([current_len], device=device, dtype=torch.long),
            )
            current_len += 1
            next_logits_BxV = self.lm_head(hidden_states_BxTxC)[:, -1, : self.config.vocab_size]

        return torch.cat([prompt_T, generated_T], dim=0)

    @torch.no_grad()
    def generate(
        self,
        idx_BxT: Tensor,
        max_new_tokens: int,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_on_eos: bool = True,
        forbidden_token_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
        if max_new_tokens == 0:
            return idx_BxT.clone()

        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(
            is_real_BxT,
            allow_all_pad=False,
            context="generation prompts",
        )

        max_real_prompt_tokens = int(is_real_BxT.sum(dim=1).max().item())
        total_real_tokens = max_real_prompt_tokens + int(max_new_tokens)
        if total_real_tokens > self.freqs_cis.shape[0]:
            raise ValueError(
                f"Cannot generate {max_new_tokens} new tokens from a prompt with "
                f"{max_real_prompt_tokens} real tokens when block size is {self.freqs_cis.shape[0]}"
            )

        output_rows = []
        with self._temporary_disable_generation_triton():
            for row_idx in range(idx_BxT.size(0)):
                row = idx_BxT[row_idx]
                prompt_T = row[is_real_BxT[row_idx]]
                generated_row = self._generate_single_unpadded(
                    prompt_T,
                    max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    stop_on_eos=stop_on_eos,
                    forbidden_token_ids=forbidden_token_ids,
                )
                generated_suffix = generated_row[prompt_T.numel() :]
                output_rows.append(torch.cat([row, generated_suffix], dim=0))
        return torch.stack(output_rows, dim=0)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
