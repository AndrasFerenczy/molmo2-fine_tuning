import torch
from torch import Tensor
from typing import Callable

# --- GENERATIO ONLY MASKS ---
def get_mask_mod_w_offset(mask_mod: Callable[[int, int, Tensor, Tensor], Tensor], _offset: Tensor) -> Callable[[int, int, Tensor, Tensor], Tensor]:
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + _offset, kv)
    return _mask_mod

def left_padding_mask_factory_method(padding_offsets: Tensor) -> Callable[[int, int, Tensor, Tensor], Tensor]:
    """
    Create a mask function for left-padded sequences.

    Args:
        padding_offsets: 1D tensor of shape (batch,) indicating the number of
                        left-padding tokens at the beginning of each sequence.

    Returns:
        Mask function that returns False for padding positions, True for valid positions.
    """
    def mask(b, h, q_idx, kv_idx):
        return kv_idx >= padding_offsets[b]
    return mask

def cached_tokens_padding_mask_factory_method(valid_lengths: Tensor, past_max_len: int) -> Callable[[int, int, Tensor, Tensor], Tensor]:
    """
    Mask out padded positions in the middle of batched cached tokens.
    
    When sequences have different cache lengths, shorter sequences have padding
    in the middle range [valid_lengths[b], past_max_len) between their cached
    tokens and the new tokens being added.

    Args:
        valid_lengths: 1D tensor of shape (batch,) indicating number of valid cached tokens.
        past_max_len: maximum cached length across batch (before adding new tokens).

    Returns:
        Mask function that returns False for padded positions [valid_lengths[b], past_max_len),
        and True for all other positions (valid cached tokens and new tokens).
    """
    def mask(b, h, q_idx, kv_idx):
        # Padding region is [valid_lengths[b], past_max_len)
        is_padding = (kv_idx >= valid_lengths[b]) & (kv_idx < past_max_len)
        return ~is_padding
    return mask


# --- GENERAL MASKS ---

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def sliding_window_mask_factory_method(window_size: int) -> Callable[[int, int, Tensor, Tensor], Tensor]:
    return lambda b, h, q_idx, kv_idx: kv_idx >= q_idx - window_size

def fixed_beacons_positions_block_mask_factory_method(beacons_span: int) -> Callable[[int, int, Tensor, Tensor], Tensor]:
    return lambda b, h, q_idx, kv_idx: ((q_idx // (beacons_span + 1)) == (kv_idx // (beacons_span + 1))) | (kv_idx % (beacons_span + 1) == beacons_span)

def sliding_fixed_beacons_positions_block_mask_factory_method(beacons_span: int, window_size: int) -> Callable[[int, int, Tensor, Tensor], Tensor]:
    def mask(b, h, q_idx, kv_idx):
        q_idx_without_beacons = q_idx - (q_idx // (beacons_span + 1))
        kv_idx_without_beacons = kv_idx - (kv_idx // (beacons_span + 1))
        return (kv_idx_without_beacons >= q_idx_without_beacons - window_size) | ((kv_idx % (beacons_span + 1)) == beacons_span)
    return mask

def document_mask_factory_method(documents_idx: Tensor) -> Callable[[int, int, Tensor, Tensor], Tensor]:
    return lambda b, h, q_idx, kv_idx: documents_idx[b][q_idx] == documents_idx[b][kv_idx]

def arbitrary_beacons_positions_block_mask_factory_method(window_size: int, is_beacon_mask: Tensor, num_beacons_until_idx: Tensor) -> Callable[[int, int, Tensor, Tensor], Tensor]:
    """
    Create a mask function for arbitrary beacon positions with sliding window.
    
    Args:
        window_size: Size of the sliding window (excluding beacons)
        is_beacon_mask: Boolean tensor of shape (b, t) indicating beacon positions
        num_beacons_until_idx: Cumsum of is_beacon_mask, shape (b, t)
    
    Note: This function captures input-dependent tensors, so any method calling it
    should be marked with @torch.compiler.disable to avoid compilation issues.
    """
    def mask(b, h, q_idx, kv_idx):
        # Subtract beacon count to get positions in "normal token space"
        q_idx_without_beacons = q_idx - num_beacons_until_idx[b][q_idx]
        kv_idx_without_beacons = kv_idx - num_beacons_until_idx[b][kv_idx]
        # Apply sliding window in normal token space, but always attend to beacons
        return (kv_idx_without_beacons >= q_idx_without_beacons - window_size) | (is_beacon_mask[b][kv_idx])
    return mask


def sliding_gumbel_beacons_mask_factory_method(window_size: int) -> Callable[[int, int, Tensor, Tensor], Tensor]:
    """
    Sliding-window mask for doubled token/beacon sequences used by gumbel beacons.

    Sequence layout is [token0, beacon0, token1, beacon1, ...].
    - Normal-token keys are visible only within sliding window in normal-token space.
    - Beacon keys are always visible (subject to causal/document masks applied elsewhere).
    """

    def mask(b, h, q_idx, kv_idx):
        q_idx_in_t = q_idx // 2
        kv_idx_in_t = kv_idx // 2
        kv_is_beacon = (kv_idx % 2 == 1)
        return (kv_idx_in_t >= q_idx_in_t - window_size) | kv_is_beacon

    return mask


# def beacons_mask_factory_method(q_idx_to_last_beacon_idx: Tensor) -> Callable[[int, int, Tensor, Tensor], Tensor]:

    
#     def beacons_mask(b, h, q_idx, kv_idx):

#         return (q_idx_to_last_beacon_idx[q_idx] < kv_idx) & causal_mask(b, h, q_idx, kv_idx)
