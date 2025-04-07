import functools
import torch
import contextlib
import dataclasses
from collections import defaultdict
from typing import Any, Dict, DefaultDict, Optional, Tuple, Union
import os
import tempfile
import time

VERBOSE_SIMILARITY = False

@dataclasses.dataclass
class CacheContext:
    buffers: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    incremental_name_counters: DefaultDict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))

    def get_incremental_name(self, name=None):
        if name is None:
            name = "default"
        idx = self.incremental_name_counters[name]
        self.incremental_name_counters[name] += 1
        return f"{name}_{idx}"

    def reset_incremental_names(self):
        self.incremental_name_counters.clear()

    @torch.compiler.disable
    def get_buffer(self, name):
        return self.buffers.get(name)

    @torch.compiler.disable
    def set_buffer(self, name, buffer):
        self.buffers[name] = buffer

    def clear_buffers(self):
        self.buffers.clear()

_current_cache_context = None

def create_cache_context():
    return CacheContext()

def get_current_cache_context():
    return _current_cache_context

def set_current_cache_context(cache_context=None):
    global _current_cache_context
    _current_cache_context = cache_context

@contextlib.contextmanager
def cache_context(cache_context):
    global _current_cache_context
    old_cache_context = _current_cache_context
    _current_cache_context = cache_context
    try:
        yield
    finally:
        _current_cache_context = old_cache_context

class MyCacheContext:
    def __init__(self):
        self._buffers = {}
        self.first_hidden_states_residual_multi = None
        self.hidden_states_residual_multi = None
        self.encoder_hidden_states_residual_multi = None

        self.first_cat_hidden_states_residual_single = None
        self.cat_hidden_states_residual_single = None

    def get_buffer(self, name):
        return self._buffers.get(name, None)

    def set_buffer(self, name, val):
        self._buffers[name] = val


from contextlib import contextmanager
@contextmanager
def FBTransformerCacheContext():
    old_ctx = get_current_cache_context()
    new_ctx = MyCacheContext()
    set_current_cache_context(new_ctx)
    try:
        yield new_ctx
    finally:
        set_current_cache_context(old_ctx)

@torch.compiler.disable
def get_buffer(name):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_buffer(name)

@torch.compiler.disable
def set_buffer(name, buffer):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.set_buffer(name, buffer)

@torch.compiler.disable
def are_two_tensors_similar(t1, t2, *, threshold, parallelized=False):
    global VERBOSE_SIMILARITY

    mean_diff = (t1 - t2).abs().mean()
    mean_t1 = t1.abs().mean()
    if parallelized:
        pass
    diff = mean_diff / (mean_t1 + 1e-6)

    if VERBOSE_SIMILARITY:
        print(f"[are_two_tensors_similar] mean_diff={mean_diff.item():.6f}, "
              f"mean_t1={mean_t1.item():.6f}, diff={diff.item():.6f}, threshold={threshold:.3f}")

    return diff.item() < threshold, diff.item()


@torch.compiler.disable
def get_can_use_cache_multi(first_residual: torch.Tensor, threshold: float, parallelized=False):
    #prev_first = get_buffer("first_hidden_states_residual_multi")
    context = get_current_cache_context()
    prev_first = context.first_hidden_states_residual_multi
    
    if prev_first is None:
        return False, threshold

    use_cache, diff = are_two_tensors_similar(
        prev_first,
        first_residual,
        threshold=threshold,
        parallelized=parallelized,
    )

    if use_cache:
        return True, diff
    return False, diff

@torch.compiler.disable
def apply_prev_hidden_states_residual_multi(hidden_states, encoder_hidden_states):
    context = get_current_cache_context()

    hs_res = context.hidden_states_residual_multi
    enc_res = context.encoder_hidden_states_residual_multi

    assert hs_res is not None,  "hidden_states_residual_multi must be set before"
    assert enc_res is not None, "encoder_hidden_states_residual_multi must be set before"

    hidden_states = hidden_states + hs_res
    encoder_hidden_states = encoder_hidden_states + enc_res

    return hidden_states.contiguous(), encoder_hidden_states.contiguous()

'''
@torch.compiler.disable
def apply_prev_hidden_states_residual_multi(hidden_states, encoder_hidden_states):
    hs_res = get_buffer("hidden_states_residual_multi")
    enc_res = get_buffer("encoder_hidden_states_residual_multi")
    assert hs_res is not None, "hidden_states_residual_multi must be set before"
    assert enc_res is not None, "encoder_hidden_states_residual_multi must be set before"

    hidden_states = hidden_states + hs_res
    encoder_hidden_states = encoder_hidden_states + enc_res

    return hidden_states.contiguous(), encoder_hidden_states.contiguous()
'''
@torch.compiler.disable
def get_can_use_cache_single(first_cat_residual: torch.Tensor, threshold: float, parallelized=False):
    #prev_first = get_buffer("first_cat_hidden_states_residual_single")
    context = get_current_cache_context()
    prev_first = context.first_cat_hidden_states_residual_single
    
    if prev_first is None:
        return False, threshold

    use_cache, diff = are_two_tensors_similar(
        prev_first,
        first_cat_residual,
        threshold=threshold,
        parallelized=parallelized,
    )

    if use_cache:
        return True, diff
    return False, diff

@torch.compiler.disable
def apply_prev_cat_hidden_states_residual_single(cat_hidden_states):
    #cat_res = get_buffer("cat_hidden_states_residual_single")
    context = get_current_cache_context()
    cat_res = context.cat_hidden_states_residual_single
    assert cat_res is not None, "cat_hidden_states_residual_single must be set before"

    cat_hidden_states = cat_hidden_states + cat_res
    return cat_hidden_states.contiguous()


def safe_save(image, path):
    """
    Save an image safely by writing to a temporary file first and then renaming it.
    """
    # Ensure the target directory exists.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Get the extension of the target file (e.g., .png, .jpg)
    _, ext = os.path.splitext(path)
    
    # Create a temporary file in the same directory with the same extension.
    fd, temp_path = tempfile.mkstemp(suffix=ext, dir=os.path.dirname(path))
    os.close(fd)  # Close the file descriptor so that Pillow can write to it.
    
    try:
        image.save(temp_path)
        os.replace(temp_path, path)  # Atomic operation if on the same filesystem.
        time.sleep(5)
        print(f"Image saved successfully at: {path}")
    except Exception as e:
        print(f"Error saving image at {path}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
            