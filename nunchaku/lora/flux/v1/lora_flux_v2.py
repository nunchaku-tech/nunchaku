import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors import safe_open

from nunchaku.lora.flux.nunchaku_converter import pack_lowrank_weight, reorder_adanorm_lora_up, unpack_lowrank_weight

logger = logging.getLogger(__name__)

_RE_QKV_DBL_DECOMP = re.compile(r"^(transformer_blocks\.\d+)\.attn\.to_(q|k|v)(?=\.|$)")
_RE_QKV_DBL_FUSED = re.compile(r"^(transformer_blocks\.\d+)\.attn\.to_qkv(?=\.|$)")
_RE_ADDQKV_DBL_DECOMP = re.compile(r"^(transformer_blocks\.\d+)\.attn\.add_(q|k|v)_proj(?=\.|$)")
_RE_ADDQKV_DBL_FUSED = re.compile(r"^(transformer_blocks\.\d+)\.attn\.add_qkv_proj(?=\.|$)")

_RE_QKV_SGL_DECOMP = re.compile(r"^(single_transformer_blocks\.\d+)\.attn\.to_(q|k|v)(?=\.|$)")
_RE_QKV_SGL_FUSED = re.compile(r"^(single_transformer_blocks\.\d+)\.attn\.to_qkv(?=\.|$)")

_RE_OUTPROJ_DBL = re.compile(r"^(transformer_blocks\.\d+)\.out_proj(?=\.|$)")
_RE_OUTPROJ_CTX_DBL = re.compile(r"^(transformer_blocks\.\d+)\.out_proj_context(?=\.|$)")
_RE_TOOUT_DBL = re.compile(r"^(transformer_blocks\.\d+)\.attn\.to_out(?=\.|$)")
_RE_TOADDOUT_DBL = re.compile(r"^(transformer_blocks\.\d+)\.attn\.to_add_out(?=\.|$)")

_RE_PROJOUT_SGL = re.compile(r"^(single_transformer_blocks\.\d+)\.proj_out(?=\.|$)")
_RE_PROJMLP_SGL = re.compile(r"^(single_transformer_blocks\.\d+)\.proj_mlp(?=\.|$)")
_RE_TOOUT_SGL = re.compile(r"^(single_transformer_blocks\.\d+)\.attn\.to_out(?=\.|$)")

_RE_NORM_SGL = re.compile(r"^(single_transformer_blocks\.\d+)\.norm\.linear(?=\.|$)")
_RE_NORM1_DBL = re.compile(r"^(transformer_blocks\.\d+)\.norm1\.linear(?=\.|$)")
_RE_NORM1CTX_DBL = re.compile(r"^(transformer_blocks\.\d+)\.norm1_context\.linear(?=\.|$)")

_RE_FF_DBL_FC1 = re.compile(r"^(transformer_blocks\.\d+)\.ff\.net\.0(?:\.proj)?(?=\.|$)")
_RE_FF_DBL_FC2 = re.compile(r"^(transformer_blocks\.\d+)\.ff\.net\.2(?=\.|$)")
_RE_FFCTX_DBL_FC1 = re.compile(r"^(transformer_blocks\.\d+)\.ff_context\.net\.0(?:\.proj)?(?=\.|$)")
_RE_FFCTX_DBL_FC2 = re.compile(r"^(transformer_blocks\.\d+)\.ff_context\.net\.2(?=\.|$)")

_RE_MLP_IMG_FC1 = re.compile(r"^(transformer_blocks\.\d+\.img_mlp\.net\.0(?:\.proj)?)(?=\.|$)")
_RE_MLP_IMG_FC2 = re.compile(r"^(transformer_blocks\.\d+\.img_mlp\.net\.2)(?=\.|$)")
_RE_MLP_TXT_FC1 = re.compile(r"^(transformer_blocks\.\d+\.txt_mlp\.net\.0(?:\.proj)?)(?=\.|$)")
_RE_MLP_TXT_FC2 = re.compile(r"^(transformer_blocks\.\d+\.txt_mlp\.net\.2)(?=\.|$)")

def _classify_and_map_key(key: str) -> Optional[Tuple[str, str, Optional[str], str]]:
    """
    key -> (group, base_key, comp, ab)
    """
    k = key
    if k.startswith("transformer."):
        k = k[len("transformer.") :]

    if ".lora_A" in k:
        ab = "A"
        base = k.replace(".lora_A.weight", "").replace(".lora_A", "")
    elif ".lora_B" in k:
        ab = "B"
        base = k.replace(".lora_B.weight", "").replace(".lora_B", "")
    elif ".lora_down" in k:
        ab = "A"
        base = k.replace(".lora_down.weight", "").replace(".lora_A", "")
    elif ".lora_up" in k:
        ab = "B"
        base = k.replace(".lora_down.weight", "").replace(".lora_B", "")
    elif ".alpha" in k:
        ab = "alpha"
        base = k.replace(".alpha", "")
    else:
        return None

    # QKV (double)
    m = _RE_QKV_DBL_FUSED.match(base)
    if m:
        return ("qkv", f"{m.group(1)}.attn.to_qkv", None, ab)

    m = _RE_QKV_DBL_DECOMP.match(base)
    if m:
        return ("qkv", f"{m.group(1)}.attn.to_qkv", m.group(2).upper(), ab)

    # ADD_QKV (double)
    m = _RE_ADDQKV_DBL_FUSED.match(base)
    if m:
        return ("add_qkv", f"{m.group(1)}.attn.add_qkv_proj", None, ab)

    m = _RE_ADDQKV_DBL_DECOMP.match(base)
    if m:
        return ("add_qkv", f"{m.group(1)}.attn.add_qkv_proj", m.group(2).upper(), ab)

    # QKV (single)
    m = _RE_QKV_SGL_FUSED.match(base)
    if m:
        return ("qkv", f"{m.group(1)}.attn.to_qkv", None, ab)

    m = _RE_QKV_SGL_DECOMP.match(base)
    if m:
        return ("qkv", f"{m.group(1)}.attn.to_qkv", m.group(2).upper(), ab)

    # out/ff (double)
    m = _RE_OUTPROJ_CTX_DBL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.attn.to_add_out", None, ab)

    m = _RE_TOADDOUT_DBL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.attn.to_add_out", None, ab)

    m = _RE_OUTPROJ_DBL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.attn.to_out.0", None, ab)

    m = _RE_TOOUT_DBL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.attn.to_out.0", None, ab)

    m = _RE_FF_DBL_FC1.match(base)
    if m:
        return ("regular", f"{m.group(1)}.mlp_fc1", None, ab)

    m = _RE_FF_DBL_FC2.match(base)
    if m:
        return ("regular", f"{m.group(1)}.mlp_fc2", None, ab)

    m = _RE_FFCTX_DBL_FC1.match(base)
    if m:
        return ("regular", f"{m.group(1)}.mlp_context_fc1", None, ab)

    m = _RE_FFCTX_DBL_FC2.match(base)
    if m:
        return ("regular", f"{m.group(1)}.mlp_context_fc2", None, ab)

    # single
    m = _RE_PROJOUT_SGL.match(base)
    if m:
        return ("single_proj_out", f"{m.group(1)}.proj_out", None, ab)

    m = _RE_PROJMLP_SGL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.mlp_fc1", None, ab)

    m = _RE_TOOUT_SGL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.attn.to_out", None, ab)

    # norm.linear
    m = _RE_NORM_SGL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.norm.linear", None, ab)

    m = _RE_NORM1_DBL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.norm1.linear", None, ab)

    m = _RE_NORM1CTX_DBL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.norm1_context.linear", None, ab)

    m = _RE_MLP_IMG_FC1.match(base) or _RE_MLP_TXT_FC1.match(base)
    if m:
        return ("regular", m.group(1), None, ab)

    m = _RE_MLP_IMG_FC2.match(base) or _RE_MLP_TXT_FC2.match(base)
    if m:
        return ("regular", m.group(1), None, ab)

    return None


def _resolve_module_name(model: nn.Module, name: str) -> Tuple[str, Optional[nn.Module]]:
    """
    Resolve a name string path to a module, attempting fallback paths only when necessary.
    Returns immediately when found, and only tries correction branches when not found.
    """
    # 1) First try to find it as-is
    m = _get_module_by_name(model, name)
    if m is not None:
        return name, m

    # 2) Only try correction paths when not found

    # to_out(.0) bidirectional correction
    if name.endswith(".attn.to_out.0"):
        alt = name[:-2]  # Remove ".0"
        m = _get_module_by_name(model, alt)
        if m is not None:
            # print("[OK] resolved to:", alt)
            return alt, m
    elif name.endswith(".attn.to_out"):
        alt = name + ".0"
        m = _get_module_by_name(model, alt)
        if m is not None:
            # print("[OK] resolved to:", alt)
            return alt, m

    # FF ↔ MLP correction
    mapping = {
        ".ff.net.0.proj": ".mlp_fc1",
        ".ff.net.2": ".mlp_fc2",
        ".ff_context.net.0.proj": ".mlp_context_fc1",
        ".ff_context.net.2": ".mlp_context_fc2",
        ".mlp_fc1": ".ff.net.0.proj",
        ".mlp_fc2": ".ff.net.2",
        ".mlp_context_fc1": ".ff_context.net.0.proj",
        ".mlp_context_fc2": ".ff_context.net.2",
    }
    for src, dst in mapping.items():
        if src in name:
            alt = name.replace(src, dst)
            m = _get_module_by_name(model, alt)
            if m is not None:
                # print("[OK] resolved to:", alt)
                return alt, m

    # 3) If still not found, it really doesn't exist
    print("[MISS] not found:", name)
    return name, None


def _is_indexable_module(m):
    return isinstance(m, (nn.ModuleList, nn.Sequential, list, tuple))


def _get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """
    Traverse a path like 'a.b.3.c' to find and return a module.
    """
    if not name:
        return model

    module = model
    for raw in name.split("."):
        part = raw.strip()
        if part == "":
            continue

        # Numeric index
        if part.isdigit():
            if _is_indexable_module(module):
                idx = int(part)
                try:
                    module = module[idx]
                except Exception:
                    return None
            else:
                return None
            continue

        # Regular attribute
        if hasattr(module, part):
            module = getattr(module, part)
            continue

        # Handle dict-like containers like ModuleDict (if present)
        if hasattr(module, "__getitem__"):
            try:
                module = module[part]
                continue
            except Exception:
                pass

        return None

    return module


def _load_lora_state_dict(lora_state_dict_or_path: Union[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Load LoRA state dict from path or return existing dict."""
    if isinstance(lora_state_dict_or_path, str):
        path = Path(lora_state_dict_or_path)
        if path.suffix == ".safetensors":
            state_dict = {}
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            return state_dict
        else:
            return torch.load(path, map_location="cpu")
    return lora_state_dict_or_path


def _fuse_qkv_lora(qkv_weights: Dict[str, torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Fuse Q/K/V LoRA weights into a single QKV tensor.
    Parameters
    ----------
    qkv_weights : Dict[str, torch.Tensor]
        Dictionary with keys like "Q_A", "Q_B", "K_A", "K_B", "V_A", "V_B"
    Returns
    -------
    Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
        Fused A and B tensors, or (None, None) if not all components present
    """
    # Check if we have all components
    required_keys = ["Q_A", "Q_B", "K_A", "K_B", "V_A", "V_B"]
    if not all(k in qkv_weights for k in required_keys):
        return None, None

    # Get individual A and B matrices
    A_q = qkv_weights["Q_A"]
    A_k = qkv_weights["K_A"]
    A_v = qkv_weights["V_A"]
    B_q = qkv_weights["Q_B"]
    B_k = qkv_weights["K_B"]
    B_v = qkv_weights["V_B"]

    if any(x is None for x in [A_q, A_k, A_v, B_q, B_k, B_v]):
        return None, None

    alpha_q = qkv_weights.get("Q_alpha")
    alpha_k = qkv_weights.get("K_alpha")
    alpha_v = qkv_weights.get("V_alpha")

    alpha_fused: Optional[float] = None
    if all(alpha is not None for alpha in [alpha_q, alpha_k, alpha_v]):
        q_val, k_val, v_val = alpha_q.item(), alpha_k.item(), alpha_v.item()
        if q_val == k_val == v_val:
            alpha_fused = q_val

    # Validate dimensions
    if not (A_q.shape == A_k.shape == A_v.shape):
        raise ValueError(f"Q/K/V LoRA A dimensions mismatch: {A_q.shape}, {A_k.shape}, {A_v.shape}")

    # Fuse: for QKV, we concatenate along output dimension
    # A remains the same for all three (shared input projection)
    # B is block-diagonal concatenation

    # A matrix: [r, in_features] - shared across Q/K/V
    # We need to replicate it 3 times for the 3 rank components
    A_fused = torch.cat([A_q, A_k, A_v], dim=0)  # [3*r, in_features]

    # B matrix: block diagonal structure for Q, K, V outputs
    # B_q: [out_q, r], B_k: [out_k, r], B_v: [out_v, r]
    # Fused B: [out_q + out_k + out_v, 3*r] with block diagonal structure
    r = B_q.shape[1]
    out_q, out_k, out_v = B_q.shape[0], B_k.shape[0], B_v.shape[0]
    total_out = out_q + out_k + out_v

    # Create block-diagonal B matrix
    B_fused = torch.zeros(total_out, 3 * r, dtype=B_q.dtype, device=B_q.device)
    B_fused[:out_q, :r] = B_q
    B_fused[out_q : out_q + out_k, r : 2 * r] = B_k
    B_fused[out_q + out_k :, 2 * r :] = B_v

    return A_fused, B_fused, alpha_fused  # Return without transpose - already in correct shape


def _handle_proj_out_split(
    lora_dict: Dict[str, Dict[str, torch.Tensor]], base_key: str, model: nn.Module
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], List[str]]:
    """
    Split single-block proj_out LoRA into two branches:
      - single_transformer_blocks.{i}.attn.to_out
      - single_transformer_blocks.{i}.mlp_fc2
    A_full: [r_lora, attn_in + mlp_in],  B_full: [out, r_lora] (shared)
    """
    import re

    result, consumed = {}, []
    m = re.search(r"single_transformer_blocks\.(\d+)", base_key)
    if not m or base_key not in lora_dict:
        return result, consumed

    block_idx = m.group(1)
    block = _get_module_by_name(model, f"single_transformer_blocks.{block_idx}")
    if block is None:
        return result, consumed

    A_full = lora_dict[base_key].get("A")
    B_full = lora_dict[base_key].get("B")
    if A_full is None or B_full is None:
        return result, consumed

    attn_to_out = getattr(block.attn, "to_out", None)  # SVDQW4A4Linear
    mlp_fc2 = getattr(block, "mlp_fc2", None)  # SVDQW4A4Linear
    if attn_to_out is None or mlp_fc2 is None:
        return result, consumed

    attn_in, mlp_in = attn_to_out.in_features, mlp_fc2.in_features
    out_attn, out_mlp = attn_to_out.out_features, mlp_fc2.out_features
    if out_attn != out_mlp:
        raise ValueError(f"proj_out split: out mismatch ({out_attn} vs {out_mlp})")

    if A_full.shape[1] != attn_in + mlp_in:
        raise ValueError(f"{base_key}: A_full {A_full.shape} vs attn_in({attn_in})+mlp_in({mlp_in})")

    A_attn = A_full[:, :attn_in]
    A_mlp = A_full[:, attn_in:]

    result[f"single_transformer_blocks.{block_idx}.attn.to_out"] = (A_attn, B_full)
    result[f"single_transformer_blocks.{block_idx}.mlp_fc2"] = (A_mlp, B_full)
    consumed.append(base_key)
    return result, consumed


def update_lora_params_v2(
    model: torch.nn.Module,
    lora_state_dict_or_path: Union[str, Dict[str, torch.Tensor]],
    strength: float = 1.0,
    allow_expand: bool = False,
) -> None:
    """
    Load LoRA weights and in-place append low-rank factors to SVDQW4A4Linear modules.
    - No module replacement or new wrappers.
    - Maps LoRA keys to V2 names; handles fused QKV, add_qkv, and single-block proj_out split.
    - Applies 'strength' by scaling appended B; records base/appended ranks in model._lora_slots.
    Parameters
    ----------
    model : torch.nn.Module
        The V2 model to update
    lora_state_dict_or_path : Union[str, Dict[str, torch.Tensor]]
        Path to LoRA weights or state dict
    strength : float, optional
        LoRA strength scaling factor (default: 1.0)
    allow_expand : bool, optional
        Allow expanding input dimensions if needed (default: False)
    """
    # Load LoRA weights
    lora_state_dict = _load_lora_state_dict(lora_state_dict_or_path)

    # Initialize tracking
    if not hasattr(model, "_lora_slots"):
        model._lora_slots = {}
    if not hasattr(model, "_lora_strength"):
        model._lora_strength = strength

    # Group LoRA weights by base module - need special handling for fused modules
    lora_dict: Dict[str, Dict[str, torch.Tensor]] = {}
    unused_keys: List[str] = []

    for key, value in lora_state_dict.items():
        parsed = _classify_and_map_key(key)
        if parsed is None:
            unused_keys.append(key)
            continue

        group, base_key, comp, ab = parsed
        entry = lora_dict.setdefault(base_key, {})

        if group in ("qkv", "add_qkv"):
            # Fused priority: if attn.to_qkv.lora_A/B format comes in, store A/B directly
            if comp is None:
                entry[ab] = value  # 'A' or 'B'
            else:
                entry[f"{comp}_{ab}"] = value  # 'Q_A', 'K_B', ...
        elif group == "single_proj_out":
            # Single block proj_out → split later
            entry[ab] = value  # 'A' or 'B'
        else:
            # Regular module (A/B pair)
            entry[ab] = value

    # Process each module
    applied_modules = []
    special_handled = []

    for base_key, lw in lora_dict.items():
        if base_key in special_handled:
            continue

        alpha = None
        # --- QKV / ADD_QKV ---
        if (".to_qkv" in base_key) or (".add_qkv_proj" in base_key):
            if ("A" in lw) and ("B" in lw):
                A_fused, B_fused = lw["A"], lw["B"]
                alpha = lw.get("alpha")
            else:
                A_fused, B_fused, alpha = _fuse_qkv_lora(lw)
            if A_fused is None or B_fused is None:
                logger.warning(f"Incomplete QKV LoRA at {base_key} (skip)")
                continue

            resolved_name, module = _resolve_module_name(model, base_key)
            if module is None:
                logger.warning(f"Module not found: {base_key} (resolved={resolved_name})")
                continue

            _apply_lora_to_module(module, A_fused, B_fused, strength, resolved_name, model, alpha=alpha)
            applied_modules.append(resolved_name)
            continue

        # --- single proj_out special split ---
        if base_key.endswith(".proj_out") and ("single_transformer_blocks." in base_key):
            lora_alpha = lw.get("alpha")
            split_map, consumed = _handle_proj_out_split(lora_dict, base_key, model)
            for mname, (A_part, B_part) in split_map.items():
                rname, module = _resolve_module_name(model, mname)
                if module is None:
                    logger.warning(f"proj_out split: module not found: {mname} (resolved={rname})")
                    continue
                if not (hasattr(module, "proj_down") and hasattr(module, "proj_up")):
                    logger.warning(f"proj_out split: target has no proj_down/up: {rname}")
                    continue
                _apply_lora_to_module(module, A_part, B_part, strength, rname, model, alpha=alpha)
                applied_modules.append(rname)
            special_handled.extend(consumed)
            continue

        A, B, alpha = lw.get("A"), lw.get("B"), lw.get("alpha")
        if A is None or B is None:
            logger.warning(f"Missing A or B for {base_key}")
            continue
        rname, module = _resolve_module_name(model, base_key)
        if module is None:
            logger.warning(f"Module not found: {base_key} (resolved={rname})")
            unused_keys.append(base_key)
            continue
        if hasattr(module, "proj_down") and hasattr(module, "proj_up"):
            _apply_lora_to_module(module, A, B, strength, rname, model, alpha)
            applied_modules.append(rname)

    logger.info(f"Applied LoRA to {len(applied_modules)} modules")
    if unused_keys:
        logger.warning(f"Unused keys ({len(unused_keys)}): {unused_keys[:5]}...")


def _apply_lora_to_module(
    module: nn.Module,
    A: torch.Tensor,  # [r_lora, in_features]
    B: torch.Tensor,  # [out_features, r_lora]
    strength: float,
    module_name: str,
    model: nn.Module,
    alpha: Optional[float] = None,
) -> None:
    if ".norm1.linear" in module_name or ".norm1_context.linear" in module_name:
        # Double block Adanorm is 6-way
        B = reorder_adanorm_lora_up(B, splits=6)
    elif ".single_transformer_blocks." in module_name and ".norm.linear" in module_name:
        # Single block Adanorm is 3-way
        B = reorder_adanorm_lora_up(B, splits=3)
    # --- (2) Apply LoRA scale (alpha / r) ---
    # Assume A is [r, in] (auto-correct orientation below), extract r
    r_lora = A.shape[0]
    if alpha is None:
        alpha = float(r_lora)  # When alpha not provided, typically alpha=r
    scale = strength * (alpha / max(1.0, float(r_lora)))
    B = B * scale

    # dtype/device
    A = A.to(dtype=module.proj_down.dtype, device=module.proj_down.device)
    B = (B * strength).to(dtype=module.proj_up.dtype, device=module.proj_up.device)

    # shape checks
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"{module_name}: A/B must be 2D, got {A.shape}, {B.shape}")
    if A.shape[1] != module.in_features:
        raise ValueError(f"{module_name}: A {A.shape} vs in_features={module.in_features}")
    if B.shape[0] != module.out_features:
        raise ValueError(f"{module_name}: B {B.shape} vs out_features={module.out_features}")

    pd, pu = module.proj_down.data, module.proj_up.data
    pd = unpack_lowrank_weight(pd, down=True)
    pu = unpack_lowrank_weight(pu, down=False)

    # proj_down : [rank, in] → column concat
    if pd.shape[1] == module.in_features:
        base_rank = pd.shape[0]
        new_proj_down = torch.cat([pd, A], dim=0)  # increase rank
        axis_down = 0
    # Case: [in, rank] → row concat
    elif pd.shape[0] == module.in_features:
        base_rank = pd.shape[1]
        new_proj_down = torch.cat([pd, A.T], dim=1)  # increase rank
        axis_down = 1
    else:
        raise RuntimeError(f"{module_name}: unexpected proj_down {tuple(pd.shape)}")

    # proj_up : [out, rank] → row concat
    if pu.shape[0] != module.out_features:
        raise RuntimeError(f"{module_name}: unexpected proj_up {tuple(pu.shape)}")
    new_proj_up = torch.cat([pu, B], dim=1)

    new_proj_down = pack_lowrank_weight(new_proj_down, down=True)
    new_proj_up = pack_lowrank_weight(new_proj_up, down=False)

    module.proj_down = nn.Parameter(new_proj_down, requires_grad=False)
    module.proj_up = nn.Parameter(new_proj_up, requires_grad=False)
    module.rank = base_rank + A.shape[0]

    if not hasattr(model, "_lora_slots"):
        model._lora_slots = {}
    slot = model._lora_slots.get(module_name, {"base_rank": base_rank, "appended": 0, "axis_down": axis_down})
    if "initialized" not in slot:
        slot["base_rank"] = base_rank
        slot["initialized"] = True
    slot["appended"] = slot.get("appended", 0) + A.shape[0]
    slot["axis_down"] = axis_down
    model._lora_slots[module_name] = slot


def set_lora_strength_v2(model: nn.Module, strength: float) -> None:
    if not hasattr(model, "_lora_slots") or not model._lora_slots:
        logger.warning("No LoRA weights loaded")
        return
    old = getattr(model, "_lora_strength", 1.0) or 1.0
    s = strength / old
    for name, info in model._lora_slots.items():
        m = _get_module_by_name(model, name)
        if m is None:
            continue
        base, add = info["base_rank"], info["appended"]
        if add <= 0:
            continue
        with torch.no_grad():
            m.proj_up.data[:, base : base + add] *= s
    model._lora_strength = strength


def reset_lora_v2(model: nn.Module) -> None:
    if not hasattr(model, "_lora_slots"):
        return
    for name, info in model._lora_slots.items():
        m = _get_module_by_name(model, name)
        if m is None:
            continue
        base = info["base_rank"]
        with torch.no_grad():
            if info.get("axis_down", 0) == 0:  # [rank, in]
                m.proj_down = nn.Parameter(m.proj_down[:base, :].clone(), requires_grad=False)
            else:  # [in, rank]
                m.proj_down = nn.Parameter(m.proj_down[:, :base].clone(), requires_grad=False)
            m.proj_up = nn.Parameter(m.proj_up[:, :base].clone(), requires_grad=False)
            m.rank = base
    model._lora_slots.clear()
    model._lora_strength = 1.0