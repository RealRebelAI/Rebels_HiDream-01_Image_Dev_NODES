"""
lora_stack.py — LoRA merge/unmerge for HiDream-01 (Qwen3VL backbone).

HiDream-01 isn't a diffusers pipeline, so we can't lean on diffusers'
`load_lora_weights`. This module handles raw safetensors LoRA files
(kohya / PEFT naming variants) by merging directly into Linear weights,
with a state tracker so the stack can be swapped without reloading
the 16GB base model.

Memory note: deltas are stored on CPU for unmerge, never duplicated on GPU.
"""

import torch
from safetensors.torch import load_file


STATE_ATTR = "_rebel_lora_stack_state"


# ---------------------------------------------------------------------------
# Key parsing — handle kohya, PEFT, and diffusers-style names
# ---------------------------------------------------------------------------

_PREFIX_STRIPS = (
    "base_model.model.",
    "lora_unet_",
    "lora_te_",
    "model.diffusion_model.",
    "transformer.",
    "diffusion.",
)

def _strip_prefix(key):
    for p in _PREFIX_STRIPS:
        if key.startswith(p):
            return key[len(p):]
    return key


def _normalize_module_path(key):
    """Strip known prefixes but preserve the rest as-is.
    We no longer blindly replace underscores with dots — that breaks
    submodule names like final_layer2, language_model, down_proj, etc."""
    return _strip_prefix(key)


# ---------------------------------------------------------------------------
# Pair extraction from LoRA state dict
# ---------------------------------------------------------------------------

def _extract_lora_pairs(lora_sd):
    """Group LoRA tensors by target module. Returns {module_path: {A, B, alpha}}."""
    pairs = {}

    def _put(base_key, slot, value):
        clean = _normalize_module_path(base_key)
        pairs.setdefault(clean, {})[slot] = value

    for key, tensor in lora_sd.items():
        if key.endswith(".lora_A.weight") or key.endswith(".lora_down.weight"):
            base = key.rsplit(".lora_", 1)[0]
            _put(base, "A", tensor)
        elif key.endswith(".lora_B.weight") or key.endswith(".lora_up.weight"):
            base = key.rsplit(".lora_", 1)[0]
            _put(base, "B", tensor)
        elif key.endswith(".alpha"):
            base = key[: -len(".alpha")]
            alpha_val = tensor.item() if tensor.numel() == 1 else float(tensor.flatten()[0])
            _put(base, "alpha", alpha_val)

    return pairs


# ---------------------------------------------------------------------------
# Find a Linear submodule by dotted path
# ---------------------------------------------------------------------------

def _find_linear(model, dotted_path):
    """Return the nn.Linear at dotted_path, or None if not found / not Linear."""
    try:
        mod = model.get_submodule(dotted_path)
    except (AttributeError, ValueError):
        return None
    if isinstance(mod, torch.nn.Linear):
        return mod
    if hasattr(mod, "base_layer") and isinstance(mod.base_layer, torch.nn.Linear):
        return mod.base_layer
    return None


# ---------------------------------------------------------------------------
# Underscore-flattened reverse lookup for kohya-style key matching
# ---------------------------------------------------------------------------

def _build_underscore_map(model):
    """Build {underscored_flat_path: real_dotted_path} for all Linear modules.
    This lets us match kohya-style keys (all underscores) against the model's
    actual dotted submodule paths without guessing which underscores are
    separators vs part of names like final_layer2 or down_proj."""
    result = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            flat = name.replace(".", "_")
            result[flat] = name
    return result


# ---------------------------------------------------------------------------
# Core merge / unmerge
# ---------------------------------------------------------------------------

_FLAT_PREFIX_STRIPS = (
    "diffusion_",
    "base_model_model_",
    "transformer_",
    "lora_unet_",
    "lora_te_",
    "model_diffusion_model_",
)


def merge_lora(model, lora_sd, strength):
    """Merge a single LoRA into the model in-place. Returns list of
    (dotted_path, delta_cpu) tuples for later unmerge."""
    if abs(strength) < 1e-8:
        return []

    pairs = _extract_lora_pairs(lora_sd)
    underscore_map = _build_underscore_map(model)
    applied = []
    matched = 0
    skipped_meta = 0
    unmatched_examples = []

    for path, parts in pairs.items():
        if "A" not in parts or "B" not in parts:
            continue

        # Strategy 1: try the normalized dotted path directly (PEFT/diffusers style)
        linear = _find_linear(model, path)
        resolved_path = path

        # Strategy 2: flatten to underscores and match against model tree (kohya style)
        if linear is None:
            flat = path.replace(".", "_")
            for prefix in _FLAT_PREFIX_STRIPS:
                if flat.startswith(prefix):
                    flat = flat[len(prefix):]
                    break
            if flat in underscore_map:
                resolved_path = underscore_map[flat]
                linear = _find_linear(model, resolved_path)

        # Strategy 3: try without any prefix stripping (raw path in underscore map)
        if linear is None:
            raw_flat = path.replace(".", "_")
            if raw_flat in underscore_map:
                resolved_path = underscore_map[raw_flat]
                linear = _find_linear(model, resolved_path)

        if linear is None:
            if len(unmatched_examples) < 5:
                unmatched_examples.append(path)
            continue

        W = linear.weight

        # Skip disk-offloaded layers — accelerate puts them on meta device
        # with no actual data, so we can't merge into them
        if W.device.type == "meta":
            skipped_meta += 1
            continue

        A = parts["A"].to(W.device, W.dtype)
        B = parts["B"].to(W.device, W.dtype)
        rank = A.shape[0]
        alpha = parts.get("alpha", float(rank))
        scale = (alpha / rank) * strength

        delta = (B @ A) * scale
        if delta.shape != W.shape:
            continue

        with torch.no_grad():
            W.add_(delta)
        applied.append((resolved_path, delta.detach().to("cpu")))
        matched += 1

    print(f"[Rebels_HiDream_01] LoRA merge: {matched}/{len(pairs)} pairs applied "
          f"(strength={strength:.3f})")
    if skipped_meta > 0:
        print(f"[Rebels_HiDream_01] Skipped {skipped_meta} meta-device (disk-offloaded) layers")
    if matched == 0 and unmatched_examples:
        print(f"[Rebels_HiDream_01] No matches found. Example unmatched paths: "
              f"{unmatched_examples}")
    elif unmatched_examples:
        print(f"[Rebels_HiDream_01] {len(unmatched_examples)} unmatched paths "
              f"(first few): {unmatched_examples}")

    return applied


def unmerge_lora(model, applied):
    """Reverse a previously applied merge. `applied` is the list returned by merge_lora."""
    for path, delta_cpu in applied:
        linear = _find_linear(model, path)
        if linear is None:
            continue
        if linear.weight.device.type == "meta":
            continue
        with torch.no_grad():
            linear.weight.sub_(delta_cpu.to(linear.weight.device, linear.weight.dtype))


# ---------------------------------------------------------------------------
# Stack-level helpers (used by the LoRA Stack Injector node)
# ---------------------------------------------------------------------------

def stack_fingerprint(stack):
    """Hashable identity of a LoRA stack — used to detect re-application.
    stack: list of (lora_path, strength, bypass) tuples."""
    return tuple(
        (path, round(float(s), 6), bool(b))
        for path, s, b in stack
    )


def apply_stack(model, stack):
    """Apply a full stack. Returns list-of-list of applied deltas, one per slot.
    Bypassed slots contribute an empty list."""
    per_slot = []
    for path, strength, bypass in stack:
        if bypass or not path or abs(strength) < 1e-8:
            per_slot.append([])
            continue
        try:
            sd = load_file(path)
        except Exception as e:
            print(f"[Rebels_HiDream_01] LoRA load failed: {path}: {e}")
            per_slot.append([])
            continue
        per_slot.append(merge_lora(model, sd, strength))
    return per_slot


def unmerge_stack(model, per_slot_applied):
    """Reverse a previously-applied stack (in reverse order for numerical stability)."""
    for applied in reversed(per_slot_applied):
        unmerge_lora(model, applied)
    for key, tensor in lora_sd.items():
        if key.endswith(".lora_A.weight") or key.endswith(".lora_down.weight"):
            base = key.rsplit(".lora_", 1)[0]
            _put(base, "A", tensor)
        elif key.endswith(".lora_B.weight") or key.endswith(".lora_up.weight"):
            base = key.rsplit(".lora_", 1)[0]
            _put(base, "B", tensor)
        elif key.endswith(".alpha"):
            base = key[: -len(".alpha")]
            alpha_val = tensor.item() if tensor.numel() == 1 else float(tensor.flatten()[0])
            _put(base, "alpha", alpha_val)

    return pairs


# ---------------------------------------------------------------------------
# Find a Linear submodule by dotted path, tolerant of common renames
# ---------------------------------------------------------------------------

def _find_linear(model, dotted_path):
    """Return the nn.Linear at dotted_path, or None if not found / not Linear."""
    try:
        mod = model.get_submodule(dotted_path)
    except (AttributeError, ValueError):
        return None
    if isinstance(mod, torch.nn.Linear):
        return mod
    # Some weights live under .base_layer (PEFT-wrapped, edge case)
    if hasattr(mod, "base_layer") and isinstance(mod.base_layer, torch.nn.Linear):
        return mod.base_layer
    return None


# ---------------------------------------------------------------------------
# Core merge / unmerge
# ---------------------------------------------------------------------------

def merge_lora(model, lora_sd, strength):
    """Merge a single LoRA into the model in-place. Returns list of
    (dotted_path, delta_cpu) tuples for later unmerge."""
    if abs(strength) < 1e-8:
        return []

    pairs = _extract_lora_pairs(lora_sd)
    applied = []
    matched = 0
    unmatched_examples = []

    for path, parts in pairs.items():
        if "A" not in parts or "B" not in parts:
            continue
        linear = _find_linear(model, path)
        if linear is None:
            if len(unmatched_examples) < 3:
                unmatched_examples.append(path)
            continue

        W = linear.weight  # [out, in]
        A = parts["A"].to(W.device, W.dtype)   # [r, in]
        B = parts["B"].to(W.device, W.dtype)   # [out, r]
        rank = A.shape[0]
        alpha = parts.get("alpha", float(rank))
        scale = (alpha / rank) * strength

        delta = (B @ A) * scale  # [out, in]
        if delta.shape != W.shape:
            # Shape mismatch — skip this entry rather than crash
            continue

        with torch.no_grad():
            W.add_(delta)
        applied.append((path, delta.detach().to("cpu")))
        matched += 1

    print(f"[Rebels_HiDream_01] LoRA merge: {matched}/{len(pairs)} pairs applied "
          f"(strength={strength:.3f})")
    if matched == 0 and unmatched_examples:
        print(f"[Rebels_HiDream_01] No matches found. Example unmatched paths: "
              f"{unmatched_examples}")

    return applied


def unmerge_lora(model, applied):
    """Reverse a previously applied merge. `applied` is the list returned by merge_lora."""
    for path, delta_cpu in applied:
        linear = _find_linear(model, path)
        if linear is None:
            continue
        with torch.no_grad():
            linear.weight.sub_(delta_cpu.to(linear.weight.device, linear.weight.dtype))


# ---------------------------------------------------------------------------
# Stack-level helpers (used by the LoRA Stack Injector node)
# ---------------------------------------------------------------------------

def stack_fingerprint(stack):
    """Hashable identity of a LoRA stack — used to detect re-application.
    stack: list of (lora_path, strength, bypass) tuples."""
    return tuple(
        (path, round(float(s), 6), bool(b))
        for path, s, b in stack
    )


def apply_stack(model, stack):
    """Apply a full stack. Returns list-of-list of applied deltas, one per slot.
    Bypassed slots contribute an empty list."""
    per_slot = []
    for path, strength, bypass in stack:
        if bypass or not path or abs(strength) < 1e-8:
            per_slot.append([])
            continue
        try:
            sd = load_file(path)
        except Exception as e:
            print(f"[Rebels_HiDream_01] LoRA load failed: {path}: {e}")
            per_slot.append([])
            continue
        per_slot.append(merge_lora(model, sd, strength))
    return per_slot


def unmerge_stack(model, per_slot_applied):
    """Reverse a previously-applied stack (in reverse order for numerical stability)."""
    for applied in reversed(per_slot_applied):
        unmerge_lora(model, applied)
