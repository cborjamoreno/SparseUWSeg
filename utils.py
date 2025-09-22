"""
Merged utility file: contains all helpers from multi_scale_utils.py and seed_utils.py
---
multi_scale_utils.py: Downscaling, coverage map, mask merging, etc.
seed_utils.py: Global seed setting for reproducibility.
---
"""

# --- multi_scale_utils.py ---
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Literal
import numpy as np
import cv2

ScalePolicy = Literal['auto', 'fixed']
ReplacePolicy = Literal['bg_only', 'always']

@dataclass
class ScaleConfig:
    policy: ScalePolicy = 'auto'
    fixed_scale: float = 0.5          # used if policy == 'fixed'
    target_min_object_px: int = 4     # desired smallest object after scaling
    estimated_min_object_px: Optional[int] = None  # if you can estimate per dataset
    min_scale: float = 0.35
    max_scale: float = 0.75
    min_short_edge_trigger: int = 800  # skip downscale if short edge < this

def choose_scale(shape: Tuple[int, int, int], cfg: ScaleConfig) -> float:
    h, w = shape[:2]
    short_edge = min(h, w)
    if cfg.policy == 'fixed':
        return float(np.clip(cfg.fixed_scale, 0.1, 1.0))
    if short_edge < cfg.min_short_edge_trigger:
        return 1.0
    if cfg.estimated_min_object_px is None:
        est_min = 12
    else:
        est_min = max(1, cfg.estimated_min_object_px)
    raw_scale = cfg.target_min_object_px / float(est_min)
    scale = float(np.clip(raw_scale, cfg.min_scale, cfg.max_scale))
    return min(scale, 1.0)

def downscale_image(img: np.ndarray, scale: float, interpolation=cv2.INTER_AREA) -> np.ndarray:
    if scale >= 0.999:
        return img
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)

def map_points_to_high(points_low: Iterable[Tuple[int, int]], scale: float, full_shape: Tuple[int, int, int]) -> List[Tuple[int, int]]:
    if scale >= 0.999:
        h, w = full_shape[:2]
        return [(_clamp_int(x, 0, w-1), _clamp_int(y, 0, h-1)) for x, y in points_low]
    inv = 1.0 / scale
    h, w = full_shape[:2]
    mapped = []
    for x, y in points_low:
        X = int(round(x * inv))
        Y = int(round(y * inv))
        X = _clamp_int(X, 0, w - 1)
        Y = _clamp_int(Y, 0, h - 1)
        mapped.append((X, Y))
    return mapped

def _clamp_int(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v

def init_coverage_map(low_shape: Tuple[int, int]) -> np.ndarray:
    return np.zeros(low_shape, dtype=np.uint16)

def update_coverage_map(coverage: np.ndarray, new_mask_full: np.ndarray, scale: float, bg_value: int = 0) -> None:
    if scale >= 0.999:
        fg = (new_mask_full != bg_value).astype(np.uint8)
        coverage[...] = coverage + fg.astype(np.uint16)
        return
    fg_full = (new_mask_full != bg_value).astype(np.uint8)
    h_lr = max(1, int(round(new_mask_full.shape[0] * scale)))
    w_lr = max(1, int(round(new_mask_full.shape[1] * scale)))
    fg_low = cv2.resize(fg_full, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
    present = (fg_low > 0).astype(np.uint16)
    coverage[...] = coverage + present

def propose_points_low(coverage: np.ndarray, k: int, min_dist: int = 0) -> List[Tuple[int, int]]:
    h, w = coverage.shape
    flat = coverage.ravel()
    if k >= flat.size:
        coords = [(i % w, i // w) for i in range(flat.size)]
    else:
        idx = np.argpartition(flat, k)[:k]
        coords = [(int(i % w), int(i // w)) for i in idx]
    if min_dist <= 0:
        return coords
    selected: List[Tuple[int, int]] = []
    for x, y in coords:
        if all(max(abs(x - sx), abs(y - sy)) >= min_dist for sx, sy in selected):
            selected.append((x, y))
        if len(selected) == k:
            break
    return selected

def vectorized_merge(
    base_mask: np.ndarray,
    new_mask: np.ndarray,
    background_value: int = 0,
    replace_policy: ReplacePolicy = 'bg_only'
) -> Tuple[np.ndarray, int]:
    if base_mask.shape != new_mask.shape:
        raise ValueError('Shape mismatch in vectorized_merge')
    if base_mask.dtype != new_mask.dtype:
        raise ValueError('Dtype mismatch in vectorized_merge')
    new_fg = new_mask != background_value
    if replace_policy == 'bg_only':
        can_write = (base_mask == background_value) & new_fg
    elif replace_policy == 'always':
        can_write = new_fg & (base_mask != new_mask)
    else:
        raise ValueError(f'Unknown replace_policy {replace_policy}')
    changed = int(can_write.sum())
    if changed:
        base_mask[can_write] = new_mask[can_write]
    return base_mask, changed

def test_equivalence_pixel_loop(base_mask: np.ndarray, new_mask: np.ndarray, background_value: int = 0, replace_policy: ReplacePolicy = 'bg_only') -> bool:
    naive = base_mask.copy()
    h, w = naive.shape[:2]
    for y in range(h):
        for x in range(w):
            b = naive[y, x]
            n = new_mask[y, x]
            if n != background_value:
                if replace_policy == 'bg_only':
                    if b == background_value:
                        naive[y, x] = n
                else:
                    if n != b:
                        naive[y, x] = n
    vec = base_mask.copy()
    vectorized_merge(vec, new_mask, background_value, replace_policy)
    return np.array_equal(naive, vec)

# --- seed_utils.py ---
import os
import random
import numpy as np
try:
    import torch
except Exception:
    torch = None
_APPLIED = False
def set_global_seed(seed: int | None = 42, deterministic: bool = True, verbose: bool = True, strict: bool = True) -> int:
    global _APPLIED
    if seed is None:
        env_seed = os.getenv("GLOBAL_SEED")
        seed = int(env_seed) if env_seed is not None else 42
    if _APPLIED:
        return int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            if deterministic and torch.cuda.is_available():
                if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.benchmark = False
                if hasattr(torch.backends.cudnn, 'deterministic'):
                    torch.backends.cudnn.deterministic = True
                if strict and hasattr(torch, 'use_deterministic_algorithms'):
                    try:
                        torch.use_deterministic_algorithms(True)
                    except RuntimeError as e:
                        if verbose:
                            print(f"[seed_utils] Strict deterministic algorithms failed: {e}\n"
                                  f"            Falling back to soft-deterministic mode (reproducibility usually sufficient).\n"
                                  f"            To enable strict mode, set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching.")
                        strict = False
                    except Exception:
                        strict = False
        except Exception:
            pass
    _APPLIED = True
    if verbose:
        mode = "strict" if (deterministic and strict) else ("soft" if deterministic else "off")
        ws = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<unset>")
        print(f"[seed_utils] Global seed set to {seed} (mode={mode}, CUBLAS_WORKSPACE_CONFIG={ws})")
    return int(seed)

__all__ = ["set_global_seed"]
