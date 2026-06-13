

import hashlib
import json
import os
import pickle
import sys

import numpy as np

from arm_predictor.config import JOINT_LIMITS


_HASHED_CONFIG_KEYS = (
    "num_demos",
    "demo_seed",
    "solve_steps",
    "max_displacement",
    "min_displacement",
    "style_ranges",
    "num_basis",
    "basis_width",
    "phase_points",
    "q_start",
)


def _normalize(obj):
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return [_normalize(x) for x in obj]
    if isinstance(obj, list):
        return [_normalize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    return obj


def compute_config_hash(config):
    
    payload = {k: _normalize(config[k]) for k in _HASHED_CONFIG_KEYS}
    payload["JOINT_LIMITS"] = _normalize(JOINT_LIMITS)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _apply_legacy_shim():
    
    import arm_predictor.motion_prior as _mp
    sys.modules.setdefault("arm_predictor.motion_prior_v2", _mp)
    sys.modules["__main__"].ProMP = _mp.ProMP


def load_cache(cache_path, config, allow_stale=False):
   
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cache not found at {cache_path}. "
            f"Run: python build_cache.py"
        )

    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    except ModuleNotFoundError as e:
        if "motion_prior_v2" not in str(e):
            raise
        _apply_legacy_shim()
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print("WARNING: Legacy cache detected (motion_prior_v2 module path).")
        print("         Run: python build_cache.py --quick-fix")

    expected = compute_config_hash(config)
    actual = cache.get("config_hash")
    if actual is None:
        print("WARNING: Legacy cache without config hash.")
        print("         Run: python build_cache.py --rebuild")
    elif actual != expected:
        msg = (
            "Cache is stale — config changed since cache was built.\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            "Run: python build_cache.py --rebuild"
        )
        if allow_stale:
            print(f"WARNING: {msg}")
        else:
            raise RuntimeError(msg)

    return cache
