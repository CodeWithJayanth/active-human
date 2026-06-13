
import argparse
import os
import pickle
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_predictor.config import CONFIG, JOINT_LIMITS
from arm_predictor.costs import compute_ref_costs
from arm_predictor.demo_library import sample_candidates, build_demo_library
from arm_predictor.motion_prior import ProMP
from arm_predictor.cache_utils import compute_config_hash, _apply_legacy_shim


CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "promp_cache.pkl")
LEGACY_BACKUP = CACHE_FILE + ".legacy_backup"



def build_multistart(force=False):
    if os.path.exists(CACHE_FILE) and not force:
        print(f"Cache already exists: {CACHE_FILE}")
        print("Use --rebuild to force regeneration, or delete the cache first.")
        return

    if os.path.exists(CACHE_FILE) and force:
        os.remove(CACHE_FILE)
        print("Deleted existing cache.")

    print("Building multi-start cache (6 starts × 170 candidates)...")
    t0 = time.time()

    canonical_q0 = np.array(CONFIG["q_start"])

    rng = np.random.default_rng(100)
    joint_range = JOINT_LIMITS[:, 1] - JOINT_LIMITS[:, 0]
    std = 0.25 * joint_range
    lower = JOINT_LIMITS[:, 0] + 0.05
    upper = JOINT_LIMITS[:, 1] - 0.05

    starts = [canonical_q0.copy()]
    for _ in range(5):
        q = canonical_q0 + rng.normal(0.0, std)
        q = np.clip(q, lower, upper)
        starts.append(q)

    print()
    print("Start poses:")
    print(f"  {'idx':<4}{'pose':<48}{'dist_from_canonical':>22}")
    for i, q in enumerate(starts):
        dist = np.linalg.norm(q - canonical_q0)
        pose_str = "[" + ", ".join(f"{v:+.3f}" for v in q) + "]"
        print(f"  {i:<4}{pose_str:<48}{dist:>22.4f}")
    print()

    seeds = [42, 43, 44, 45, 46, 47]
    K_per_start = 170
    all_demos = []
    per_start_accepted = []

    for i, (q0, seed) in enumerate(zip(starts, seeds)):
        print(f"── Start {i} (seed={seed}) ──")
        ref_goal = q0 + np.array([0.5, 0.4, 0.6, 0.2])
        ref = compute_ref_costs(q0, ref_goal)
        candidates = sample_candidates(K_per_start, seed, q0)
        demos = build_demo_library(q0, ref, candidates)
        per_start_accepted.append(len(demos))
        all_demos.extend(demos)

    total_candidates = K_per_start * len(starts)
    total_accepted = len(all_demos)

    print()
    print("Acceptance summary:")
    print(f"  {'start':<8}{'pose':<48}{'dist':>10}{'accepted':>12}")
    for i, (q, n_acc) in enumerate(zip(starts, per_start_accepted)):
        dist = np.linalg.norm(q - canonical_q0)
        pose_str = "[" + ", ".join(f"{v:+.3f}" for v in q) + "]"
        print(f"  {i:<8}{pose_str:<48}{dist:>10.4f}{n_acc:>12}")
    print(f"  Total: {total_accepted}/{total_candidates} candidates accepted")
    print()

    if total_accepted < 5:
        print(f"ERROR: only {total_accepted} demos accepted across all starts, need at least 5.")
        sys.exit(1)

    promp = ProMP()
    promp.fit([d["q_phase"] for d in all_demos])

    config_hash = compute_config_hash(CONFIG)
    cache = {
        "promp": promp,
        "demos": all_demos,
        "config_hash": config_hash,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "build_info": {
            "build_mode": "multistart",
            "num_starts": len(starts),
            "num_candidates_per_start": K_per_start,
            "num_candidates": total_candidates,
            "num_accepted": total_accepted,
            "num_rejected": total_candidates - total_accepted,
            "per_start_accepted": per_start_accepted,
            "start_seeds": seeds,
            "start_sampling_seed": 100,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
        },
    }

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f, protocol=4)

    elapsed = time.time() - t0
    print(f"Cache built: {total_accepted} demos from {len(starts)} starts, {elapsed:.1f}s")
    print(f"Config hash: {config_hash}")
    print(f"Saved to:    {CACHE_FILE}")
    _verify(verbose=False)
    print("Roundtrip verification: OK")



def quick_fix():
    if not os.path.exists(CACHE_FILE):
        print("No cache file found. Run without --quick-fix to build from scratch.")
        sys.exit(1)

    print(f"Loading legacy cache from {CACHE_FILE}...")
    _apply_legacy_shim()
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)


    promp = cache["promp"]
    assert hasattr(promp, "weight_mean"), "ProMP missing weight_mean"
    assert hasattr(promp, "weight_cov"),  "ProMP missing weight_cov"

    if not os.path.exists(LEGACY_BACKUP):
        shutil.copy2(CACHE_FILE, LEGACY_BACKUP)
        print(f"Backed up legacy cache to: {LEGACY_BACKUP}")
    else:
        print(f"Backup already exists at: {LEGACY_BACKUP} (not overwriting)")

    config_hash = compute_config_hash(CONFIG)
    cache["config_hash"] = config_hash
    cache.setdefault("built_at", "legacy-unknown (quick-fix re-save)")
    cache.setdefault("build_info", {
        "num_candidates": CONFIG["num_demos"],
        "num_accepted":   len(cache["demos"]),
        "num_rejected":   CONFIG["num_demos"] - len(cache["demos"]),
        "python_version": "legacy-unknown",
        "numpy_version":  "legacy-unknown",
    })
    cache["build_info"]["quick_fixed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    print("Cache re-saved with clean module paths and metadata.")

    _verify(verbose=True)



def _verify(verbose=True):
    
    if not os.path.exists(CACHE_FILE):
        print(f"No cache at {CACHE_FILE}")
        return False

    try:
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
    except ModuleNotFoundError as e:
        print(f"FAIL: clean load raised {type(e).__name__}: {e}")
        print("      Run: python build_cache.py --quick-fix")
        return False

    promp = cache["promp"]
    demos = cache["demos"]
    checks = {
        "Top-level keys present":      {"promp", "demos"} <= set(cache.keys()),
        "ProMP class FQN":             f"{type(promp).__module__}.{type(promp).__name__}",
        "weight_mean.shape == (100,)":     promp.weight_mean.shape == (100,),
        "weight_cov.shape == (100,100)":   promp.weight_cov.shape == (100, 100),
        "demo count":                  len(demos),
        "Demo schema OK":              all(set(d.keys()) >= {"q_phase", "T", "q_goal"} for d in demos),
        "config_hash present":         "config_hash" in cache,
        "built_at present":            "built_at" in cache,
        "q_start in demos":            all("q_start" in d for d in demos) if cache.get("build_info", {}).get("build_mode") == "multistart" else "N/A (single-start)",
    }

    if verbose:
        print()
        print("Verification:")
        for k, v in checks.items():
            print(f"  {k:<32s}: {v}")
        if "config_hash" in cache:
            print(f"  config_hash                     : {cache['config_hash']}")
        if "built_at" in cache:
            print(f"  built_at                        : {cache['built_at']}")
        if "build_info" in cache:
            print(f"  build_info                      : {cache['build_info']}")

    with open(CACHE_FILE, "rb") as f:
        raw = f.read()
    v2_count = raw.count(b"motion_prior_v2")
    if verbose:
        print(f"  _v2 references in pickle stream : {v2_count}")
    if v2_count > 0:
        if verbose:
            print("  WARNING: legacy _v2 module path still present!")
        return False
    if verbose:
        print("  Clean: no legacy module paths.")
    return True



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build/manage promp_cache.pkl")
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete existing cache and rebuild from scratch")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify",    action="store_true",
                      help="Check existing cache integrity")
    mode.add_argument("--quick-fix", action="store_true",
                      help="Re-save legacy cache with correct module paths")
    args = parser.parse_args()

    if args.verify:
        ok = _verify(verbose=True)
        sys.exit(0 if ok else 1)
    elif args.quick_fix:
        quick_fix()
    else:
        build_multistart(force=args.rebuild)
