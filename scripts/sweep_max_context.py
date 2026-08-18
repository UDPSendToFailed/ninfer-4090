#!/usr/bin/env python3
"""
NInfer-4090 Max-Context Ceiling Sweeper
Binary-searches the exact physical VRAM allocation limit across all KV modes,
MTP speculation windows, and Vision configurations.
"""

import sys
import time
import subprocess
import re
from pathlib import Path

# Paths
SERVER_EXE = Path("./build-ninja/apps/ninfer-serve.exe")
MODEL_PATH = Path("../LLM/qwen3_8_27b.ninfer")

# Sweep Grid Configurations
KV_DTYPES = ["rk2v4-e8", "rk4v4-e8", "rk4v4", "rk8v4", "int8"]
SPEC_CONFIGS = [
    {"name": "No-Spec (MTP0)", "args": []},
    {"name": "MTP3",           "args": ["--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft"]},
    {"name": "MTP4",           "args": ["--spec", "mtp", "--draft-tokens", "4", "--lm-head-draft"]},
]
VISION_CONFIGS = [
    {"name": "Text-Only", "args": []},
    {"name": "Vision (8k Default)", "args": ["--vision"]},
    {"name": "Vision (4k Small)",   "args": ["--vision", "--vision-max-tokens", "4096"]},
]

PAGE_SIZE = 64
SEARCH_GRANULARITY = 1024  # Settle within 1,024 tokens (~16 pages)
MIN_CONTEXT = 32768
MAX_CONTEXT_CEILING = 600000

def test_startup_context(kv_dtype, spec_args, vision_args, test_tokens):
    """
    Spawns ninfer-serve with the target context and checks if it reaches listening state.
    Returns: (success: bool, resolved_tokens: int, runtime_str: str, slack_str: str)
    """
    cmd = [
        str(SERVER_EXE),
        str(MODEL_PATH),
        "--kv-dtype", kv_dtype,
        "--max-context", str(test_tokens),
        "--prefill-chunk", "1024",
        "--preserve-thinking"
    ] + spec_args + vision_args

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    success = False
    resolved_tokens = 0
    runtime_str = "n/a"
    slack_str = "n/a"

    try:
        t0 = time.time()
        for line in iter(proc.stdout.readline, ''):
            # Check for success
            if "KV capacity explicit resolved=" in line:
                m = re.search(r"resolved=(\d+).*?runtime=([^\s]+).*?slack=([^\s]+)", line)
                if m:
                    resolved_tokens = int(m.group(1))
                    runtime_str = m.group(2)
                    slack_str = m.group(3)

            if "listening on http" in line:
                success = True
                break

            # Check for out-of-memory failure
            if "requires" in line and "available" in line:
                success = False
                break
            if "error" in line.lower() and ("memory" in line.lower() or "allocation" in line.lower()):
                success = False
                break

            # Timeout if stuck over 25 seconds
            if time.time() - t0 > 25.0:
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    return success, resolved_tokens, runtime_str, slack_str


def find_max_context_for_profile(kv_dtype, spec_cfg, vision_cfg):
    """Binary searches the maximum context between MIN_CONTEXT and MAX_CONTEXT_CEILING."""
    low = MIN_CONTEXT
    high = MAX_CONTEXT_CEILING
    best_tokens = 0
    best_runtime = "n/a"
    best_slack = "n/a"

    print(f"\n>>> Sweeping: KV={kv_dtype:<8} | Spec={spec_cfg['name']:<14} | Mode={vision_cfg['name']}")

    # Quick sanity check on lower bound
    passed_min, res_tok, rt, sl = test_startup_context(kv_dtype, spec_cfg["args"], vision_cfg["args"], low)
    if not passed_min:
        print(f"    [FAIL] Could not even boot at minimum {low} tokens.")
        return 0, "n/a", "n/a"

    best_tokens = res_tok if res_tok > 0 else low
    best_runtime = rt
    best_slack = sl

    while (high - low) > SEARCH_GRANULARITY:
        mid = ((low + high) // 2 // PAGE_SIZE) * PAGE_SIZE
        print(f"    Testing context: {mid:,} tokens...", end="", flush=True)

        passed, res_tok, rt, sl = test_startup_context(kv_dtype, spec_cfg["args"], vision_cfg["args"], mid)
        if passed:
            print(f" [PASSED] (Runtime: {rt}, Slack: {sl})")
            best_tokens = res_tok if res_tok > 0 else mid
            best_runtime = rt
            best_slack = sl
            low = mid
        else:
            print(" [FAILED/OOM]")
            high = mid

    print(f"  ==> Verified Max: {best_tokens:,} tokens (Runtime: {best_runtime}, Slack: {best_slack})")
    return best_tokens, best_runtime, best_slack


def main():
    if not SERVER_EXE.exists():
        print(f"[ERROR] Cannot find server binary at {SERVER_EXE}")
        sys.exit(1)
    if not MODEL_PATH.exists():
        print(f"[ERROR] Cannot find model artifact at {MODEL_PATH}")
        sys.exit(1)

    print("=" * 80)
    print("   NInfer-4090: Automated Context Window Ceiling Sweep")
    print("   Target: RTX 4090 (24 GB) / Direct3D 12 WDDM Residency Mode")
    print("=" * 80)

    results = []

    for vision_cfg in VISION_CONFIGS:
        for spec_cfg in SPEC_CONFIGS:
            for kv_dtype in KV_DTYPES:
                max_tok, rt, sl = find_max_context_for_profile(kv_dtype, spec_cfg, vision_cfg)
                results.append({
                    "vision": vision_cfg["name"],
                    "spec": spec_cfg["name"],
                    "kv_dtype": kv_dtype,
                    "max_tokens": max_tok,
                    "runtime": rt,
                    "slack": sl
                })

    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY REPORT: Maximum Verified Context Ceilings (24 GB RTX 4090)")
    print("=" * 80)
    print(f"| {'Vision Mode':<20} | {'Speculation':<16} | {'KV Dtype':<10} | {'Max Context':<14} | {'Runtime':<10} | {'Slack':<10} |")
    print(f"|{'-'*22}|{'-'*18}|{'-'*12}|{'-'*16}|{'-'*12}|{'-'*12}|")

    for r in results:
        tok_str = f"{r['max_tokens']:,} tok" if r['max_tokens'] > 0 else "FAILED"
        print(f"| {r['vision']:<20} | {r['spec']:<16} | {r['kv_dtype']:<10} | {tok_str:<14} | {r['runtime']:<10} | {r['slack']:<10} |")

if __name__ == "__main__":
    main()