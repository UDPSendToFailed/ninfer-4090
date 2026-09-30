# Quantized small-T attention synchronization repair

Base: `221290ba`; branch: `fix/i8-small-t-attn-sync`. User authorizes two
CTA barriers in `gqa_attention_decode_i8.cuh`, regression tests, measurements,
and merging only into `feat/kvmem`. Q5 residual accumulation stays unchanged.
Existing dense dispatch and other kernels stay unchanged.

## Implementation and verification sequence

- [x] Add an independent FP64 oracle and deterministic encoded-cache fixture.
- [x] Register Qwen27B (24 Q heads / 4 KV heads), T=1/2/4, 512 keys and a
  length crossing the small-T split boundary; cover INT8 and rk4v4-e8.
  Require 64 bitwise-equal repeats and direct comparison with the FP64 oracle.
- [x] Build/run the unchanged kernel to establish the regression failure;
  measure dense decode and MTP-3 at 32K/128K/262K before repair.
- [x] Add the two unconditional CTA barriers: after Q/scale zero initialization;
  after all producer warps have read scale registers, before shared alias reuse.
- [x] Audit other small-T/prefill kernels; record suspected patterns. Additional
  repairs require racecheck evidence and separate user authorization.
- [x] Run regression and compute-sanitizer 13.0.85 racecheck, synccheck,
  initcheck; preserve commands/results outside the checkout.
- [x] Measure the same six dense workloads after repair; full build and CTest.
- [ ] Commit `fix(ops): ...`, push this branch, merge into feat/kvmem and push.
- [ ] On feat/kvmem preserve old baseline and rerun its nine dense cases;
  temporarily export last-position prefill logits for a 512-token checkpoint
  case, MTP off twice. Revert all diagnostic code before further commits.

## Following approved work on feat/kvmem

Transfer engine: independent H2D and D2H streams, independent writeback work;
pageable archive copies use the second worker (D15 limit two). Pinned archive
completion is event/callback driven. Preserve read-before-overwrite and staging
consumer ordering across the separated streams. A delayed producer must not
hold up a later unrelated prefetch; test and record both completion times.

Attention step 3: new partial prefill and small-T APIs read original logical /
physical page pairs and effective-key prefix sums; BF16, INT8 and rk4v4-e8;
FP32 unnormalized (O,m,l), fixed-order merge. Test distant/gapped pages, partial
tails, causality, split edges, empty splits and multi-pass FP64 equivalence;
repeat bitwise, racecheck and 264 MiB single-layer compute benchmark. Commit
each component separately and stop before runtime integration.

Evidence root: `D:\deeplearning\NInfer\logs\kvmem-stage3-attention`.

## Initial code audit

`gqa_attention_decode.cu` dispatches ordinary INT8, packed V/rotated K,
packed K/V, rk4v4-e8 lattice and E8 root to the same
`gqa_attention_decode_i8_tiled_kernel` body. Both requested barriers are
unconditional for every non-neutral CTA; neither is inside the producer-warp
conditional. Uniform neutral/invalid split exits precede Q initialization.

The BF16 small-T kernel stages mutually exclusive valid/zero K/V destinations,
then executes `cp_wait<0>()` and a CTA barrier before loading fragments. Its
per-warp probability storage does not alias Q scale storage. INT8 prefill
allocates separate Q scales and probability tiles, assigns each row/group to
one quantizing warp and executes a CTA barrier after Q quantization. BF16
prefill Q/K padding writes and asynchronous copies use disjoint tile cells;
its consumers wait for the copy group and CTA barrier. No additional instance
of the two diagnosed patterns was found in this audit. This is source review,
not a claim that every attention configuration has been sanitizer-qualified.
No other dense kernel or dispatcher is changed.

The regression uses exact FP16 scales, independently decoded signed code bytes,
and an independent FP64 inverse H64 for packed/rotated cache values. Packed K
8-vectors lie in the integral even-sum D8 coset. The oracle uses original BF16 Q
and FP64 dot/softmax/value reduction, never the GPU output as a reference.
INT8 retains the existing registered numerical criterion (relative L2 3.15e-3);
rotated output allows two BF16 rounding boundaries (6e-3). These criteria are
fixed before running the regression. Each default invocation compares 64
identical-input calls bitwise. `--sanitizer` uses two repeats of the full shape
matrix to avoid spending instrumentation time on duplicate launches; the
64-repeat CTest remains mandatory.

## Unmodified-kernel evidence (2026-10-01)

The original regression exits 1: rk4v4-e8 at 8199 keys differs bitwise on the
second invocation for T=1,2,4. The other nine shape/format combinations in this
invocation were bitwise stable; this does not prove absence of races.
Compute-sanitizer 13.0.85 racecheck, the same full matrix with two repeats,
exits 99 and reports **64 displayed hazards (64 errors, 0 warnings)**. Under
instrumentation some rk4v4-e8 cases also fail the fixed FP64 criterion.
Files: `sync-before-regression.log`, `sync-before-racecheck.log`, and
`sync-before-racecheck.stdout.txt`; saved old executable:
`sync-before-regression.exe`.

Six unmodified-kernel process runs, fixed stage-0 synthetic prompts, 64 greedy
tokens, rk4v4-e8, default CUDA Graph, no vision; no intentional extra desktop
GPU load. Stage-0 prompts reserve exactly 64 output tokens in each context.
Measurements use the CLI's decode-only speed, not total prefill/load wall time.

| Context | Decode tok/s | MTP-3 tok/s | MTP acceptance |
|---|---:|---:|---:|
| 32768 | 48.54 | 154.19 | 100% |
| 131072 | 44.02 | 137.63 | 100% |
| 262144 | 39.17 | 120.05 | 100% |

Commands, base commit, executable/kernel/prompt hashes, GPU memory before/after
and token IDs are recorded in `before/*.json`. Source was based on `221290ba`
with test/document additions; production kernels were unmodified at build
time. The final old-binary run remained isolated in `before/ninfer.exe` while
the two barrier source edits were prepared; compilation waited for its finish.

## Repaired-kernel verification (2026-10-01)

All 12 cases pass the independent FP64 criterion and every 64-repeat result is
bitwise equal. INT8 relative L2 is 0.00279735–0.00285466; rk4v4-e8 is
0.00301066–0.00319963. The fixed input and numerical criteria are unchanged
from the failing test. See `sync-after-regression.log` (full numerical stats).

Full build exits 0; the subsequent full CTest has **97 tests: 93 pass,
4 unavailable model artifacts skipped, 0 failures**, 62.58 s.
Logs: `sync-full-build.log`, `ctest-sync-fix.log`.

Compute-sanitizer **13.0.85**, no kernel filter, two repeats per case across the
entire matrix: racecheck **0 displayed hazards / 0 errors / 0 warnings**;
synccheck **0 errors**; initcheck **0 errors**. Each process exits 0, and the
FP64/repeat checks also pass under instrumentation. Logs are
`sync-after-{racecheck,synccheck,initcheck}.{log,stdout.txt}`.

Reproducible PowerShell command, from the repository root (runtime DLL directory
must be on PATH):

```powershell
$san = 'D:\deeplearning\NInfer\logs\kvmem-stage0-baseline\sanitizer-tools\cuda_sanitizer_api-windows-x86_64-13.0.85-archive\compute-sanitizer\compute-sanitizer.exe'
$data = 'D:\deeplearning\NInfer\logs\kvmem-stage3-attention'
$env:NINFER_OP_REPORT_STATS = '1'
foreach ($check in @('racecheck', 'synccheck', 'initcheck')) {
    & $san --tool $check --error-exitcode 99 `
        --log-file "$data\sync-after-$check.log" `
        '.\build-vision-integration\tests\ninfer_gqa_small_t_sync_test.exe' `
        --sanitizer *> "$data\sync-after-$check.stdout.txt"
    if ($LASTEXITCODE -ne 0) { throw "$check failed: $LASTEXITCODE" }
}
```

## Before/after performance

One process per configuration, identical prompts and flags as above. Both use
rk4v4-e8, greedy 64-token generation, and CUDA Graph; no vision. This is a
single-sample regression check, not a statistical performance estimate.

| Context / mode | Before tok/s | After tok/s | Change |
|---|---:|---:|---:|
| 32768 decode | 48.54 | 48.61 | +0.144% |
| 32768 MTP-3 | 154.19 | 153.87 | -0.208% |
| 131072 decode | 44.02 | 44.00 | -0.045% |
| 131072 MTP-3 | 137.63 | 137.55 | -0.058% |
| 262144 decode | 39.17 | 39.16 | -0.026% |
| 262144 MTP-3 | 120.05 | 112.71 | -6.114% |

262K MTP acceptance changes from 100% (47/47) to 93.88% (46/49);
verification rounds increase from 16 to 17. The 64 output token IDs are
identical in this configuration. CLI decode duration per round, inferred from
63 decode tokens and the reported speed, is approximately 32.80 ms before and
32.88 ms after (+0.25%). The extra verification round explains nearly all of
the observed throughput difference; one sample cannot distinguish the remaining
Q5 accumulation variability from changes in corrected attention values. No Q5
code is modified. 32K/128K MTP acceptance is 100% before and after.
Windows desktop GPU memory before each process is 1558 MiB in the original
series and 1156 MiB in the repaired series; no intentional background GPU
workload was added. GPU memory and hashes are preserved per run.

Evidence: `before/summary.json`, `after/summary.json`,
`performance-comparison.json`; reproducible driver outside the checkout:
`measure_dense.py before` and `measure_dense.py after`.
