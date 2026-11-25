# Inkling 2 Mini – Stage 7A + 7B (experimental)

Base: `6dc762eabdf40f310595adbdfed60f02854ca9b8`, branch
`lab/inkling-mini-2-support` in `Xtravaganz/ninfer-rtx-pro-4000`.

## Scope

- 7A: NVFP4 G16 packed vector loads, scale reused per group, warp-level reduction,
  fused interleaved W13 gate/up + exact Inkling softcap, fused W2 + weighted reduce.
- 7B: same design for MXFP8 E4M3/G32 UE8M0, parallel FP32 sigmoid router with
  correction bias (bias affects ranking only), normalized coefficients, deterministic
  low-ID tie rule for exactly equal computed ranks. The fast router is shared by
  both formats.
- Four CUDA launches instead of Stage6H's six. No full expert bank copy. Existing
  Stage6G source-order planes, original factors (amax/denominator), FP32 inputs and
  accumulation. Workspace is 37,440 bytes (plus CUDA allocator overhead).
- The altered FP32 summation order may change rounding, including routing at
  near-ties. No claim of bitwise equality. Benchmark rejects any route-ID mismatch.
- No Tensor Core MMA, no new activation quantization, no grouped prefill, no
  native model executor, no MTP, no serving changes. These are experimental vector
  decode kernels inspired by NInfer's T=1 path, not copied upstream implementations.
- Separate DSO in `native7ab`, linked to the existing tested `native6ghi` library.
  Do not rebuild 6G/H/I using the legacy GPU1 build script. Stage6K/L is untouched.
- Workspace is single-stream and not reentrant; all tensors and resident store must
  outlive queued work. Graph compatibility is a test goal, not an established result.

## Build and first tests (physical GPU 0)

```bash
bash tools/inkling2/build_native_stage7ab.sh
python3 tools/inkling2/test_native_stage7ab_cpu.py
nvidia-smi
bash tools/inkling2/run_bench_stage7ab.sh 0
bash tools/inkling2/run_bench_stage7ab.sh 20
```

One physical bank per benchmark, so full model residency is not needed. Use the
same GPU with no competing workload. Each test checks zero input and twelve seeded
random inputs (default), exact ordered route IDs against a reproduction of 6E
routing, coefficient closeness (atol/rtol 2e-5), finite output and output closeness
(atol .005, rtol .002, inherited from existing reference check). Reports median
CUDA event time for the identical input on the same bank, interleaved A/B order.
These tolerances are initial gates, not proof of model quality. Inspect actual
errors and validate real hidden vectors and end-to-end tokens before promotion.

```bash
for layer in 6 12 17 19 23; do
  bash tools/inkling2/run_bench_stage7ab.sh "$layer" || break
done
INKLING_STAGE7AB_GRAPH_TEST=1 bash tools/inkling2/run_bench_stage7ab.sh 0
INKLING_STAGE7AB_GRAPH_TEST=1 bash tools/inkling2/run_bench_stage7ab.sh 20
```

The graph test uses Torch's non-default capture stream, compares replay to 6H,
and is deliberately opt-in. Timings remain eager versus eager. A speedup <=1
is a failed performance goal even if numerical parity passes.

## Optional full-forward check after isolated parity

The old runner previously failed to upload layer 0 (CUDA OOM). Check GPU 0 memory
first; this extension does not solve the earlier residency failure. The runner
prints free VRAM before preload and preserves `INKLING_NONEXPERT_CACHE_MIB`.
Use the old Stage6JKL runner for reference and the new runner with exactly the
same environment. Both remain Torch-controlled and have sequential prefill.

```bash
INKLING_NONEXPERT_CACHE_MIB=4096 INKLING_NATIVE_VERIFY_FIRST=1 \
INKLING_CUDA_GRAPH=0 INKLING_NATIVE_CONV=0 \
bash tools/inkling2/run_ask_native_stage6jkl.sh "Hi" 10

INKLING_NONEXPERT_CACHE_MIB=4096 INKLING_NATIVE_VERIFY_FIRST=1 \
INKLING_CUDA_GRAPH=0 INKLING_NATIVE_CONV=0 \
bash tools/inkling2/run_ask_native_stage7ab.sh "Hi" 10
```

After matched tokens and stable latency, repeat both with `INKLING_NATIVE_CONV=1`
to isolate the convolution integration. Compare token IDs, timing summaries and
`native_stage7ab_calls`; require actual calls >0. The inherited PERF SUMMARY
still uses the Stage6JKL mode label and excludes output head/sampling time.
Full-forward 7AB graph mode is explicitly disabled so the old 6K graph does not
silently bypass the candidate. No automatic fallback is enabled.

## Validation status at packaging

Host-only packing/ABI tests, Python syntax and shell syntax checked. CUDA
compiler, real GPU parity, graph replay and timing remain unverified. No remote
commit or model changes. This package is an additive implementation candidate.
