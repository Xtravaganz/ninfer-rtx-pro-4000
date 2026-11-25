# Inkling 2 Mini Stage 7C/D — opt-in native C++ executor and resident greedy head

Base: 6dc762e, plus installed and GPU-tested Stage7AB. No existing files or DSOs replaced. GPU 0 only. Stage 7C is a transitional C++/ATen implementation: all 48 visits, three core passes, cache updates, controllers, cross-pass and final residuals are orchestrated in one C++ operator, no Python callback per layer. Dense math and attention still use ATen CUDA operators, not optimized NInfer kernels. Stage7AB and Stage6L are called directly via their existing C ABIs, on the same stream. Python owns startup, tokenizer, per-token driver and validation. This is not registered ninfer-serve; full model graphs, batch prefill, optimized attention, persistent scratch arena and stochastic sampling are out of scope. ATen intermediates still allocate through its caching allocator. KV is fixed-capacity per slot and not a paged/ring cache. All model formats/weights remain unchanged.

Stage 7D binds the original embedding in its stored dtype and the first 200058 original output rows as FP32 once. Native C++ ATen GEMV + softcap + GPU max with lowest-ID tie handling; only ID and selected logit are read back. No per-token head weight streaming, no full-logit readback. This is greedy only. Extra VRAM for the FP32 head is ~1758 MiB; embedding ~883 MiB if BF16 or ~1767 MiB if FP32, plus KV (at capacity 128 ~48 MiB). Startup is excluded from warm timing. All GPU stages require separate parity validation.

## Install/build
From the Inkling worktree, install.sh then `bash tools/inkling2/build_native_stage7cd.sh`. The builder derives an image from the existing HF runtime (adds compiler and ninja, needs apt access) to ensure the same libtorch ABI. It compiles only the C++ extension and runs actual host tests before publishing the library. No nvcc build or old DSO replacement. CPU-only compilation and primitive tests were also performed with local Torch 2.10.0; the exact target image and GPU were not available locally.

## GPU tests
```
INKLING_STAGE7CD_VERIFY=1 bash tools/inkling2/run_ask_native_stage7cd.sh "Hi" 10
INKLING_STAGE7CD_MODE=c bash tools/inkling2/run_ask_native_stage7cd.sh "Hi" 10
bash tools/inkling2/run_ask_native_stage7cd.sh "Hi" 10
```
Verify runs both native and Stage7AB+6L Torch paths over all prompt and generated inputs, independent KV/convolution state, shared immutable experts. Every final hidden is checked at atol=.005/rtol=.002; greedy IDs on both hidden states must match. In CD mode the resident head is also checked against the original streaming scanner on the native hidden. Verification aborts on mismatch; combined timing is deliberately omitted. Mode c runs new C++ executor with old streaming head; mode cd uses new head. Compare the non-verification runs for the head speedup. Head times and combined forward+head times are wall-clock, startup/tokenization/rendering excluded. `expected_moe_visits` is a schedule-derived count, not a GPU counter. Full-model quality and official HF parity are not claimed. No performance target is promised.

Defaults: capacity 128, nonexpert budget 4096 MiB. Only original geometry H2304/Hq24/Hkv8/D128/rel16/window1024, kernel4. `INKLING_STAGE7CD_CAPACITY` up to 4096 (check VRAM). No CUDA graphs. The short Hi regression from 7AB+6L is [200001,200008,371,150409,150239,150330,196980,4896,1,13].
