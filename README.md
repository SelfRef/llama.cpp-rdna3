# llama.cpp-rdna3

**A llama.cpp fork for AMD RDNA3 / RDNA3.5 on Vulkan — Radeon RX 7900 XTX (gfx1100),
RX 7800 XT (gfx1101) and Strix Halo / Ryzen AI Max+ 395 (gfx1151). Nothing else.**

It exists because the two things this hardware needs have never been in one tree:

1. **The ROCmFPx weight formats** — `Q4_0_ROCMFP4*`, `Q2/Q3/Q6/Q8_0_ROCMFPX` (ggml type ids
   100-107), which stock llama.cpp cannot even *load*. They come from
   [ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX) (originally charlie12345/ROCmFPX) via
   [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp), which is the base
   of this branch. On a 7900 XTX, Qwen3.8-27B at ROCmFP4-FAST decodes 33-38 % faster than the same
   model as a Q4_K_M on stock llama.cpp, in 2.4 GiB less VRAM, for +4.3 % perplexity.
2. **RDNA3 Vulkan work that was never upstreamed** — a body of measured patches that exists only as
   patch files passed between community forks, with no pull request behind any of them. Carrying
   them means carrying them ourselves.

Upstream is where this should all live, and some of it is on its way there
([#28898](https://github.com/ggml-org/llama.cpp/pull/28898) would put FP8/NVFP4 quant scales in ggml
with a Vulkan implementation; [#27952](https://github.com/ggml-org/llama.cpp/pull/27952) brings int8
coopmat1 MMQ to RDNA3). When it does, this fork should shrink, not grow.

## What branch `rdna3` carries

Base: `LaurentZuijdwijk/llama.cpp` @ `11bfe8a6` (upstream `0190529e`, 2026-08-30) — the ROCmFPx
formats, the batch-3..8 mat-vec path, `--spec-draft-adaptive`, and the RADV ≥ 25.3 coopmat LDS pad
gate — and, since 2026-09-18, **upstream master itself**: the re-port is complete, the fork is no longer behind.
On top of upstream there are five carried patches (the swiglu fusion was dropped once upstream's #27220
superseded it) plus the fork's own ROCmFPx type plumbing, its delta-net concat-transpose kernel, and the
UMA readback guard:

| # | Patch | Author | Why it is here |
|---|---|---|---|
| 1 | `vulkan: hoist the coopmat1 FA P-fragment load out of the hsv_tile loop` | Nathan Wilson | coopmat1 flash-attention is the path RDNA3 actually takes |
| 2 | `vulkan: store coopmat1 FA Psh query-major so the GEMM2 A load vectorizes` | Nathan Wilson | same |
| 3 | `vulkan: pin a 32-wide subgroup for coopmat1 FA where narrowing is free` | Nathan Wilson | same |
| 5 | `server: keep speculative checkpoints on device` | Gaetan Puleo | MTP speculative decoding is on for every model we run |
| 6 | `llama: opt-in KV cache row padding to defeat power-of-2 channel aliasing` | Nathan Wilson | **opt-in, off by default** (`LLAMA_KV_ROW_PAD`); measured null on gfx1100, kept as a knob for gfx1151 |

Every one of them is in the measured set below. **Nothing goes on this branch on inspection alone**
— see "Tried and parked" for what happened the one time it did.

Provenance: 1-3 and 5 are cherry-picked from
[voidsurfer/llama.cpp-nudge](https://github.com/voidsurfer/llama.cpp-nudge) with original authorship
intact; 4 and 6 come via [guevae2/paoai-strix-engine](https://github.com/guevae2/paoai-strix-engine),
which had already rebased them onto this exact base. Every commit carries a `cherry picked from`
line. All sources are MIT, as is this fork.

## Tried and parked

Two further patches by the same authors looked right on inspection — one numerical, one a
correctness fix — and were briefly carried on that basis. They were then measured, and bisected:

| | prose | json | refactor | prefill @32k | output |
|---|---|---|---|---|---|
| `rdna3` (patches 1-6) | 76.2 | 107.7 | 130.6 | 843.6 | matches base |
| `carry/vulkan-fa-mmq-fp32` (+ FA MMQ fp32 narrowing) | 76.0 | 107.7 | 130.4 | 841.4 | matches base |
| + `carry/mtp-full-checkpoints` as well | **58.4** | **80.5** | 120.2 | 806.4 | **differs** |

- **`vulkan: scale the FA MMQ dot product in fp32 before narrowing` — neutral, parked.** It costs
  nothing and changes nothing on an FP4 model, which is expected: MMQ is the integer-dot path the
  K-quants take, not this one. It is worth re-testing on a K-quant, and it is a reasonable candidate
  once the base moves up far enough that one binary serves every model.
- **`Reapply "common: use full checkpoints for MTP rollback"` — rejected.** On its own it accounts
  for the whole **-23 % prose / -24 % json** and it **changes the greedy output**, on a path that is
  supposed to be lossless. Draft acceptance rises (53 % vs 44 %) while throughput falls, i.e. the
  rollback is paying on every accepted token. Do not merge without understanding the output change.

The lesson is on the branch now: a patch gets carried when a benchmark says so, not when the commit
message is persuasive.

## A failure mode every canary missed

The first step-4 build passed the compiler, the ROCmFPx-type and adaptive-drafting canaries and a
smoke request (it answered "pong"), and then benchmarked at **2.7 t/s** — Qwen3.8-27B on sixteen CPU
threads. `--list-devices` printed `(none)`. Cause: the hunk cleanup had cut the only definition of
`ggml_vk_wait_for_fence` (a definition count that mistook the header declaration for one), a shared
library links fine with an undefined symbol, `dlopen` of `libggml-vulkan.so` then fails at runtime,
and ggml registers the CPU backend alone — correct output, silently 25× slower.

Two guards now sit in the build pipeline: `ldd -r` on every `libggml-*.so` must report no undefined
symbol (`ldd` without `-r` does not resolve symbols and sees nothing), and every hop's smoke step
requires `--list-devices` to show `Vulkan0` before any benchmark runs. If you build this tree
yourself, run both.

**Deliberately not carried:** everything DeepSeek-V4-specific (not run here), the DFlash2 draft-cache
patches (DFlash2 loses to the baked MTP head on these cards, and it cannot be combined with it), the
`mul_mat_id` IQ-type pipelines (second half of a two-commit series whose first half does not apply,
and no IQ quant runs here), and anything already in the base — notably the RADV coopmat pad-2 driver
gate, which is present and is worth ~11 % prefill on its own.

## Measured

7900 XTX (gfx1100), RADV / Mesa 26.2.2, Qwen3.8-27B ROCmFP4-FAST, 131k ctx, f16 K / q4_0 V, MTP n4,
greedy, median of 2 — patches 1-6 against the same base without them:

| | prose | json | refactor | prefill @32k |
|---|---|---|---|---|
| ROCmFPx base (`11bfe8a6`, upstream 08-30) | 75.6 | 106.5 | 129.3 | 818.6 t/s |
| the six patches, at that base | 76.2 | 107.7 | 130.6 | **843.6 t/s** |
| base moved to upstream 09-09 (step 1) | 72.1 | 103.9 | 129.8 | 841.6 t/s |
| through #25773 (step 2) | 71.9 | 104.3 | 130.0 | 859.6 t/s |
| base moved to upstream 09-17 (step 3) | 74.4 | 106.1 | 130.4 | 864.3 t/s |
| **this branch** (upstream master 09-18, step 4) | 74.1 | 105.7 | 129.9 | 868.1 t/s |

The 5 % prose / 4 % json decode this branch gives up against the row above is **entirely** upstream
[#28068](https://github.com/ggml-org/llama.cpp/pull/28068), isolated by building with it reverted
(75.6 / 108.3, i.e. reference speed restored). It recomposes the gated-delta-net L2 norm as
`rms_norm(eps/n) · 1/√n` so epsilon is applied the way the reference implementation does, across
`q_conv` and `k_conv` in all 48 GDN layers — two ops where there was one. Perplexity is unmoved
(6.9218 vs 6.9211, ~1 % of one standard error), so the cost buys fidelity that wikitext cannot see.
It stays: reverting would mean carrying a divergence from upstream on model correctness, forever,
at every future hop.

Step 3 (upstream to 09-17) won back part of the #28068 cost: +3.0 % prose / +1.8 % json over step 2 in one
session, with upstream's #28457 small-M kernels (the MTP verify shape) as the visible cause — the prose hash
changes, json's does not. Step 4 (current master, through the Vulkan source split): decode and prefill within noise of step 3 (−0.4…−0.8 %), output byte-identical on all four
presets. The re-port is complete; the fork is at upstream master.

Step 2 (#25773, upstream's spec-constant matmul rewrite) then cost nothing: decode within noise of
step 1, prefill +1.9 %, output byte-identical on all four presets. The ROCmFPx types now register
through upstream's own per-type `X(TYPE, tstr)` idiom (`FOR_EACH_ROCMFPX_TYPE`) instead of the
fork's macro-per-type scheme, and the fork's f16-B routing — which was **on by default**, not an
unused knob — is replaced by upstream doing the same f32→f16 B conversion unconditionally on
coopmat1. Two fork tuning knobs went with it (`GGML_VK_DENSE_WAVE32`, off and measured −5.8 %;
`GGML_VK_MMID_WG256/WAVE32`, off and never measured). **A fused eps-aware `l2_norm` in ggml would recover most of the 5 % — that is
the first thing this fork should try to upstream.**

Decode is a tie; prefill is **+2.8 % at 32k** and +1.3-3.3 % on short prompts, with identical VRAM
and **byte-identical greedy output**. `LLAMA_KV_ROW_PAD=256` measured null on this card and cost
0.6 GiB, so leave it unset unless you are on gfx1151 and have measured otherwise.

## Branches

| Branch | What it is |
|---|---|
| `master` | untouched mirror of `ggml-org/llama.cpp`. Never commit here — it is what keeps "Sync fork" and every cross-fork compare working. |
| `rdna3` | **default**, and what gets built. The base above plus the eight commits. |
| `carry/*` | one branch per carried patch set, so a bad upstream rebase blows up in one place instead of all eight. |

## Roadmap: moving the base

The base is 330 commits behind upstream, and catching up is a **re-port, not a rebase**: since the
fork point upstream rewrote matmul pipeline creation into one spec-constant shader per quant family
([#25773](https://github.com/ggml-org/llama.cpp/pull/25773)) and split the Vulkan sources into
separate files ([#28732](https://github.com/ggml-org/llama.cpp/pull/28732)).

Measured merge cost from this branch, 2026-09-18:

| merge target | conflicting files | conflict hunks | status |
|---|---|---|---|
| upstream just before #25773 (09-09) | 10 | 26 | **done** |
| **at #25773** (spec-constant matmul) | 2 | 19 | **done** — registration work, not shader work |
| just before #28732 (09-17) | 1 | 1 | **done** — `src/CMakeLists.txt` only |
| current master 44be98f0 (09-18), through the source split | 3 | 7 | **merged** — **done** — benchmarked, see "Measured" |

(Costs re-measured from each adopted tip; the original single-jump estimate was 13 files / 47 hunks.)

So roughly **half the work is the single #25773 step**, and it is exactly where the ROCmFPx types
have to be re-expressed in the new `create_mm_pipelines` / spec-constant scheme rather than merged.
`ggml-vulkan.cpp` carries 28 of the 47 hunks at the far end; the rest are `qwen4exp.cpp` (4),
`llama-memory-hybrid-idx.*` (5) and single hunks in converters, CMake and tests.

Do it in those four steps, building and benchmarking each one against the table in "Measured" —
a step that costs decode is a step to stop and understand, not to push through. The prize at the
end is that this tree can take upstream PRs *and* the FP4 types at once, which is what the separate
patched `llama-server` in [llama-swap-docker-amd](https://github.com/SelfRef/llama-swap-docker-amd)
exists to work around today. Once it can, that binary goes back to being stock upstream.

## Phase 2: one binary

With the base at master, the tree can take upstream PRs — the thing the separate patched `llama-server`
in [llama-swap-docker-amd](https://github.com/SelfRef/llama-swap-docker-amd) exists to work around.
Measured against master on 2026-09-18, of that image's 13 `LLAMA_PATCHES` **eleven merge cleanly**;
two conflict in one hunk each and need a rebase onto this tree:

| PR | what it is | state |
|---|---|---|
| #27952 | int8 coopmat1 MMQ for RDNA3 — the measured RDNA3 prefill win (+4.6 % dense / +18.5 % MoE) | 1 hunk vs master, in the int-shmem warptile selection |
| #25666 | no MMVQ on speculative-decode steps — `qwen38-bart`'s draft acceptance | 1 hunk vs master, a device-tuning constant block |
| #28243 | Qwen3.8-Flash-Next MTP head | the image's local `28243-rebased.patch` no longer applies (2 of 19 files), but the PR head itself now merges cleanly — use the PR ref |

Once those land here, `llama-server` in the image goes back to byte-for-byte upstream and this fork is
the one binary for every entry. Not before: that order was chosen so no production entry ever sees a
regression window.

## Build

Vulkan only — the ROCmFPx types ship no HIP kernels.

```bash
cmake -B build -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON \
      -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

Two invariants worth asserting in any build pipeline — if either fails, the tree has silently lost
the reason it exists and is about to ship as a plain llama.cpp:

```bash
llama-quantize --help | grep -q Q4_0_ROCMFP4_FAST     # the ROCmFPx types
llama-server   --help | grep -q -- --spec-draft-adaptive
```

Both binaries print help to **stdout and exit 1**, so capture before grepping under `pipefail`.

For anything not specific to this fork — general build options, the server's flags, the model zoo —
follow [upstream's documentation](https://github.com/ggml-org/llama.cpp): this tree only adds the
RDNA3 pieces described above and is otherwise current master.

In [SelfRef/llama-swap-docker-amd](https://github.com/SelfRef/llama-swap-docker-amd) this tree is the
`llama-rdna3` stage and installs as `llama-server-rdna3`, `llama-cli-rdna3`, `llama-bench-rdna3`,
`llama-quantize-rdna3`, `llama-perplexity-rdna3` (build args `WITH_RDNA3`, `RDNA3_REPO`,
`RDNA3_BRANCH`, `RDNA3_COMMIT` — pin the commit).
