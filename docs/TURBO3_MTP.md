# TurboQuant KV Cache + MTP Speculative Decoding

This build merges two extensions onto llama.cpp:

- **MTP speculative decoding** (upstream PR #22673) — Multi-Token Prediction head generates draft tokens; main model verifies in one forward pass. 75–91% draft acceptance depending on KV type and n-max.
- **TurboQuant KV cache** (based on arXiv 2504.19874) — WHT rotation + polar quantization at 3.5 bpv (turbo3), 2.5 bpv (turbo2), 4 bpv (turbo4). Specialized flash-attention VEC kernel with shared-memory LUT.

Tested on RTX 3090 (CC 8.6, Ampere) + RTX 3060, Qwen3.6-27B Q4_K_M with MTP head.

---

## Requirements

- CUDA-capable GPU (Ampere CC 8.x tested; Ada CC 8.9+ also supported)
- Model with MTP head layers (e.g. Qwen3.6-27B MTP GGUF from Unsloth)
- cmake ≥ 3.21, CUDA toolkit ≥ 12.0

---

## Build

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc) --target llama-server
```

The turbo3 CUDA kernels compile with the rest — no extra flags needed.

---

## Launch configs

### Single user — max throughput (q4_0 + ngram + MTP)

```bash
./build/bin/llama-server \
  --model /path/to/model-MTP.gguf \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --spec-type ngram-mod,draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.75 \
  --parallel 1 \
  -fit \
  --port 8080
```

**66.9 tok/s** generation, 86% draft acceptance, ~21.9 GB VRAM (single 3090). This is the fastest config tested.

### Multi-user / long context — turbo3 + MTP

```bash
TURBO_AUTO_ASYMMETRIC=0 \
./build/bin/llama-server \
  --model /path/to/model-MTP.gguf \
  --cache-type-k turbo3 \
  --cache-type-v turbo3 \
  --spec-type draft-mtp \
  --spec-draft-n-max 1 \
  --spec-draft-p-min 0.75 \
  --parallel 4 \
  -fit \
  --port 8080
```

**37.7 tok/s** generation per slot on RTX 3090 + RTX 3060 with 4 slots, 95% draft acceptance, ~22.7 GB total VRAM. Each slot gets 65K context (`-fit` divides the total context pool across slots). Use this for concurrent serving, not single-request speed.

---

## Key flags

| Flag | Values | Effect |
|------|--------|--------|
| `--cache-type-k` | `f16`, `q8_0`, `q4_0`, `turbo2`, `turbo3`, `turbo4` | K cache quantization |
| `--cache-type-v` | same | V cache quantization |
| `--spec-type` | `draft-mtp`, `ngram-mod,draft-mtp` | Enable MTP alone or combined n-gram + MTP speculative decoding |
| `--spec-draft-n-max` | 1–6 | Max draft tokens per verify pass |
| `--spec-draft-p-min` | 0.0–1.0 | Min probability threshold to accept a draft token |
| `-fit` | flag | Auto-size n_ctx to fill available VRAM |
| `TURBO_AUTO_ASYMMETRIC=0` | env var | Disable auto-upgrade of K from turbo3 → q8_0 on GQA ratio ≥ 6:1 |

---

## n-max sweep (Qwen3.6-27B, RTX 3090, CUDA_VISIBLE_DEVICES=0)

| Config | n-max 1 | n-max 2 | n-max 6 |
|--------|---------|---------|---------|
| q4_0 gen tok/s | 49.9 | **53.9** | 55.7 |
| q4_0 accept % | 94.8% | 91% | 77% |
| turbo3 gen tok/s | **48.2** | 47.4 | 37.8 |
| turbo3 accept % | 95.4% | 78% | 48% |

**q4_0**: n-max 2 is the sweet spot.
**turbo3**: n-max 1 = n-max 2 in throughput; n-max 6 actively regresses (acceptance collapses to 48%, verification overhead dominates). Use n-max 1.

---

## Full benchmark (RTX 3090 only, 1 parallel slot, CUDA_VISIBLE_DEVICES=0)

| Configuration | ctx / slot | VRAM | gen tok/s | prefill tok/s | MTP accept |
|---|---|---|---|---|---|
| q4_0, no MTP | 262,144 | 20.8 GB | 36.5 | 187.8 | — |
| turbo3, no MTP | 262,144 | 19.6 GB | 35.8 | 178.2 | — |
| q4_0 + MTP n-max 2 | 262,144 | 22.2 GB | 53.9 | 149.4 | 91% |
| turbo3 + MTP n-max 1 | 262,144 | 20.9 GB | 48.2 | 162.9 | 95% |
| **q4_0 + ngram-mod + MTP** | **262,144** | **21.9 GB** | **66.9** | 158.3 | **86%** |
| k=q8_0, v=turbo3 + MTP n-max 1 | 262,144 | 23.4 GB | 47.0 | 174.9 | 86% |

### Multi-GPU (RTX 3090 + RTX 3060, 4 parallel slots, turbo3+MTP n-max 1)

| Metric | Value |
|---|---|
| VRAM GPU0 (3090) | 13.8 GB |
| VRAM GPU1 (3060) | 8.9 GB |
| ctx / slot | 65,536 (4 slots × 65K = 262K total) |
| gen tok/s | 37.7 |
| MTP accept | 95% |

Speed penalty from inter-GPU communication: ~22% vs single-GPU. Use for concurrency (4 simultaneous users), not single-request speed.

---

## Caveats

**TURBO_AUTO_ASYMMETRIC** (default: on): when the model's GQA ratio hits 6:1 (e.g. Qwen3.6-27B: 24Q/4KV), K is silently upgraded from turbo3 → q8_0. This preserves attention quality at the cost of losing most VRAM savings. Set `TURBO_AUTO_ASYMMETRIC=0` to disable and keep full turbo3 on both K and V.

**-fit with turbo3 + multiple GPUs**: `-fit` computes one total context pool and divides it across slots. With 4 slots in the tested RTX 3090 + RTX 3060 setup, turbo3 loaded cleanly at 65K tokens per slot (262K total), using 13.8 GB on the 3090 and 8.9 GB on the 3060. Use `CUDA_VISIBLE_DEVICES=0` to constrain to one GPU if you want single-GPU performance numbers.

**MTP draft context inherits KV types**: the MTP draft context uses the same `--cache-type-k/v` settings as the main context. Both contexts are allocated; budget accordingly (~1.4 GB extra for MTP draft at 262K).

**n-max > 2 hurts turbo3**: turbo3's 78% per-token acceptance means the probability of all 6 draft tokens being accepted is ~48% — most long chains get partially rejected. Verification overhead of rejected chains is expensive. Stay at n-max 1.

**CUDA Ampere dispatch fix**: turbo types have no `ggml_get_to_fp16_cuda` converter registered. Without the fix in `fattn.cu`, the MMA kernel calls a null function pointer on the CPU → SIGSEGV. This fix routes all turbo K/V combinations to the VEC kernel on pre-Ada Ampere (CC < 8.9) regardless of batch size.

---

## Tested configs — findings

**`--spec-type ngram-mod,draft-mtp` is the single-GPU speed champion.** 66.9 tok/s vs 53.9 for MTP alone — a 24% gain. ngram-mod drafts from the token n-gram cache for repetitive segments; MTP covers the rest. Combined, more draft tokens are generated per verify pass (259 vs 230) at 86% acceptance. Same VRAM as q4_0+MTP. Add `--spec-type ngram-mod,draft-mtp` to the q4_0 launch config above.

**`--cache-type-k q8_0 --cache-type-v turbo3` is not worth it.** Expected to recover acceptance toward 91% (high-quality K) while saving VRAM vs full q4_0. Reality: acceptance at n-max 2 is 79.6% (barely better than turbo3+turbo3 at 78%), and it costs +3 GB VRAM vs full turbo3 — making it strictly worse than both alternatives. The V=turbo3 compression is what hurts acceptance; K quality doesn't compensate.

**Multi-GPU 4-slot turbo3 loads cleanly and hits 95% acceptance.** The 22% speed penalty vs single-GPU is from PCIe inter-GPU traffic. Total context across 4 slots is 262K (65K per slot), not 262K per slot — `-fit` divides the pool. For maximizing total throughput across concurrent users, multi-GPU still wins over single-slot: 4 × 37.7 = 150.8 effective tok/s system-wide.
