# OLMO_opencl: instructions for Claude

Read this before changing anything here. [README.md](README.md) is the
user manual. [PLAN.md](PLAN.md) is the design plus the dated log of
everything established so far (§2 reference facts, §3a reused code, §11a
status). Keep §11a up to date when you finish a phase.

## Motivation (real, not hypothetical)

The ALMA wheat model was fine-tuned with PyTorch on a CUDA Quadro P1000
that no longer exists. All its checkpoints were lost: the scratch dir
`~/RSDATA/alma_wheat_festival/` kept only the dataset and the 2026-06-04
inference PNGs/CSV. The dev team needs a new `.ckpt`. The only GPU is an
AMD Radeon Pro **WX 7100, Polaris10, 16 GiB** on `yann@10.42.0.89`
(connected from Friday 2 October 2026). PyTorch ROCm does not support
Polaris, so the compute is done in C + OpenCL.

The user has confirmed the 16 GiB many times. Don't question it.

## The goal defines "correct"

The deliverable is a Lightning `.ckpt` that
`~/dev/olmoearth_projects/ALMA_Inference_Wheat_Festival.sh` (olmoearth_run)
loads unchanged. So `olmo_cl` must reproduce **the reference training
recipe**, not a better one. Do not "improve" the model, loss, crops,
schedule or augmentation. Any deliberate deviation needs the user's
approval and a line in PLAN.md.

## Reference sources: read them, don't guess

Everything is in `~/dev/olmoearth_projects/.venv/lib/python3.12/site-packages/`:

| Topic | File |
|---|---|
| Config actually trained | `~/dev/olmoearth_projects/olmoearth_run_data/wheat_festival/model.yaml` |
| Encoder wrapper, pooling, legacy timestamps | `rslearn/models/olmoearth_pretrain/model.py` |
| Normalisation | `rslearn/models/olmoearth_pretrain/norm.py`, `olmoearth_pretrain/data/norm_configs/computed.json` |
| ViT, encodings, token order | `olmoearth_pretrain/nn/flexi_vit.py`, `nn/encodings.py`, `nn/attention.py` |
| Patch embed + resize | `olmoearth_pretrain/nn/flexi_patch_embed.py` |
| Band sets | `olmoearth_pretrain/data/constants.py` (SENTINEL2_L2A) |
| Crops, fixed val crops, layer stacking | `rslearn/train/dataset.py`, `rslearn/train/data_module.py`, `rslearn/dataset/storage/file.py` |
| Loss, metrics | `rslearn/train/tasks/segmentation.py` |
| Optimiser, scheduler | `rslearn/train/lightning_module.py`, `train/optimizer.py`, `train/scheduler.py` |
| Freeze callback | `rslearn/train/callbacks/freeze_unfreeze.py` |

Per `~/.claude/CLAUDE.md`: when a library question comes up (PyTorch
kernels, Lightning internals), search `$HOME/dev/` for its source first,
and ask the user to clone it if it's missing.

## Invariants that were verified: do not change without new evidence

Each item below cost a golden comparison to establish. The test that
catches a regression is in brackets.

1. **Token order** of the flattened sequence is (h, w, t, s):
   `row = ((h·16 + w)·T + t)·3 + s`. [`make check`, patch/encoded]
2. **Resize**: PyTorch `_upsample_bicubic2d_aa` with the **PIL cubic
   a = -0.5**, not the usual -0.75. Taps are clipped at the borders and
   renormalised. Only applies because patch 4 < base patch 8
   (64 → 128 px). [`resized_s*`]
3. **Encodings**: four 48-wide slots: channel (learned, per band set) |
   time `pos_embed[t]` | month `month_embed[t]` (legacy timestamps:
   month = t) | spatial `sincos(col·4,24) ++ sincos(row·4,24)`, column
   first, with frequencies `1/10000^(i/24/2)`. The time and month tables
   are read from the weights file, not recomputed. [`encoded`]
4. **Timestep order = raw `readdir()` order**, as rslearn's `iterdir()`.
   Never sort silently. `order=index` exists only as an explicit option.
   The order is written to `run_config.json`. [`input`]
5. **Stochastic depth is ON in training** (p = 0.1, per sample, separately
   for the attention and MLP branches of every block). rslearn's freeze
   callback only sets `requires_grad=False` and never calls `.eval()`.
   Kept branches are scaled by 1/0.9 (GEMM `alpha` plus a pre-scaled
   bias); dropped branches are skipped. Off in validation. [golden
   `train_mode` + `drops.json`]
6. **Val/test crops are fixed** (`fix_patch_pick=(split != "train")`) and
   stored in `data/val_test_crops.tsv`. Train crops are random.
7. **Head** = conv then bilinear ×4 (equivalent to rslearn's
   upsample-then-conv, because both are linear and the bilinear weights
   sum to 1). torch `align_corners=False`. CE is averaged over valid
   pixels (label ≠ 255). [`make headcheck`]
8. **AdamW / ReduceLROnPlateau** follow torch semantics exactly (weight
   decay 0.01 decoupled, plateau threshold 1e-4 relative, monitored on the
   epoch **train** loss). [`make test_optim`]
9. **Checkpoint selection**: top 3 by val wheat F1, where a new score must
   **strictly** beat the worst kept one (Lightning `torch.gt`), plus last.
   Names are `epoch=E-step=S` with a 0-based epoch and the global step
   after it.
10. LayerNorm eps = 1e-5 everywhere; GELU is the exact erf form; the
    attention scale is 1/8.

Tolerances in `src/check.c`: 1e-5 for host stages, 1e-3 after the
transformer blocks, 1e-4 for the head. The observed errors are far below
them (see README.md). If a change pushes an error up by an order of
magnitude, investigate even if it still passes.

## Data facts

- 74 windows, split 58 / 8 / 8 (train/val/test), 77–191 px, all with 6
  timesteps, so the encoder's unmasked fast path is always used. Mixed
  lengths fail loudly (not implemented).
- **Every window is all wheat or all background** (one polygon each).
  3 of the 8 val crops have no wheat, so val F1 over a subset of windows
  can be 0 legitimately.
- Labels: 0, 1, 255 (nodata). Any other value is a fatal error.
- The acquisition dates in readdir order are Nov, Dec, Jan, Apr, Mar,
  Feb, and the model sees them as "months" 0..5.

## Workflow

```sh
make && make test                  # seconds; run after any change to head/optim
make selftest DEVICE=cpu           # ~1 min on PoCL
make check DEVICE=cpu              # 2.5 min per golden on PoCL; run in background
make golden                        # only if the reference or dataset changed
```

- Laptop: PoCL only (Intel i7-1165G7), and slow (about 2 GFLOPS on the
  GEMMs). Use `augment=0 max_train=… max_val=…` for training-loop tests.
- Server: `export RUSTICL_ENABLE=radeonsi`, then `DEVICE=gpu
  PLATFORM=rusticl`. Without the variable, rusticl hides the GPU and the
  program silently lands on Clover or PoCL, so read the device line
  `olmo_cl` prints. Deploy by rsyncing the source and building on the
  host (the RRI.opencl procedure). The exact commands are in PLAN.md
  §11a. Expected speed: 0.3–0.6 s per crop, extrapolated from
  i.sam.opencl (ViT-H, 3.8 s per 1024² crop at about 0.8 TFLOPS).
- Long runs: start them in the background and watch the log. Don't block
  on them.

## Code conventions

- C11 (`-std=gnu11`), must build with `-Wall -Wextra` and no warnings.
  Format with `make format`, which uses GRASS's `.clang-format`: LLVM
  base, 4 spaces, Stroustrup braces.
- Comments are full sentences explaining intent. No decorative banners.
- **Fail loudly** through `fatal()` (`src/util.h`): on shape or size
  mismatches, missing files, bad labels, and paths too long for
  `xsnprintf`. Never guess or pad silently.
- Messages go through `msg_info/verbose/debug/warning`. Only data output
  uses `printf` (`info`, `data`, `weights`).
- Kernels: OpenCL C 1.1-compatible, **32-bit indices**
  (64-bit division is emulated on the GPU), **no private arrays** for
  accumulators (Mesa spills them to scratch, 70× slower; use `float4`
  registers as `gemm_big` does), and nothing that needs fp16 or
  subgroups.
- `ocl_backend.c`, `pth_loader.c`, the GEMM and LayerNorm kernels and the
  `ops.c` launchers come from `~/dev/i.sam.opencl`. The `_()` no-op macro
  is kept so they stay diffable against it. Port fixes in both directions.
- Licence: **Unlicense** (user's decision, 2026-09-29). Every new file
  gets `SPDX-License-Identifier: Unlicense` and no copyright line.
- Python tools run in the olmoearth_projects venv and follow PEP 8. They
  must build the model through rslearn (`make_golden.build()`), never by
  re-implementing it.
- Don't overwrite a run: `olmo_cl train` refuses an existing
  `run_log.jsonl` without `overwrite=1`, and `export_ckpt.py` needs
  `--overwrite`.

## Open items

- **Phase 4 (Friday):** GPU build, `selftest` and `check` on rusticl,
  then profiling. The GEMMs are narrow (D = 192), so check `gemm_big`
  occupancy. The attention score buffer is 255 MB; `ocl_alloc` fails
  loudly if `max_alloc` is smaller, and head-chunking (as in
  i.sam.opencl) would be the fix.
- **Phase 5:** full run; `make ckpt VALIDATE=1`. The Lightning val F1 must
  match `run_log.jsonl` within 0.5 %. `--validate` has not run yet.
- **Phase 6:** inference with the unchanged pipeline, and a comparison
  with `~/RSDATA/alma_wheat_festival/inference_outputs/20260604/`.
- **Timestep order across machines** (invariant 4): a decision for the
  user and the dev team, not something to fix silently.
- `ALMA_Train_Wheat_Festival.sh` looks for `best-*.ckpt`, but Lightning
  writes `epoch=E-step=S.ckpt`. This is not fixed in that repo.
- Out of scope unless asked: unfreezing the encoder (it would need
  backprop through the ViT), inference in C, and other OlmoEarth sizes
  (the loader reads D/depth/heads from the weights, but memory budgets
  assume Tiny).
