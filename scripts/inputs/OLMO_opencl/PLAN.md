# OLMO_opencl — C + OpenCL re-implementation of the ALMA wheat fine-tuning

Plan written 2026-09-29. Target: produce a wheat-segmentation checkpoint
equivalent to what `ALMA_Train_Wheat_Festival.sh` produces, using the AMD GPU
on `yann@10.42.0.89` (OpenCL), with no CUDA and no PyTorch GPU backend.

## 1. Goal and deliverable

The dev team needs a `.ckpt` that `olmoearth_run` / `ALMA_Inference_Wheat_Festival.sh`
can load unchanged. So the deliverable is **a PyTorch Lightning checkpoint
with the exact `state_dict` layout of `RslearnLightningModule`**, not a new
model format. The C/OpenCL tool does the expensive compute; a small CPU-only
Python exporter writes the final `.ckpt`.

"Same effect" means:

- same data (the 74 windows in `~/RSDATA/alma_wheat_festival/dataset`, same
  train/val/test split);
- same model (OlmoEarth-v1-Tiny encoder, frozen, + Upsample ×4 bilinear +
  1×1 conv 192→2 + softmax);
- same training recipe (random 64×64 crop + random H/V flip per window per
  epoch, cross-entropy on valid pixels, AdamW lr 1e-4, ReduceLROnPlateau on
  train loss, 80 epochs, keep the top-3 by val wheat F1 + last);
- a checkpoint loadable by the existing inference pipeline.

Bit-exact reproduction is **not** a goal: the original ran the encoder under
bf16 autocast and the head under fp16-mixed, with a different RNG. We run
FP32 throughout, and I expect agreement within a few % on val F1.

## 2. Key facts from the reference code (read 2026-09-29)

All paths are relative to
`~/dev/olmoearth_projects/.venv/lib/python3.12/site-packages/`.

| Aspect | Reference behaviour | Source |
|---|---|---|
| Encoder | OlmoEarth-v1-Tiny: D=192, depth 12, 3 heads (head dim 64), MLP ratio 4 (768), GELU, pre-norm LayerNorm, no qk-norm, no LayerScale, no register tokens, **drop_path 0.1 active during training** (the freeze callback only sets `requires_grad=False`, never `.eval()`); off at validation | HF `allenai/OlmoEarth-v1-Tiny/config.json`, `olmoearth_pretrain/nn/attention.py` |
| Frozen | `FreezeUnfreeze unfreeze_at_epoch: 9999` → encoder never trains | `olmoearth_run_data/wheat_festival/model.yaml` |
| Input | 12 S2 bands, order `B02 B03 B04 B08 B05 B06 B07 B8A B11 B12 B01 B09`, T=6 timesteps (layers `sentinel2`, `sentinel2.1` … `sentinel2.5`), UInt16 GeoTIFF, 10 m | dataset + `model.yaml` |
| Normalisation | per band: `(x - (mean - 2·std)) / (4·std)` using `olmoearth_pretrain.data.normalize.load_computed_config()["sentinel2_l2a"]` | `rslearn/models/olmoearth_pretrain/norm.py` |
| Band sets | 3 tokens per (pixel-patch, timestep): {B02,B03,B04,B08}, {B05,B06,B07,B8A,B11,B12}, {B01,B09} — each with its own patch-embed conv | `olmoearth_pretrain/data/constants.py:246` |
| Patch embed | model base patch 8, requested patch 4 → input per band set is **bicubic-resized (antialias=True) 64×64 → 128×128**, then Conv2d kernel 8 stride 8 → 16×16×192 | `olmoearth_pretrain/nn/flexi_patch_embed.py:92-160` |
| Encodings | D split in 4 × 48: `[0:48]` learnable channel embed per band set; `[48:96]` 1D sincos time pos (t=0..5); `[96:144]` month sincos table (legacy: month = t, i.e. 0..5); `[144:192]` 2D sincos spatial with gsd ratio = 10·4/10 = 4.0 | `olmoearth_pretrain/nn/flexi_vit.py:728-955`, `nn/encodings.py` |
| Sequence | 16·16·6·3 = **4608 tokens**, all valid (every window has 6 timesteps → `fast_pass=True`, no mask) | checked on dataset: 74/74 windows have 6 layers |
| Output | final LayerNorm → reshape BHWTSC → **mean over T and S** → 192×16×16 | `rslearn/models/olmoearth_pretrain/model.py` `forward` |
| Decoder | `Upsample(scale 4, bilinear, align_corners=False)` → `Conv 1×1 192→2` (Identity act) → `SegmentationHead` (softmax) | `model.yaml`, `rslearn/models/upsample.py` |
| Loss | `cross_entropy(logits, labels, reduction="none")`, masked by valid (label ≠ 255), divided by Σvalid; no class weights, no dice | `rslearn/train/tasks/segmentation.py:293-350` |
| Optimiser | AdamW, lr 1e-4, torch defaults (β 0.9/0.999, eps 1e-8, wd 0.01) | `rslearn/train/lightning_module.py`, `train/optimizer.py` |
| Scheduler | ReduceLROnPlateau on epoch `train_loss`, factor 0.2, patience 2, cooldown 10, min_lr 0 | `lightning_module.py:199` |
| Batching | batch_size 1; 58 train, 8 val, 8 test windows → 58 steps/epoch (the old `epoch=12-step=754` = 13·58) | dataset metadata |
| Crop | train: uniform random 64×64 offset per sample (all windows ≥ 77 px, so never padded); **val/test: fixed per window** (`fix_patch_pick=(split != "train")`, `random.Random(idx)`), exported to `data/val_test_crops.tsv` | `rslearn/train/dataset.py:807-840`, `train/data_module.py:140` |
| Timestep order | rslearn stacks the layers in **raw `readdir` order** (`Path.iterdir()`, unsorted). Here: `sentinel2.5, .4, .3, sentinel2, .1, .2` in all 74 windows (acquisitions Nov, Dec, Jan, Apr, Mar, Feb). With legacy timestamps, position t sets the time and month encodings | `rslearn/dataset/storage/file.py:158` |
| Augment | train only: independent 50 % horizontal and 50 % vertical flip of image + label | `rslearn/train/transforms/flip.py` |
| Checkpointing | `ModelCheckpoint(monitor=val_wheat_seg/wheat_f1, mode=max, save_top_k=3, save_last=True)` | `model.yaml` |

### Consequence that makes this project small

Because the encoder is frozen, **the only trainable parameters are the 1×1
conv: 192×2 weights + 2 biases = 386 numbers.** Also, bilinear upsampling is
linear with weights summing to 1, so

    conv1x1(upsample(F)) == upsample(conv1x1(F))

The head can therefore be computed at 16×16 and upsampled afterwards. All the
GPU work is the **forward pass of a frozen 12-block ViT over 4608 tokens**.
No autograd through the transformer, no activation storage, and no optimiser
state on the GPU. The head's forward, backward and AdamW run on the CPU in
microseconds.

## 3. Things to settle before/while building

1. **CPU fallback exists and needs no new code.** The PyTorch pipeline can
   train on this laptop's CPU today. In `model.yaml`, set `precision: 32` (or
   `bf16-mixed`), because `16-mixed` is CUDA-only. Estimated time: 3–7 h for
   80 epochs, or ~1–2 h with `max_epochs: 20` (the old best was epoch 12).
   Use this path if a checkpoint is needed before OLMO_opencl is validated.
   It is also the **reference oracle** for §8.
2. **Server GPU: settled.** It is an AMD Radeon Pro WX 7100 (Polaris10,
   36 CUs, **16 GiB**), as documented in `i.sam.opencl.md` and
   `i.sen2sr.opencl.md`. The peak working set here is < 0.5 GB (§9).
3. **Server OpenCL runtime: settled, see §3a.** Use Mesa **rusticl** with
   `RUSTICL_ENABLE=radeonsi`, FP32 only. Kernels stay OpenCL C 1.1-compatible
   so that Clover also works.

## 3a. Proven OpenCL configurations and code to reuse (from `~/dev/*opencl`)

What is proven on `yann@10.42.0.89`:

| Project | Platform on the WX 7100 | Workload | Result |
|---|---|---|---|
| `i.sam.opencl` | Mesa **rusticl** (`RUSTICL_ENABLE=radeonsi`) | **ViT-H image encoder**, 64×64 = 4096 tokens, global + windowed attention, read directly from `.pth` | 3.8 s per 1024² crop (≈ 3 TFLOP → **≈ 0.8 TFLOPS effective**) |
| `i.sen2sr.opencl` | Mesa rusticl | Sentinel-2 super-resolution CNNs, safetensors | 11–27× the laptop CPU; **matches PyTorch within 2e-6** |
| `RRI.opencl`, `r.watershed.opencl` | Mesa **Clover**, OpenCL 1.1 (rusticl there reported 0 devices when it was not enabled) | fp64 hydrology | ctest 7/7 on GPU; rsync-source-and-build-on-host procedure |
| all of the above, laptop | PoCL (`cpu-skylake-avx512`) | same kernels on CPU | development / CI device |

Direct reuse from `~/dev/i.sam.opencl` (the closest match: a ViT encoder in
C + OpenCL with PyTorch weights and no Python at run time):

- **`ocl_backend.c/.h`**: device selection (`auto|gpu|cpu`, platform-name
  substring filter such as `rusticl`/`Clover`/`Portable`), automatic
  `-cl-std=CL1.1|CL1.2 -cl-mad-enable`, buffer helpers, error strings, and
  optional per-kernel profiling via an env var. Replace `G_fatal_error()` with
  a local `fatal()`, because OLMO_opencl is not a GRASS module.
- **`pth_loader.c/.h`**: reads a `torch.save` zip `state_dict` directly
  (F32/F16/BF16 → F32), with `pth_require()` shape checks. This **replaces
  `tools/export_weights.py`**: `olmo_cl` reads the HF `weights.pth` itself.
  Check in phase 1 that the OlmoEarth file uses the stored (uncompressed)
  zip format the loader expects.
- **Kernels in `sam_kernels.cl`**:
  - `gemm` / `gemm_big`: batched, with PyTorch `[out,in]` weights via
    `transb`, and bias + GELU(erf) + residual fused. This covers qkv, proj,
    fc1 (+GELU), fc2 (+residual), the patch embed as im2col·W, and the
    attention matmuls.
  - `layernorm`, `patch_im2col`, `add_bcast`.
  - The attention pattern: **QKᵀ via `gemm` into a materialised score buffer,
    a row-softmax kernel, then S·V via `gemm`**, with a head chunk sized from
    `max_alloc` (≤ 512 MB). For OlmoEarth, 3 heads × 4608² × 4 B = 255 MB
    fits in one chunk. Use SAM's `softmax_relpos` without the rel-pos terms.
    This replaces the fused flash-attention kernel of the first draft, which
    was riskier and has no precedent here.
- **The embedding of kernels into the binary**: the Makefile `sed` rule turns
  `*.cl` into a C string header, so no `.cl` file has to be deployed.
- **Dump + reference-compare pattern**: an env var (`OLMO_CL_DUMP=dir`) writes
  raw `.f32` intermediates, and `tests/reference_compare.py` compares them with
  PyTorch. This is the same idea as §8's `make_golden.py`, reversed, and is
  proven workable.

Lessons already paid for (from the kernel comments and READMEs):

- Index with **32-bit ints** in kernels. 64-bit division is emulated on the
  GPU and dominated the run time. Keep every tensor < 2³¹ elements.
- **No private arrays** for accumulators. Mesa puts them in scratch memory
  (70× slower). Use explicit `float4` register accumulators, as `gemm_big`
  does.
- rusticl does not expose the GPU unless `RUSTICL_ENABLE=radeonsi` is set.
  Otherwise the program silently lands on Clover or PoCL. Print the chosen
  platform and device, and allow `platform=rusticl` to force it.
- Deploy by **rsyncing the source tree and building on the host**, as in the
  RRI procedure, not by copying binaries.
4. **Encoder weights are no longer cached** (`~/.cache/huggingface` has no
   OlmoEarth files). Download `allenai/OlmoEarth-v1-Tiny/weights.pth`
   (57 MB) once.
5. **Bug in `ALMA_Train_Wheat_Festival.sh`**: it searches for `best-*.ckpt`,
   but `ModelCheckpoint` has no `filename=`, so Lightning writes
   `epoch=E-step=S.ckpt`. The script always reports "not found — using
   last.ckpt". OLMO_opencl will name files the Lightning way. Fix the script
   separately if the dev team keeps using it.

## 4. Tool layout

```
~/dev/OLMO_opencl/
├── PLAN.md                     # this file
├── README.md                   # user manual
├── README_for_LLM.md           # instructions for AI agents (invariants, workflow)
├── Makefile                    # gcc -O3 -std=gnu11, -lOpenCL -lgdal -lm, OpenMP; sed rule embeds kernels/*.cl (as i.sam.opencl)
├── .clang-format               # LLVM base, 4-space, Stroustrup braces
├── src/
│   ├── main.c                  # CLI: olmo_cl train|eval|features|selftest  [device=auto|gpu|cpu] [platform=rusticl]
│   ├── ocl_backend.c/.h        # from i.sam.opencl (G_fatal_error → fatal)
│   ├── pth_loader.c/.h         # from i.sam.opencl: reads HF weights.pth directly
│   ├── dataset.c/.h            # window discovery, split tags, GDAL reads, crop, flip, normalise
│   ├── encoder.c/.h            # host-side orchestration of the 12 blocks (enqueue kernels)
│   ├── encodings.c/.h          # precompute channel/time/month/spatial tables on CPU (constant per run)
│   ├── head.c/.h               # 1×1 conv, bilinear up/adjoint, softmax-CE, AdamW, plateau LR
│   ├── metrics.c/.h            # confusion matrix → precision / recall / F1 (class 1), micro acc
│   ├── rng.c/.h                # seeded PCG32 for crops and flips (reproducible runs)
│   └── ckpt_state.c/.h         # write head weights + run log (JSON) per epoch
├── kernels/
│   ├── resize_bicubic_aa.cl    # 64→128 bicubic (A=-0.75, PyTorch convention); antialias is a no-op when upsampling
│   ├── patch_embed.cl          # conv 8×8 stride 8, per band set
│   ├── add_encodings.cl
│   ├── layernorm.cl            # from i.sam.opencl; eps read from the reference (see §8)
│   ├── gemm.cl                 # gemm + gemm_big from i.sam.opencl (bias, GELU, residual fused)
│   ├── softmax.cl              # row softmax over 4608 scores (i.sam softmax_relpos minus rel-pos)
│   └── pool_tsc.cl             # mean over T·S → 192×16×16
├── tools/
│   ├── export_norm.py          # normalisation constants → norm.tsv
│   ├── make_golden.py          # PyTorch CPU reference tensors for fixed crops (§8)
│   └── export_ckpt.py          # head weights → Lightning .ckpt via the real RslearnLightningModule
└── tests/
    ├── test_kernels.c          # per-kernel vs CPU C reference
    └── test_golden.sh          # end-to-end vs make_golden.py outputs
```

Name of the binary: `olmo_cl`.

## 5. Encoder forward, step by step (one 64×64 crop)

Input: `x[t][c][64][64]` float, t=0..5, c=0..11, normalised (§2).

1. For each band set s∈{0,1,2} (channels {0-3}, {4-9}, {10,11}) and each t:
   bicubic-resize the s channels to 128×128 (`F.interpolate(mode="bicubic",
   antialias=True, align_corners=False)`). With antialias on an upsample, the
   kernel support is unchanged, but the separable weights must match
   PyTorch's `_upsample_bicubic2d_aa` exactly. Validate with a golden test.
2. Conv 8×8 stride 8 → tokens `tok[s][t][16][16][192]` (weights
   `patch_embeddings.per_modality_embeddings.sentinel2_l2a.sentinel2_l2a__{s}.proj.{weight,bias}`).
   Implement as a GEMM: im2col `(256·6) × (|s|·64)` × `(|s|·64) × 192`.
3. Add encodings (precomputed once on the CPU, identical for every crop):
   channel embed (learnable table `[3][48]`, from the weights), time sincos
   `pos_embed[t]`, month table row `t` (legacy timestamps), 2D spatial sincos
   with grid scaled by 4.0. Watch the `meshgrid(indexing="xy")` h/w order.
4. Flatten to `X[4608][192]` in the order produced by
   `collapse_and_combine_hwtc` (b, h, w, t, s). Because attention is
   permutation-equivariant, the order only matters for the final reshape.
   Keep it identical anyway to simplify golden comparisons.
5. 12 × Block:
   `X += proj(attn(LN1(X)))`, `X += fc2(GELU(fc1(LN2(X))))`
   - qkv: 192→576 GEMM + bias; split into 3 heads × 64.
   - attention: S = QKᵀ/8 (batched `gemm`, 3 heads → 3 × 4608² scores,
     255 MB), row softmax, O = S·V (`gemm`). This is the pattern proven by
     i.sam.opencl; no fused kernel.
   - proj 192→192, fc1 192→768, fc2 768→192.
6. Final LayerNorm (`encoder.norm`).
7. Mean over the 18 (t,s) tokens of each spatial cell → `F[192][16][16]`.

FLOPs per crop: ≈ 20.5 GFLOP/block × 12 ≈ **250 GFLOP**. Attention
(2 × 8.2 GFLOP/block) dominates, so the attention kernel is where tuning pays.

## 6. Head and training loop (CPU side, per sample)

    F = encoder(crop)                      # GPU, 192×16×16
    L16 = W·F + b                          # 2×16×16
    L64 = bilinear_up4(L16)                # align_corners=False, edge clamp
    P = softmax(L64)
    loss = Σ_valid CE / Σ_valid            # valid = label != 255
    dL64 = (P - onehot) · valid / Σvalid
    dL16 = bilinear_up4ᵀ(dL64)             # adjoint (scatter of the same weights)
    dW = dL16 · Fᵀ ; db = Σ dL16
    AdamW step (lr, β1 .9, β2 .999, eps 1e-8, wd 0.01 decoupled, bias corrected)

- Init W, b like PyTorch `Conv2d` defaults: U(±1/√192) (kaiming_uniform a=√5
  for W, and the same bound for b). Seeded.
- Per epoch: shuffle the 58 train windows, draw crop offset `(ox,oy)` ∈
  `[0, w-64]×[0, h-64]` and two coin flips; apply the flips to the image
  **before** the encoder (the ViT is not flip-equivariant) and to the label.
- Validation per epoch: 8 val windows at the **fixed crops of
  `data/val_test_crops.tsv`** (the ones rslearn uses), eval mode (no stochastic
  depth), no flip. The encoder is frozen and these crops never change, so the
  8 val feature maps are computed once and cached for all epochs. Accumulate the 2×2
  confusion matrix over valid pixels; argmax of P. F1/precision/recall for
  class 1 (torchmetrics `average=None`, `class_idx=1`).
- Plateau scheduler on the epoch-mean train loss, with PyTorch
  `ReduceLROnPlateau` semantics (mode min, threshold 1e-4 rel, patience 2,
  cooldown 10, factor 0.2).
- After each epoch, write `head_epoch{E}.bin` (386 floats) and append
  `{epoch, step, train_loss, val_loss, val_f1, val_prec, val_rec, lr}` to
  `run_log.jsonl`. Track the top-3 by val F1, plus last.
- Stochastic depth in training (as the reference): for each train sample and
  block, draw keep ~ Bernoulli(0.9) separately for the attention and the MLP
  branch. When a branch is dropped, skip it entirely, which saves ≈ 10 % of
  the compute. When it is kept, scale it by 1/0.9 through the GEMM `alpha`,
  with a pre-scaled copy of the proj/fc2 bias.

## 7. Data pipeline (C, GDAL)

- Discover `windows/random_split/*/metadata.json`, read `options.split`
  (a tiny JSON parse: a vendored `cJSON` or a hand parser for the one key).
- Per window, read `layers/sentinel2{,.1..5}/B02_…_B09/geotiff.tif`
  (12 × UInt16) and `layers/label/label/geotiff.tif` (Byte, 255 = nodata),
  all into RAM once. 250 MB total, trivial.
- Verify that the t-order matches rslearn's `load_all_layers` ordering
  (`sentinel2`, `.1`, … `.5`). Confirm in `make_golden.py` by dumping the
  tensor rslearn builds.
- Fail loudly on: missing layer, band count ≠ 12, size mismatch between S2
  and label, window < 64 px.

## 8. Validation strategy (golden tensors from PyTorch on CPU)

`tools/make_golden.py` runs the real rslearn/OlmoEarth modules on the CPU in
FP32 (autocast disabled) for 3 fixed crops (one per split, fixed offsets and
flips), and dumps: the normalised input, the resize output, the patch-embed
tokens, the post-encoding tokens, the output after blocks 0/5/11, the final
norm, the pooled `F`, and logits, loss and gradients for a given `W,b`. It
also prints the LayerNorm eps and module names actually used.

Acceptance thresholds (FP32 vs FP32):

| Tensor | max rel. error |
|---|---|
| resize, patch-embed, encodings | 1e-5 |
| after block 11 / pooled F | 1e-3 |
| logits, loss, dW | 1e-4 given identical F |

Then, end to end:

1. `olmo_cl train` for 80 epochs on the server GPU.
2. `export_ckpt.py` → `.ckpt`.
3. Load it with the unchanged pipeline on the CPU: run
   `ALMA_Inference_Wheat_Festival.sh` (with `precision: 32`) and compare the
   result raster and `field_stats.csv` with `inference_outputs/20260604/`
   from the lost `epoch=12-step=754` checkpoint.
4. Also run PyTorch `validate` on the val split with the exported ckpt. Val
   F1 must match what `olmo_cl` logged, to within 0.5 %. This proves the
   export mapping is right.

## 9. Performance budget (server GPU)

- Memory per crop: X 4608×192×4 B = 3.5 MB; MLP hidden 4608×768×4 = 14 MB;
  QKV 10.6 MB; attention scores 255 MB; weights 22 MB (5.5 M params FP32);
  inputs < 1 MB. **About 0.3 GB of the 16 GiB.**
- Throughput, anchored on a measurement: i.sam.opencl runs a ViT-H encoder
  (≈ 3 TFLOP) in 3.8 s on this GPU under rusticl, ≈ 0.8 TFLOPS effective
  with the same gemm kernels. At 250 GFLOP per crop, that gives
  **≈ 0.3 s per crop**. Tiny's narrow matrices (D=192) may use the GPU less
  well, so budget 0.3–0.6 s.
- Per epoch: 66 crops ≈ 20–40 s. **80 epochs ≈ 30–55 min**, versus 3–7 h on
  this laptop's CPU.
- Development on this laptop: PoCL (`cpu-skylake-avx512`) is installed and
  runs the same kernels, which is slow but correct. Use it for golden tests
  before Friday.

## 10. Checkpoint export (`tools/export_ckpt.py`, CPU-only)

1. Build the real module from `olmoearth_run_data/wheat_festival/model.yaml`
   (the `model:` section via jsonargparse / LightningCLI instantiation), which
   loads the Tiny encoder weights from HF, exactly as in production.
2. Find the decoder conv parameters **by walking `state_dict()`** (expected
   under `model.decoders.wheat_seg.1.…weight/bias`; do not hardcode the
   names). Assert the shapes `[2,192,1,1]` / `[2]`, then load the 386 floats.
3. Save with Lightning's own format:
   `{"state_dict", "pytorch-lightning_version", "epoch", "global_step",
   "hyper_parameters"(from model.yaml), "callbacks": {}, "loops": …}`.
   Easiest robust route: attach the module to a `Trainer(accelerator="cpu",
   logger=False)` and call `trainer.save_checkpoint(path, weights_only=True)`,
   so the keys are whatever this Lightning version expects.
4. Filenames: `epoch={E}-step={S}.ckpt` for the top-3, plus `last.ckpt`,
   in `${ALMA_SCRATCH}/trainer_checkpoints/`, the same place the shell script
   would have used.
5. Hand the dev team: the chosen `.ckpt`, `olmoearth_run_data/wheat_festival/`
   (config), and `run_log.jsonl`.

## 11. Phases and schedule

| Phase | Where | Work | Exit criterion |
|---|---|---|---|
| 0 | laptop, now | Optional: start the CPU PyTorch fallback run (`precision: 32`, `max_epochs: 20`) in the background as insurance | ckpt exists |
| 1 | laptop | Copy `ocl_backend`, `pth_loader`, gemm/layernorm/softmax kernels from i.sam.opencl; load HF `weights.pth` with `pth_loader` (check tensor names/shapes); `export_norm.py`, `make_golden.py`; dataset loader in C | weights load; golden tensors exist |
| 2 | laptop (PoCL) | New kernels only: bicubic resize, encodings add, T·S pooling; wire up the 12 blocks with the reused gemm/softmax; `test_kernels.c` | pooled F matches golden ≤ 1e-3 on PoCL |
| 3 | laptop (PoCL) | head + AdamW + plateau + metrics + training loop; 2-epoch smoke run on 5 windows | loss decreases; dW matches golden |
| 4 | server, Fri 2 Oct | rsync source; build on host; `RUSTICL_ENABLE=radeonsi olmo_cl selftest platform=rusticl`; golden tests on the GPU; profile (env var) | golden passes on GPU; ≤ 0.6 s/crop |
| 5 | server | full 80-epoch run; `export_ckpt.py`; PyTorch CPU validation cross-check (§8.4) | val F1 agreement ≤ 0.5 % |
| 6 | laptop | inference with the exported ckpt via the unchanged pipeline; compare with the 2026-06-04 outputs; hand-off package | dev team receives ckpt + config + log |

Phases 1–3 are laptop-only and fit before Friday. With the i.sam.opencl
reuse, most of the new code is the OlmoEarth-specific glue (resize,
encodings, pooling, head/training). The server runtime is already known to
work, so phase 4 is mostly tuning.

## 11a. Status

**Phase 1: done (2026-09-29).**

- Tree set up. `ocl_backend`, `pth_loader`, the GEMM/LayerNorm kernels and
  the launch helpers are ported from i.sam.opencl, with GRASS calls replaced
  by `src/util.[ch]`. A new `softmax_rows` kernel was added. Builds with no
  warnings (`-Wall -Wextra`); clang-format uses GRASS's `.clang-format`.
- `olmo_cl info|weights`: `pth_loader` reads the HF `weights.pth` (all 552
  zip entries stored, 546 tensors). The S2 encoder uses 5.49 M parameters.
  q/k/v are separate `[192,192]` tensors, fused at load into `[576,192]`.
  The time `pos_embed` and `month_embed` tables are **stored in the file**, so
  only the 2D spatial encoding has to be computed. All LayerNorm eps = 1e-5.
- `olmo_cl data`: the GDAL loader reads the 74 windows (58/8/8), in rslearn's
  readdir layer order (`order=index` is available for comparison).
- `olmo_cl selftest` on PoCL, at the full 4608 tokens: all linear shapes
  (qkv, fc1+GELU, fc2+residual), 3-head attention in the strided qkv layout,
  and LayerNorm match double-precision C to ≤ 3.7e-6.
- `tools/make_golden.py` builds the real `RslearnLightningModule` and
  `RslearnDataModule` from `model.yaml` and dumps 4 eval-mode samples
  (val0/1, test0/1) plus one train-mode sample with its 24 drop decisions
  recorded. It also exports `data/val_test_crops.tsv`. The dumped model input
  equals raw GeoTIFF + readdir order + `data/s2_norm.tsv` normalisation to
  7.8e-8, and the label crop is identical, so the C data path is confirmed.
- Weights: `~/models/olmoearth/OlmoEarth-v1-Tiny/{weights.pth,config.json}`.

**Phase 2 findings (verified against golden in numpy before coding):**

- The resize is PyTorch's `_upsample_bicubic2d_aa`, which uses the **PIL
  cubic a = -0.5** (not the -0.75 of plain bicubic). Taps
  `[trunc(c - 2 + .5), trunc(c + 2 + .5))` with `c = (i + .5)·in/out`, clipped
  at the borders and renormalised. Matches to 1.7e-7.
- The spatial encoding slot is `sincos(col·4, 24) ++ sincos(row·4, 24)`,
  with the column first: `meshgrid(indexing="xy")` makes `grid[0]` the
  column index. Frequencies are `1/10000^(i/24/2)`. Matches to 2e-6.

**Phase 2: done (2026-09-29).** `src/encoder.c`: resize and im2col on the
host (OpenMP), patch-embed GEMMs writing straight into the (h,w,t,s) token
order, a constant encoding table added by `add_inplace`, 12 blocks
(LN → fused qkv → per-head QKᵀ/softmax/SV through strided `gemm` → proj +
residual; LN → fc1+GELU → fc2 + residual), final LN, then pooling on the
host. Stochastic depth: a dropped branch is skipped, a kept one uses
`alpha = 1/0.9` and a pre-scaled bias.

`olmo_cl check` (`make check`) on PoCL, full C data path (GeoTIFF → crop →
normalise):

| golden | input | resize | patch/encoded | block 0 → 11 | norm | features |
|---|---|---|---|---|---|---|
| val0 (eval) | 6.0e-8 | ≤ 3.0e-7 | 1.1e-6 | 1.4e-6 → 6.2e-6 (max 1.1e-5) | 4.0e-6 | 6.8e-7 |
| test0 (eval) | 5.8e-8 | ≤ 3.6e-7 | 1.0e-6 | from 1.4e-6 | 3.6e-6 | 5.5e-7 |
| train_mode (drops replayed) | 6.0e-8 | ≤ 3.0e-7 | 1.2e-6 | from 1.3e-6 | 4.3e-6 | n/a |

PoCL speed: 2.5 min per crop (gemm_big 135 s, softmax 7 s), which is fine
for tests only.

**Phase 3: done (2026-09-29).**

- `src/head.c`: 1×1 conv at 16×16, torch bilinear ×4 (align_corners=False),
  softmax-CE over valid pixels, exact adjoint for dW/db, and a confusion
  matrix. `make headcheck`: probs, loss, dW and db match PyTorch autograd
  within 1.2e-6 on the 4 eval goldens.
- `src/optim.c`: AdamW (torch defaults: wd 0.01, β 0.9/0.999, eps 1e-8)
  and ReduceLROnPlateau (rel threshold 1e-4, factor 0.2, patience 2,
  cooldown 10). `make test_optim`: AdamW matches torch to 3e-8 (float64
  state on both sides), and the plateau schedule is identical.
- `src/train.c` + `olmo_cl train`:
  - PCG32-seeded shuffle, uniform crop offsets, two flip coins, and 24
    stochastic-depth draws per sample; batch 1, one AdamW step per sample.
  - Val on the fixed rslearn crops, encoded once.
  - Lightning-style `epoch=E-step=S.head` for the top 3 by val F1 (a new
    score must strictly beat the worst kept), plus `last.head`.
  - `run_log.jsonl`, and `run_config.json` with the device and the
    timestep order. Refuses to overwrite a run without `overwrite=1`.
  - `augment=0` (centre crops, eval-mode encoder, cached features) is for
    fast debugging runs.
- Smoke run on PoCL (`augment=0`, 3 train and 2 val windows, lr 1e-3,
  150 epochs): train loss 0.465 → 0.0029, val loss 0.678 → 0.100, lr
  never decayed (loss kept improving). Val F1 is 0 because those 2 val
  windows hold no wheat. **Every window is all wheat or all background**
  (one field polygon each): 3 of the 8 val crops are background-only, so
  F1 is only meaningful over the full val set.
- `tools/export_ckpt.py` (pulled forward from phase 5): head file →
  `RslearnLightningModule` → `Trainer.save_checkpoint`. Checked: the head
  lands in `model.decoders.wheat_seg.1.layer.{weight,bias}`, with the
  right epoch and global_step, 233 tensors. `--validate` (Lightning
  validate with the ckpt, the §8.4 cross-check) is written but not run
  yet.

**Next, phase 4 (server, Fri 2 Oct):**
```
rsync -a --exclude build --exclude golden ~/dev/OLMO_opencl/ yann@10.42.0.89:~/dev/OLMO_opencl/
rsync -a ~/dev/OLMO_opencl/golden/ yann@10.42.0.89:~/dev/OLMO_opencl/golden/   # 283 MB, for make check
rsync -a ~/models/olmoearth/ yann@10.42.0.89:~/models/olmoearth/
rsync -a ~/RSDATA/alma_wheat_festival/dataset/ yann@10.42.0.89:~/RSDATA/alma_wheat_festival/dataset/
# on the server:
export RUSTICL_ENABLE=radeonsi
make && make selftest DEVICE=gpu PLATFORM=rusticl && make check DEVICE=gpu PLATFORM=rusticl
OLMO_CL_PROFILE=1 ./olmo_cl train ... device=gpu platform=rusticl out=runs/r1
```
Watch out: after rsync, the server's readdir order of the `sentinel2*`
layer directories may differ. `olmo_cl data` prints it. Use
`order=index` only if the order must be forced; the model is only
consistent with the order it was trained on.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Bicubic-antialias weights differ subtly from PyTorch | Golden test on the resize alone; port the weight formula from PyTorch's `UpSampleKernel` / `_upsample_bicubic2d_aa` source (ask to clone `pytorch` into `~/dev/` if needed) |
| rslearn's readdir timestep order differs on another filesystem (server after rsync, dev team's machine), so inference sees different time/month encodings than training did | olmo_cl logs the order and has `order=`. The trained model is only consistent with the order it was trained on. Tell the dev team; an upstream fix would sort by group index |
| Token order / encoding-slot mix-up | Golden after encodings; attention's permutation-equivariance means only the final reshape is at risk, which the pooled-F test catches |
| rusticl not enabled → silently runs on Clover or PoCL | Log platform + device at start-up; `platform=rusticl` to force it; README states `RUSTICL_ENABLE=radeonsi` (as in i.sam.opencl). Clover (OpenCL 1.1) is a working fallback because the kernels stay 1.1-compatible |
| `pth_loader` meets a zip layout it doesn't handle in the OlmoEarth file (e.g. compressed entries) | Phase-1 check; worst case, re-save once with `torch.save(sd, f, _use_new_zipfile_serialization=True)` on CPU |
| Numeric drift vs the original bf16/fp16 run | Accept; judge by val F1 and the inference comparison, not bitwise |
| Lightning ckpt incompatibility | Export through the real module + `Trainer.save_checkpoint`, then test by loading with the unchanged inference script |
| Scope creep (unfreezing the encoder later) | Out of scope. Full backprop through the ViT is a much larger project; with 8–16 GB it would be simpler to revisit a PyTorch path |

## 13. Out of scope

- Encoder fine-tuning (backward through the ViT).
- Inference/prediction in C. The existing `olmoearth_run` pipeline does it
  from the exported ckpt. A later `olmo_cl predict` could reuse the encoder
  (sliding 64-px windows, overlap 8, merge padding 2) if CPU inference is too
  slow.
- Other modalities (S1, Landsat) and other OlmoEarth sizes. The weight
  manifest is generic, so Base (D=768, 12 heads) would be a config change
  plus memory re-budgeting.
