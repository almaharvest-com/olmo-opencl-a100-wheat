# OLMO_opencl

`olmo_cl` fine-tunes the ALMA Sharjah wheat-field segmentation model on
an OpenCL device, with no CUDA and no PyTorch at run time. It does the
same job as `ALMA_Train_Wheat_Festival.sh` in
`~/dev/olmoearth_projects`. It reproduces the rslearn / Lightning recipe of
`olmoearth_run_data/wheat_festival/model.yaml`: a frozen OlmoEarth-v1-Tiny
encoder with a 1×1 convolution head trained on it. A small Python script
then writes a standard Lightning `.ckpt`, which the existing
`olmoearth_run` inference pipeline loads unchanged.

It exists because the CUDA GPU that ran the PyTorch pipeline (a Quadro
P1000) is gone. The available GPU is an AMD Radeon Pro WX 7100 (Polaris,
16 GiB) on `yann@10.42.0.89`, which is reachable through Mesa's rusticl
OpenCL driver.

## Why this is small

The encoder stays frozen for the whole run (`unfreeze_at_epoch: 9999`), so
the only trainable parameters are the head's 2 × 192 weights and 2
biases, 386 numbers. The GPU only runs the encoder **forward** pass (about
250 GFLOP per 64 × 64 crop, 4608 tokens). The head, its gradients, AdamW
and the learning-rate schedule run on the CPU in microseconds.

## Status (2026-09-29)

| Phase | What | State |
|---|---|---|
| 1 | Weight loader, dataset loader, reused kernels | done |
| 2 | Encoder forward in OpenCL | done: matches PyTorch |
| 3 | Head, AdamW, plateau schedule, training loop, checkpoint export | done: matches PyTorch/torch |
| 4 | Build, test and tune on the AMD GPU | Friday 2 October |
| 5 | Full 80-epoch run, export, PyTorch cross-validation | after phase 4 |
| 6 | Inference with the exported `.ckpt`, hand-off to the dev team | after phase 5 |

Everything so far was validated on this laptop with PoCL (CPU OpenCL).
See [PLAN.md](PLAN.md) for the design and the full log of findings.

### Accuracy against the PyTorch reference

Measured with `make check` / `make test`, float32, full C data path
(GeoTIFF → crop → normalise). Values are relative errors.

| Stage | val0 | test0 | train mode¹ |
|---|---|---|---|
| input crop | 6.0e-8 | 5.8e-8 | 6.0e-8 |
| antialiased bicubic resize | ≤ 3.0e-7 | ≤ 3.6e-7 | ≤ 3.0e-7 |
| patch embedding + encodings | 1.2e-6 | 1.0e-6 | 1.2e-6 |
| 12 transformer blocks (worst) | 1.1e-5 | — | — |
| final LayerNorm | 4.0e-6 | 3.6e-6 | 4.3e-6 |
| pooled features | 6.8e-7 | 5.5e-7 | — |
| head: probabilities, loss, dW, db | ≤ 1.2e-6 (4 samples) | | |

¹ Stochastic depth on, with PyTorch's recorded drop decisions replayed.

AdamW matches `torch.optim.AdamW` to 3e-8 over 300 steps.
ReduceLROnPlateau reproduces torch's schedule exactly.

## Requirements

- An OpenCL 1.2 (or 1.1) platform and headers (`ocl-icd-opencl-dev`), for
  example Mesa rusticl or Clover for AMD GPUs, or PoCL for CPUs.
- GDAL development files (`libgdal-dev`), a C11 compiler with OpenMP, and
  `pkg-config`.
- For the Python tools only (golden dumps, checkpoint export, optimiser
  test): the `~/dev/olmoearth_projects` venv (torch, rslearn,
  olmoearth_pretrain).

Inputs:

| What | Default path |
|---|---|
| OlmoEarth-v1-Tiny weights (57 MB, from `allenai/OlmoEarth-v1-Tiny` on Hugging Face) | `~/models/olmoearth/OlmoEarth-v1-Tiny/weights.pth` |
| rslearn dataset (74 windows, 250 MB) | `~/RSDATA/alma_wheat_festival/dataset` |
| Normalisation constants (shipped) | `data/s2_norm.tsv` |
| Fixed val/test crops of rslearn (shipped) | `data/val_test_crops.tsv` |

To download the weights:

```sh
mkdir -p ~/models/olmoearth/OlmoEarth-v1-Tiny && cd $_
curl -L -O https://huggingface.co/allenai/OlmoEarth-v1-Tiny/resolve/main/weights.pth \
     -O https://huggingface.co/allenai/OlmoEarth-v1-Tiny/resolve/main/config.json
```

## Build and test

```sh
make                 # ./olmo_cl
make test            # head vs PyTorch + AdamW/plateau vs torch (seconds)
make selftest        # OpenCL kernels vs C loops at full size (DEVICE=, PLATFORM=)
make check           # whole encoder vs PyTorch dumps: 2.5 min per sample on PoCL
```

`make check` needs `golden/` (283 MB, not versioned). `make golden`
regenerates it from the real PyTorch model, together with
`data/val_test_crops.tsv`.

On the GPU server:

```sh
export RUSTICL_ENABLE=radeonsi    # without it, rusticl hides the GPU
make selftest DEVICE=gpu PLATFORM=rusticl
make check    DEVICE=gpu PLATFORM=rusticl
```

`olmo_cl` prints the OpenCL device and platform it uses. Check that it
says rusticl and the Radeon, not Clover or PoCL.

## Training and hand-off

### Launch script (same layout as the old PyTorch run)

```sh
bash ALMA_Train_Wheat_Festival_opencl.sh                    # ~/RSDATA/alma_wheat_festival
bash ALMA_Train_Wheat_Festival_opencl.sh /path/to/scratch
```

This is the OpenCL counterpart of `ALMA_Train_Wheat_Festival.sh` (v0.0.3)
in `~/dev/olmoearth_projects`. It uses the same scratch directory and
writes to the same `trainer_checkpoints/`: `epoch=E-step=S.ckpt` for the 3
best epochs by val wheat F1, plus `last.ckpt`, `run_log.jsonl`,
`run_config.json` and `train.log`. It then prints the
`ALMA_CHECKPOINT_PATH` to use for inference. Its steps:

1. Check the dataset, model.yaml and the venv; download the Tiny weights
   if they are missing.
2. Build `olmo_cl` if needed, then run the OpenCL self-test on the chosen
   device (the old script checked for CUDA instead).
3. Report the splits and timestep order, train, export every head to a
   Lightning `.ckpt`, and report the best one.

Defaults are `DEVICE=gpu PLATFORM=rusticl EPOCHS=80 SEED=42`, with
`RUSTICL_ENABLE=radeonsi` exported. Set `VALIDATE=1` to re-validate each
`.ckpt` with the PyTorch model, `OVERWRITE=1` to replace an earlier run,
and `OLMO_CL_EXTRA="max_train=2 max_val=1"` for quick tests.

Compared with the old script, three things are corrected:
- Its split-count `grep` never matched the metadata format, so it printed
  nothing.
- It announced "encoder unfrozen after 15 epochs", but the config keeps
  it frozen.
- It looked for `best-*.ckpt`, which Lightning never writes.

### With make

```sh
make train RUN=runs/r1 DEVICE=gpu PLATFORM=rusticl      # EPOCHS=80 SEED=42
make ckpt  RUN=runs/r1 VALIDATE=1                       # .head -> .ckpt (CPU)
```

`runs/r1/` then holds:

| File | Content |
|---|---|
| `epoch=E-step=S.head` / `.ckpt` | the 3 best epochs by val wheat F1 (Lightning naming) |
| `last.head` / `last.ckpt` | the last epoch |
| `run_log.jsonl` | one line per epoch: losses, val F1/precision/recall/accuracy, lr, time |
| `run_config.json` | command line, OpenCL device, **timestep order**, head layout |

With `VALIDATE=1`, each checkpoint is also run through Lightning's
`validate()` with the production PyTorch model. Its val wheat F1 should
match `run_log.jsonl`.

For the dev team, send the chosen `.ckpt`,
`~/dev/olmoearth_projects/olmoearth_run_data/wheat_festival/`, and
`run_log.jsonl`. The `.ckpt` loads with `ALMA_Inference_Wheat_Festival.sh`
(point its `CHECKPOINT=` variable, line 40, at the file).

## Commands

```
olmo_cl info      weights=FILE                       list a state_dict
olmo_cl weights   weights=FILE [heads=3]             load/check the S2 encoder
olmo_cl data      dataset=DIR [order=readdir|index]  summarise the windows
olmo_cl selftest  [device=] [platform=] [tokens=]    kernels vs C loops
olmo_cl check     golden=DIRS weights=FILE [dataset=DIR]
olmo_cl headcheck golden=DIRS
olmo_cl train     dataset=DIR weights=FILE out=DIR [epochs=80] [lr=1e-4]
                  [seed=42] [top_k=3] [augment=1] [max_train=0] [max_val=0]
                  [init=HEAD] [order=readdir|index] [overwrite=0]
                  [device=auto|gpu|cpu] [platform=NAME]
```

`device=auto` takes the first GPU, otherwise a CPU. `platform=` restricts
to platform names containing that text (`rusticl`, `Clover`, `Portable`).
`augment=0` is a debugging mode: centre crops, no flips, no stochastic
depth, and train features cached after the first epoch.

Environment: `OLMO_CL_VERBOSE=0..3`, and `OLMO_CL_PROFILE=1` for
per-kernel timing.

## What is reproduced, and what is not

Reproduced from the reference (rslearn, olmoearth_pretrain, Lightning):

- **Data:** 12 S2 bands × 6 timesteps, normalised with
  `(x - (mean - 2 std)) / (4 std)`.
- **Train crops:** a random 64 × 64 crop per window and epoch, plus random
  horizontal and vertical flips.
- **Val crops:** fixed per window, the same ones rslearn picks.
- **Encoder:**
  - PyTorch's antialiased bicubic resize 64 → 128, then an 8 × 8 patch
    embedding per band set.
  - Channel, time, month and 2D spatial encodings.
  - 12 pre-norm ViT blocks, a final LayerNorm, and the mean over
    timesteps and band sets.
  - **Stochastic depth (p = 0.1) during training:** the frozen encoder
    stays in train mode in the reference, because rslearn's freeze
    callback never calls `.eval()`.
- **Head:** bilinear ×4, a 1×1 convolution 192 → 2, and softmax
  cross-entropy on valid pixels (label 255 is nodata).
- **Optimisation:** AdamW with lr 1e-4 and weight decay 0.01, batch 1, and
  ReduceLROnPlateau on the epoch train loss (factor 0.2, patience 2,
  cooldown 10).
- **Selection:** top 3 by val wheat F1, plus last.

Not bit-identical: the reference ran the encoder under bf16 autocast and
the head under fp16 mixed precision, with its own random streams.
`olmo_cl` is float32 throughout, with a seeded PCG32 stream. Expect
similar, not identical, val F1.

## Caveat: timestep order

rslearn stacks a window's `sentinel2`, `sentinel2.1` … layers in raw
directory order (`Path.iterdir()`), not by group index. On this laptop
the order is `.5, .4, .3, (0), .1, .2` in all windows. The legacy
timestamps make each position's time and month encoding depend on that
order.

`olmo_cl` reproduces the order (`order=readdir`) and records it in
`run_config.json`. A copy of the dataset on another filesystem (the
server after rsync, the dev team's machine) can list the directories in
a different order. The model is then fed time encodings it was not
trained with, and no error is raised. This affects the original PyTorch
pipeline in exactly the same way. `order=index` forces the sorted order
(0, 1, … 5).

## Layout

```
ALMA_Train_Wheat_Festival_opencl.sh   launcher (old-script layout)
src/        olmo_cl sources (C11)
  main.c        command line
  encoder.c     encoder forward (host resize/im2col, OpenCL blocks)
  head.c        head, loss, gradients, metrics
  train.c       training loop, checkpoints, logs
  optim.c       AdamW, ReduceLROnPlateau
  dataset.c     rslearn windows via GDAL, crops, flips, normalisation
  weights.c     OlmoEarth S2 encoder weights from weights.pth
  ops.c         kernel launchers        (from i.sam.opencl)
  ocl_backend.c device selection, build (from i.sam.opencl)
  pth_loader.c  torch.save zip reader   (from i.sam.opencl)
  check.c, selftest.c   comparisons with PyTorch / C loops
kernels/    olmo_kernels.cl (GEMM, LayerNorm from i.sam.opencl; softmax, add)
tools/      make_golden.py, export_ckpt.py, export_norm.py (Python, venv)
tests/      test_optim.[c|py]
data/       s2_norm.tsv, val_test_crops.tsv
golden/     PyTorch reference dumps (generated, not versioned)
```

## Credits and licence

The OpenCL backend, the PyTorch checkpoint reader, the GEMM and LayerNorm
kernels, and their launchers come from `~/dev/i.sam.opencl` (same author),
with its GRASS calls replaced by `src/util.c`. OlmoEarth is by the Allen
Institute for AI; rslearn and olmoearth_pretrain are its reference
implementation.

Written with AI assistance (Claude). Every numerical component was checked
against the PyTorch reference, as listed above.

Released into the public domain under the Unlicense, see
[LICENSE](LICENSE) (`SPDX-License-Identifier: Unlicense` in every source
file). The code taken from i.sam.opencl is relicensed here by its author.
