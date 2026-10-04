#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Turn an olmo_cl head file into a Lightning checkpoint for olmoearth_run.

Builds the real RslearnLightningModule from the project's model.yaml
(which loads the frozen OlmoEarth encoder weights from Hugging Face, as
in production), writes the trained 1x1 conv of the wheat_seg decoder,
and saves the checkpoint through Lightning's own Trainer, so the file
has whatever keys this Lightning version expects.

With --validate, it then runs Lightning's validate() on the val split
with that checkpoint (CPU, production numerics) and prints the val
metrics, to be compared with olmo_cl's run_log.jsonl.

Usage (inside the olmoearth_projects venv):
    export_ckpt.py MODEL_YAML DATASET_DIR HEAD_FILE OUT_CKPT [--validate]

The epoch and step are read from a HEAD_FILE named
"epoch=E-step=S.head"; other names (last.head) give epoch 0, step 0
unless --epoch/--step are passed.
"""

import argparse
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_golden import build  # noqa: E402


def decoder_conv(module):
    """The single Conv2d of the wheat_seg decoder, checked."""
    convs = [m for m in module.model.decoders["wheat_seg"].modules()
             if isinstance(m, torch.nn.Conv2d)]
    if len(convs) != 1:
        sys.exit(f"expected one Conv2d in the wheat_seg decoder, "
                 f"found {len(convs)}")
    return convs[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_yaml")
    ap.add_argument("dataset_dir")
    ap.add_argument("head_file")
    ap.add_argument("out_ckpt")
    ap.add_argument("--epoch", type=int)
    ap.add_argument("--step", type=int)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out_ckpt) and not args.overwrite:
        sys.exit(f"{args.out_ckpt} exists (use --overwrite)")
    m = re.search(r"epoch=(\d+)-step=(\d+)", os.path.basename(args.head_file))
    epoch = args.epoch if args.epoch is not None else int(m[1]) if m else 0
    step = args.step if args.step is not None else int(m[2]) if m else 0

    import lightning.pytorch as pl

    module, data = build(args.model_yaml, args.dataset_dir)
    conv = decoder_conv(module)
    n_out, dim = conv.weight.shape[:2]
    if conv.weight.shape[2:] != (1, 1) or n_out != 2:
        sys.exit(f"unexpected decoder conv shape {tuple(conv.weight.shape)}")
    head = np.fromfile(args.head_file, dtype="<f4")
    if head.size != 2 * dim + 2:
        sys.exit(f"{args.head_file}: {head.size} floats, expected "
                 f"{2 * dim + 2} for D={dim}")
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(head[:2 * dim]).reshape(2, dim, 1, 1))
        conv.bias.copy_(torch.from_numpy(head[2 * dim:]))

    trainer = pl.Trainer(accelerator="cpu", logger=False,
                         enable_checkpointing=False, max_epochs=epoch + 1)
    trainer.strategy.connect(module)
    trainer.fit_loop.epoch_progress.current.completed = epoch
    # trainer.global_step counts optimizer steps.
    loop = trainer.fit_loop.epoch_loop
    loop.automatic_optimization.optim_progress.optimizer.step.total.completed = step
    loop._batches_that_stepped = step
    trainer.save_checkpoint(args.out_ckpt, weights_only=True)

    # Read back: the conv must hold the head, the rest the HF encoder.
    ck = torch.load(args.out_ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    key = [k for k, v in sd.items()
           if v.shape == conv.weight.shape and torch.equal(v, conv.weight)]
    print(f"wrote {args.out_ckpt}: epoch {ck.get('epoch')}, "
          f"global_step {ck.get('global_step')}, {len(sd)} tensors, "
          f"head at {key}")

    if args.validate:
        trainer = pl.Trainer(accelerator="cpu", logger=False,
                             enable_checkpointing=False)
        res = trainer.validate(module, datamodule=data,
                               ckpt_path=args.out_ckpt)
        for k, v in sorted(res[0].items()):
            print(f"{k}\t{v:.6f}")


if __name__ == "__main__":
    main()
