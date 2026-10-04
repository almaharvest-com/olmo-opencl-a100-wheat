#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Dump PyTorch reference tensors for olmo_cl (golden tests).

Builds the real rslearn model and data module from the wheat_festival
model.yaml, exactly as the olmoearth_run training does, and runs them on
the CPU in float32 (encoder autocast disabled). For a few fixed crops it
writes, under OUT_DIR/<tag>/:

    input.f32         normalised S2 crop, [T][12][64][64] (olmo_cl layout)
    label.u8          [64][64], 0/1/255
    resized_s<s>.f32  bicubic 64 -> 128 input of band set s, [T][|s|][128][128]
    patch.f32         patch-embed tokens, [16][16][T][3][D]
    encoded.f32       tokens after the composite encodings, same layout
    block<i>.f32      tokens after block i, [N][D] (N = 16*16*T*3)
    norm.f32          after the final LayerNorm, [N][D]
    features.f32      pooled encoder output, [D][16][16]
    head_w.f32/.b     the decoder 1x1 conv weights used, [2][D] and [2]
    logits.f32        [2][64][64]
    loss.f32, dW.f32, db.f32   cross-entropy and its gradients
    meta.json         window, split, crop offset, shapes, layer order

A "train" tag repeats the forward pass in training mode (stochastic depth
on) with the drop decisions recorded in drops.json, so that olmo_cl can
replay them.

It also writes OUT_DIR/crops.tsv: the fixed val and test crop offsets
that rslearn uses (fix_patch_pick), one line per window.

Usage (inside the olmoearth_projects venv, CPU only):
    make_golden.py MODEL_YAML DATASET_DIR OUT_DIR
"""

import json
import os
import sys

import numpy as np
import torch
import yaml


def build(model_yaml, dataset_dir):
    """Instantiate RslearnLightningModule and RslearnDataModule."""
    from jsonargparse import ArgumentParser
    from rslearn.train.data_module import RslearnDataModule
    from rslearn.train.lightning_module import RslearnLightningModule

    os.environ.update(
        DATASET_PATH=dataset_dir,
        NUM_WORKERS="0",
        TRAINER_DATA_PATH="/nonexistent",
        PREDICTION_OUTPUT_LAYER="output",
    )
    with open(model_yaml) as f:
        cfg = yaml.safe_load(os.path.expandvars(f.read()))
    parser = ArgumentParser()
    parser.add_argument("--model", type=RslearnLightningModule)
    parser.add_argument("--data", type=RslearnDataModule)
    # As RslearnLightningCLI: the model gets the data module's task.
    parser.link_arguments("data.init_args.task", "model.init_args.task",
                          apply_on="instantiate")
    ns = parser.parse_object({"model": cfg["model"], "data": cfg["data"]})
    init = parser.instantiate_classes(ns)
    return init.model, init.data


def save(path, t):
    """Write a tensor or array as raw little-endian float32 (or uint8)."""
    a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    if a.dtype != np.uint8:
        a = a.astype("<f4")
    a.tofile(path)
    return list(a.shape)


def window_offset(meta):
    """Crop offset (x, y) inside the window, in pixels."""
    wb = meta.window_bounds
    pb = meta.patch_bounds
    return pb[0] - wb[0], pb[1] - wb[1]


class Recorder:
    """Forward hooks on the OlmoEarth encoder, plus DropPath recording."""

    def __init__(self, enc):
        self.out = {}
        self.drops = []
        model = enc.model  # olmoearth_pretrain Encoder
        # The encoder calls these two through .forward(), which bypasses
        # module hooks: wrap the methods instead.
        self._wrap(model.patch_embeddings, "patch")
        self._wrap(model.composite_encodings, "encoded")
        for i, blk in enumerate(model.blocks):
            blk.register_forward_hook(self._hook(f"block{i}"))
            blk.drop_path.forward = self._drop_path(blk.drop_path, i)
        model.norm.register_forward_hook(self._hook("norm"))

    def _wrap(self, mod, name):
        orig = mod.forward

        def forward(*args, **kwargs):
            out = orig(*args, **kwargs)
            self.out[name] = out
            return out

        mod.forward = forward

    def _hook(self, name):
        def fn(mod, args, out):
            self.out[name] = out

        return fn

    def _drop_path(self, mod, block):
        """DropPath.forward that also records its keep decisions."""
        calls = {"n": 0}

        def forward(x):
            if not mod.training or not mod.drop_prob:
                return x
            keep = 1.0 - mod.drop_prob
            r = (keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1),
                                   dtype=x.dtype)).floor_()
            branch = "attn" if calls["n"] % 2 == 0 else "mlp"
            calls["n"] += 1
            self.drops.append({"block": block, "branch": branch,
                               "keep": bool(r.flatten()[0] > 0)})
            return x.div(keep) * r

        return forward


def dump_sample(module, rec, sample, out_dir, tag, split, train_mode):
    from rslearn.train.model_context import ModelContext

    inputs, targets, meta = sample
    d = os.path.join(out_dir, tag)
    os.makedirs(d, exist_ok=True)
    enc = module.model.encoder[0]
    head = module.model.decoders["wheat_seg"]
    conv = head[1]
    conv_w = [p for n, p in conv.named_parameters() if n.endswith("weight")][0]
    conv_b = [p for n, p in conv.named_parameters() if n.endswith("bias")][0]

    module.train(train_mode)
    rec.out.clear()
    rec.drops.clear()
    for p in module.parameters():
        p.grad = None
    conv_w.requires_grad_(True)
    conv_b.requires_grad_(True)

    image = inputs["sentinel2_l2a"].image  # [C, T, H, W]
    x = image.permute(1, 0, 2, 3).contiguous()  # [T, C, H, W]
    shapes = {"input": save(os.path.join(d, "input.f32"), x)}
    tgt = targets["wheat_seg"]
    label = tgt["classes"].clone()
    label[tgt["valid"] == 0] = 255
    shapes["label"] = save(os.path.join(d, "label.u8"),
                           label.to(torch.uint8).numpy())

    # Reference bicubic resize, as FlexiPatchEmbed does for patch 4 < 8.
    for s, (a, b) in enumerate([(0, 4), (4, 10), (10, 12)]):
        r = torch.nn.functional.interpolate(
            x[:, a:b], size=(128, 128), mode="bicubic", antialias=True)
        shapes[f"resized_s{s}"] = save(os.path.join(d, f"resized_s{s}.f32"), r)

    context = ModelContext(inputs=[inputs], metadatas=[meta])
    outputs = module(context, [targets])
    loss = sum(outputs.loss_dict.values())
    loss.backward()

    shapes["patch"] = save(os.path.join(d, "patch.f32"),
                           rec.out["patch"]["sentinel2_l2a"][0])
    shapes["encoded"] = save(os.path.join(d, "encoded.f32"),
                             rec.out["encoded"]["sentinel2_l2a"][0])
    for i in range(len(enc.model.blocks)):
        shapes[f"block{i}"] = save(os.path.join(d, f"block{i}.f32"),
                                   rec.out[f"block{i}"][0])
    shapes["norm"] = save(os.path.join(d, "norm.f32"), rec.out["norm"][0])

    # Pooled features and logits, recomputed from the recorded tokens so
    # that they are exactly what the head saw.
    with torch.no_grad():
        feats = enc(context).feature_maps[0][0] if not train_mode else None
    if feats is not None:
        shapes["features"] = save(os.path.join(d, "features.f32"), feats)
    shapes["head_w"] = save(os.path.join(d, "head_w.f32"),
                            conv_w.reshape(2, -1))
    shapes["head_b"] = save(os.path.join(d, "head_b.f32"), conv_b)
    logits = outputs.outputs[0]["wheat_seg"]  # softmax probabilities
    shapes["probs"] = save(os.path.join(d, "probs.f32"), logits)
    save(os.path.join(d, "loss.f32"), loss.reshape(1))
    shapes["dW"] = save(os.path.join(d, "dW.f32"), conv_w.grad.reshape(2, -1))
    shapes["db"] = save(os.path.join(d, "db.f32"), conv_b.grad)

    ox, oy = window_offset(meta)
    info = {
        "window": meta.window_name,
        "split": split,
        "offset_x": int(ox),
        "offset_y": int(oy),
        "train_mode": train_mode,
        "loss": float(loss),
        "timestamps": [str(t) for t in inputs["sentinel2_l2a"].timestamps]
        if inputs["sentinel2_l2a"].timestamps else None,
        "shapes": shapes,
    }
    if train_mode:
        with open(os.path.join(d, "drops.json"), "w") as f:
            json.dump(rec.drops, f, indent=0)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"{tag}: {meta.window_name} ({split}) offset ({ox},{oy}) "
          f"loss {float(loss):.6f}")


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    model_yaml, dataset_dir, out_dir = sys.argv[1:]
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)

    module, data = build(model_yaml, dataset_dir)
    enc = module.model.encoder[0]
    enc.autocast_dtype = None  # float32 reference
    for p in enc.parameters():
        p.requires_grad_(False)  # frozen, as FreezeUnfreeze does
    rec = Recorder(enc)

    lines = []
    samples = {}
    for stage, split in [("validate", "val"), ("test", "test")]:
        data.setup(stage)
        ds = data.datasets[split]
        for i in range(len(ds)):
            sample = ds[i]
            meta = sample[2]
            ox, oy = window_offset(meta)
            lines.append(f"{meta.window_name}\t{split}\t{ox}\t{oy}")
            if i < 2:
                samples[f"{split}{i}"] = (sample, split)
    with open(os.path.join(out_dir, "crops.tsv"), "w") as f:
        f.write("# window\tsplit\toffset_x\toffset_y\n")
        f.write("\n".join(sorted(lines)) + "\n")
    print(f"crops.tsv: {len(lines)} fixed val/test crops")

    for tag, (sample, split) in samples.items():
        dump_sample(module, rec, sample, out_dir, tag, split, False)
    # Training-mode pass on the first val crop: stochastic depth active.
    torch.manual_seed(1)
    sample, split = samples["val0"]
    dump_sample(module, rec, sample, out_dir, "train_mode", split, True)


if __name__ == "__main__":
    main()
