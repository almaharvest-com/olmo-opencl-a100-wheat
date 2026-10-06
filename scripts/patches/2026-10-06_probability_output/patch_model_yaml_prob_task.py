#!/usr/bin/env python3
"""Point the wheat_seg task in model.yaml at alma_tasks.WheatProbSegmentationTask.

Also removes the output_probs / output_class_idx lines added by
patch_model_yaml_output_probs.py (the g13 rslearn does not know them).
Keeps a copy as model.yaml.bak2. Safe to run twice.
Usage: python3 patch_model_yaml_prob_task.py olmoearth_run_data/wheat_festival/model.yaml
"""
import re, shutil, sys
p = sys.argv[1]
lines = open(p).read().splitlines(keepends=True)
shutil.copy2(p, p + ".bak2")
lines = [l for l in lines if not re.match(r"\s*(output_probs|output_class_idx):", l)]
n = 0
for i, l in enumerate(lines):
    if re.match(r"\s*class_path:\s*rslearn\.train\.tasks\.segmentation\.SegmentationTask\s*$", l):
        lines[i] = l.replace("rslearn.train.tasks.segmentation.SegmentationTask",
                             "alma_tasks.WheatProbSegmentationTask")
        n += 1
s = "".join(lines)
open(p, "w").write(s)
k = s.count("alma_tasks.WheatProbSegmentationTask")
print(f"replaced {n} line(s); model.yaml now names alma_tasks.WheatProbSegmentationTask {k} time(s) (must be 1)")
i = s.index("alma_tasks.WheatProbSegmentationTask")
print(s[s.rfind("\n", 0, i - 60):i + 120])
