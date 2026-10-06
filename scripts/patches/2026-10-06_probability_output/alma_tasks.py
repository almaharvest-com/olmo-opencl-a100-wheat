"""ALMA: wheat segmentation task that writes PROBABILITY instead of 0/1.

The rslearn installed on g13 is too old for `output_probs`, so this subclass
does the same thing for any rslearn version: process_output returns the wheat
class probability (one float32 band, 0..1) instead of the argmax class id.
Training and the checkpoint are unaffected (the task has no weights).

Use in model.yaml:
    class_path: alma_tasks.WheatProbSegmentationTask
and keep this file next to ALMA_Inference_Wheat_Festival.sh (it puts that
folder on PYTHONPATH).
"""
import numpy as np
import torch
from rslearn.train.tasks.segmentation import SegmentationTask

WHEAT_CLASS = 1


class WheatProbSegmentationTask(SegmentationTask):
    def process_output(self, raw_output, metadata):
        x = raw_output if torch.is_tensor(raw_output) else torch.as_tensor(raw_output)
        x = x.detach().float()
        if x.dim() != 3:
            raise ValueError("WheatProbSegmentationTask expects a CHW tensor")
        # SegmentationHead already returns softmax probabilities; if this
        # rslearn passes logits instead, convert them.
        if x.min() < 0 or x.max() > 1.0001 or not torch.allclose(
                x.sum(dim=0), torch.ones_like(x[0]), atol=1e-3):
            x = torch.softmax(x, dim=0)
        return x[WHEAT_CLASS:WHEAT_CLASS + 1].cpu().numpy().astype(np.float32)
