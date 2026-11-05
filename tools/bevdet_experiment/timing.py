"""Inference timing helpers."""

import time

import torch

class _CUDATimer:
    def __init__(self, enabled=True):
        self.enabled = bool(enabled) and torch.cuda.is_available()
        if self.enabled:
            self.st = torch.cuda.Event(enable_timing=True)
            self.ed = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.enabled:
            torch.cuda.synchronize()
            self.st.record()
        else:
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            self.ed.record()
            torch.cuda.synchronize()
            self.ms = float(self.st.elapsed_time(self.ed))
        else:
            self.ms = (time.perf_counter() - self.t0) * 1000.0
