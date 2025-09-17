import numpy as np
import random

class ClippingAugmentation:
    def __init__(self, clip_range=(0.3, 0.9)):
        self.clip_range = clip_range

    def __call__(self, waveform: np.ndarray):
        assert len(waveform.shape) == 1
        clip_threshold = random.uniform(*self.clip_range)
        max_val = np.max(np.abs(waveform))
        threshold = max_val * clip_threshold
        waveform = np.clip(waveform, -threshold, threshold)
        return waveform