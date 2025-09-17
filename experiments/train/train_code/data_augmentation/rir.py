import random
import numpy as np
import soundfile as sf
from scipy import signal

class RIRReverberation:
    def __init__(self, samples):
        with open(samples, 'r') as f:
            self.rir_files = [line.strip() for line in f if line.strip()]

        if not self.rir_files:
            raise ValueError("RIR load error")

    def __call__(self, x):
        path = random.choice(self.rir_files)

        rir, _ = sf.read(path)
        rir = rir.astype(np.float32)
        rir = rir / (np.sqrt(np.sum(rir**2)) + 1e-8)

        x = signal.convolve(x, rir, mode='full')[:len(x)]
        return x