import numpy as np
import random

class ColorNoiseInjection:
    def __init__(self, snr_range=(5, 20)):
        self.snr_range = snr_range
        self.noise_types = ['white', 'pink', 'brown']

    def __call__(self, waveform: np.ndarray):
        assert len(waveform.shape) == 1
        length = len(waveform)
        snr_db = random.uniform(*self.snr_range)
        noise_type = random.choice(self.noise_types)
        noise = self._generate_colored_noise(length, noise_type)
        signal_power = np.mean(waveform ** 2)
        noise_power = np.mean(noise ** 2)
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        scale = np.sqrt(target_noise_power / (noise_power + 1e-10))
        noise *= scale
        return waveform + noise

    def _generate_colored_noise(self, length, noise_type):
        freqs = np.fft.rfftfreq(length)
        freqs[0] = 1e-6
        if noise_type == 'white':
            spectrum = np.ones_like(freqs)
        elif noise_type == 'pink':
            spectrum = 1 / np.sqrt(freqs)
        elif noise_type == 'brown':
            spectrum = 1 / freqs
        phases = np.exp(2j * np.pi * np.random.rand(len(freqs)))
        noise_fft = spectrum * phases
        noise = np.fft.irfft(noise_fft, n=length)
        noise = noise / (np.max(np.abs(noise)) + 1e-10)
        return noise