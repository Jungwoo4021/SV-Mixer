import os
import random
import numpy as np
import soundfile as sf
from collections import defaultdict

# noisetype sample
class MusanNoise:
    Category = ['noise','speech','music']
    SNR_RANGE = {
        'noise': (0, 15),
        'speech': (13, 20),
        'music': (5, 15)
    }
    FILE_COUNT_RANGE = {
        'noise': (1, 1),
        'speech': (4, 7),
        'music': (1, 1)
    }
    
    def __init__(self, samples):
        # train_set
        self.noise_samples = defaultdict(list)

        with open(samples, 'r') as f:
            for line in f:
                line = line.strip()
                noise_type, wav_path = line.split()

                assert noise_type in self.Category, f"Unknown noise type: {noise_type}"

                self.noise_samples[noise_type].append(wav_path)
                
    def __call__(self, x, category):
        assert category in self.Category, f"Unknown noise type: {noise_type}"
        
        # calculate dB
        x_size = x.shape[-1]
        x_dB = self.calculate_decibel(x)

        # select noise, snr
        snr = random.uniform(*self.SNR_RANGE[category])
        num_files = random.randint(*self.FILE_COUNT_RANGE[category])
        noise_paths = random.sample(self.noise_samples[category], num_files)

        # inject noise
        noises = [self.load_noise(path, x_size) for path in noise_paths]
        noise = np.mean(noises, axis=0)
        noise_dB = self.calculate_decibel(noise)
        scale = np.sqrt(10 ** ((x_dB - noise_dB - snr) / 10))
        x += scale * noise

        return x

    def load_noise(self, path, target_size):
        info = sf.info(path)
        total_size = int(info.samplerate * info.duration)

        if total_size <= target_size:
            noise, _ = sf.read(path, start=0)
            if len(noise) < target_size:
                noise = np.pad(noise, (0, target_size - len(noise)), 'wrap')
        else:
            start = random.randint(0, total_size - target_size)
            noise, _ = sf.read(path, start=start, stop=start + target_size)

        return noise

    def calculate_decibel(self, wav):
        return 10 * np.log10(np.mean(wav ** 2) + 1e-4)