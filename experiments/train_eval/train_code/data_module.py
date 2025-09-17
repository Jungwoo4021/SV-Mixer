import os
import h5py
import numpy as np
import random
import soundfile as sf
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from dataset import VoxCeleb2
from data_augmentation import *

class VoxCelebDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.crop_size = args['crop_size']
        self.num_seg = args['num_seg']
        self.batch_size = args['batch_size'] // torch.cuda.device_count()
        self.path_musan = args['path_musan']
        self.path_rir = args['path_rir']
        self.augment_probability = args['augment_probability']
        self.vox2 = VoxCeleb2(args['path_vox_train'], args['path_vox_O_trials'])

    def setup(self, stage=None):
        self.train_set = TrainSet(self.vox2.train_set, self.crop_size, self.path_musan, self.path_rir, self.augment_probability)
        self.enrollment_set = EnrollmentSet(self.vox2.enrollment_set_O, self.crop_size, self.num_seg)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            num_workers=4,
            batch_size=self.batch_size,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.enrollment_set,
            num_workers=4,
            batch_size=self.batch_size // self.num_seg,
        )
    
class TrainSet(Dataset):
    def __init__(self, items, crop_size, path_musan, path_rir, augment_probability, sample_rate=16000):
        self.items = items
        self.musan = MusanNoise(path_musan)
        self.rir = RIRReverberation(path_rir)
        self.color_noise = ColorNoiseInjection()
        self.clipping = ClippingAugmentation()
        self.crop_size = crop_size
        self.augment_probability = augment_probability
        self.sample_rate = sample_rate
        self.frame_unit = int(0.02 * sample_rate)  # 0.02s = 320 samples

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        wav_path = item.path
        
        info = sf.info(wav_path)
        total_samples = int(info.frames)

        # Crop audio
        max_start = total_samples - self.crop_size
        num_steps = max_start // self.frame_unit
        start = random.randint(0, num_steps) * self.frame_unit
        stop = start + self.crop_size
        audio, _ = sf.read(wav_path, start=start, stop=stop)

        feature = audio.copy()

        # Apply augmentation
        audio = self.augment(audio, self.augment_probability)

        return audio.astype(np.float32), feature.astype(np.float32), item.label

    def augment(self, audio, augment_probability):
        if random.random() > augment_probability:
            return audio
        aug_type = random.randint(0, 5)
        if aug_type == 0:
            audio = self.rir(audio)
        elif aug_type == 1:
            audio = self.musan(audio, 'speech')
        elif aug_type == 2:
            audio = self.musan(audio, 'music')
        elif aug_type == 3:
            audio = self.musan(audio, 'noise')
        elif aug_type == 4:
            audio = self.color_noise(audio)
        elif aug_type == 5:
            audio = self.clipping(audio)
        return audio

class EnrollmentSet(Dataset):
    def __init__(self, items, crop_size, num_seg):
        self.items = items
        self.crop_size = crop_size
        self.num_seg = num_seg
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):        
        # Sample an item
        item = self.items[index]

        # Read WAV file
        audio, _ = sf.read(item.path)
            
        # Segment the audio
        if audio.shape[0] <= self.crop_size:
            shortage = self.crop_size - audio.shape[0]
            audio = np.pad(audio, (0, shortage), mode='wrap')
            buffer = [audio for _ in range(self.num_seg)]
            buffer = np.stack(buffer, axis=0)
        else:
            buffer = []
            indices = np.linspace(0, audio.shape[0] - self.crop_size, self.num_seg)
            for idx in indices:
                idx = int(idx)
                buffer.append(audio[idx:idx + self.crop_size])
            buffer = np.stack(buffer, axis=0)

        return buffer.astype(float), item.key
