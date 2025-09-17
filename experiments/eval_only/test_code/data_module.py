import numpy as np
import random
import soundfile as sf
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from dataset import *

class VoxCelebDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.crop_size = args['crop_size']
        self.num_seg = args['num_seg']
        self.batch_size = args['batch_size'] // torch.cuda.device_count()
        self.vox2 = VoxCeleb2(args['path_vox2_train'], args['path_vox_O_trials'], args['path_vox_E_trials'], args['path_vox_H_trials'])

    def setup(self, stage=None):
        self.enrollment_set = EnrollmentSet(self.vox2.enrollment_set_O, self.crop_size, self.num_seg)

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return DataLoader(
            self.enrollment_set,
            num_workers=4,
            batch_size=self.batch_size // self.num_seg,
        )

class VoxCelebExtendDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.crop_size = args['crop_size']
        self.num_seg = args['num_seg']
        self.batch_size = args['batch_size'] // torch.cuda.device_count()
        self.vox2 = VoxCeleb2(args['path_vox2_train'], args['path_vox_O_trials'], args['path_vox_E_trials'], args['path_vox_H_trials'])

    def setup(self, stage=None):
        self.enrollment_set = EnrollmentSet(self.vox2.enrollment_set_E, self.crop_size, self.num_seg)

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return DataLoader(
            self.enrollment_set,
            num_workers=4,
            batch_size=self.batch_size // self.num_seg,
        )

class VoxCelebHardDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.crop_size = args['crop_size']
        self.num_seg = args['num_seg']
        self.batch_size = args['batch_size'] // torch.cuda.device_count()
        self.vox2 = VoxCeleb2(args['path_vox2_train'], args['path_vox_O_trials'], args['path_vox_E_trials'], args['path_vox_H_trials'])

    def setup(self, stage=None):
        self.enrollment_set = EnrollmentSet(self.vox2.enrollment_set_H, self.crop_size, self.num_seg)

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return DataLoader(
            self.enrollment_set,
            num_workers=4,
            batch_size=self.batch_size // self.num_seg,
        )

class VCMixDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.crop_size = args['crop_size']
        self.num_seg = args['num_seg']
        self.batch_size = args['batch_size'] // torch.cuda.device_count()
        self.vcmix = VCMixTest(args['path_vcmix_trials'])

    def setup(self, stage=None):
        self.enrollment_set = EnrollmentSet(self.vcmix.enrollment_set, self.crop_size, self.num_seg)

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return DataLoader(
            self.enrollment_set,
            num_workers=4,
            batch_size=self.batch_size // self.num_seg,
        )

class VoxSRCDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.crop_size = args['crop_size']
        self.num_seg = args['num_seg']
        self.batch_size = args['batch_size'] // torch.cuda.device_count()
        self.voxsrc = VoxSRC23Test(args['path_voxsrc_trials'])

    def setup(self, stage=None):
        self.enrollment_set = EnrollmentSet(self.voxsrc.enrollment_set, self.crop_size, self.num_seg)

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return DataLoader(
            self.enrollment_set,
            num_workers=4,
            batch_size=self.batch_size // self.num_seg,
        )

class VoicesDevDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.crop_size = args['crop_size']
        self.num_seg = args['num_seg']
        self.batch_size = args['batch_size'] // torch.cuda.device_count()
        self.voices = VOiCES(args['path_voices_dev_trials'], args['path_voices_eval_trials'])

    def setup(self, stage=None):
        self.enrollment_set = EnrollmentSet(self.voices.dev_enrollment_set, self.crop_size, self.num_seg)

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return DataLoader(
            self.enrollment_set,
            num_workers=4,
            batch_size=self.batch_size // self.num_seg,
        )

class VoicesEvalDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.crop_size = args['crop_size']
        self.num_seg = args['num_seg']
        self.batch_size = args['batch_size'] // torch.cuda.device_count()
        self.voices = VOiCES(args['path_voices_dev_trials'], args['path_voices_eval_trials'])

    def setup(self, stage=None):
        self.enrollment_set = EnrollmentSet(self.voices.eval_enrollment_set, self.crop_size, self.num_seg)

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return DataLoader(
            self.enrollment_set,
            num_workers=4,
            batch_size=self.batch_size // self.num_seg,
        )

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
