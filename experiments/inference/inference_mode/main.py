import os
import random
import shutil
import numpy as np
import torch
import pytorch_lightning as pl
from thop import profile
from transformers import AutoConfig, WavLMModel
from models import SVMixer, ECAPA_TDNN
from data_module import *
from model_module import ModelModule
from arguments import get_args
from loss import AAMSoftmax
from utils import FrozenWrapper



# ---------------------
#   [0] Environment setting
# ---------------------
# seed setting
args = get_args()
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# script backup
backup = os.path.join(args['result'], args['project'], args['name'])

# ---------------------
#   [1] Logging
# ---------------------
csv_logger = pl.loggers.CSVLogger(
    save_dir=backup
)

# ---------------------
#   [2] Data
# ---------------------
# Create the Trainer with DDP strategy and configured callbacks and logger
voxceleb_module = VoxCelebDataModule(args)
voxcelebE_module = VoxCelebExtendDataModule(args)
voxcelebH_module = VoxCelebHardDataModule(args)
voices_dev_module = VoicesDevDataModule(args)
voices_eval_module = VoicesEvalDataModule(args)
voxsrc_module = VoxSRCDataModule(args)
vcmix_module = VCMixDataModule(args)

# ---------------------
#   [3] Model
# ---------------------
# Build the student model
student_model = SVMixer(
    args['num_hidden_layers'],
    args['seq_len'],
    args['hidden_size'],
)
checkpoint = torch.hub.load_state_dict_from_url(args['path_pretrined_svmixer'])
student_model.load_state_dict(checkpoint)
macs, _ = profile(student_model, inputs=(torch.randn(1, 48000),))
num_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)

# Build the student model
classifier = ECAPA_TDNN(
    args['num_hidden_layers'],
    args['hidden_size'],
    args['channel'],
    args['embedding_size']
)
checkpoint = torch.hub.load_state_dict_from_url(args['path_pretrined_classifier'])
classifier.load_state_dict(checkpoint)
macs, _ = profile(classifier, inputs=(torch.randn(1, args['num_hidden_layers'] + 1, args['seq_len'], args['hidden_size']),))
num_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

# ---------------------
#   [6] Run
# ---------------------
# Create the Lightning model instance
model = ModelModule(
    args=args,
    student_model=student_model,
    classifier=classifier,
)

trainer = pl.Trainer(
    max_epochs=1,
    accelerator="gpu",
    devices=-1,
    logger=[csv_logger],
    strategy='auto',
    num_sanity_val_steps=0 
)

# VoxCeleb O
model.set_trials('vox_O', voxceleb_module.vox2.trials_O)
trainer.validate(model, datamodule=voxceleb_module)

# VoxCeleb E
model.set_trials('vox_E', voxceleb_module.vox2.trials_E)
trainer.validate(model, datamodule=voxcelebE_module)

# VoxCeleb H
model.set_trials('vox_H', voxceleb_module.vox2.trials_H)
trainer.validate(model, datamodule=voxcelebH_module)

# voices
model.set_trials('VOiCES_Dev', voices_dev_module.voices.dev_trials)
trainer.validate(model, datamodule=voices_dev_module)
model.set_trials('VOiCES_Eval', voices_eval_module.voices.eval_trials)
trainer.validate(model, datamodule=voices_eval_module)

# voxsrc
model.set_trials('VoxSRC', voxsrc_module.voxsrc.trials)
trainer.validate(model, datamodule=voxsrc_module)

# vcmix
model.set_trials('VCmix', vcmix_module.vcmix.trials)
trainer.validate(model, datamodule=vcmix_module)