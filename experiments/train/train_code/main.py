import os
import random
import shutil
import numpy as np
import torch
import pytorch_lightning as pl
from thop import profile
from transformers import AutoConfig, WavLMModel
from data_module import VoxCelebDataModule
from model_module import ModelModule
from arguments import get_args
from models import SVMixer, ECAPA_TDNN
from loss import AAMSoftmax, DistilHubertKDLoss
from utils import *


# ---------------------
#   [0] Environment setting
# ---------------------
# seed setting
args = get_args()
args['name'] = f'{args["name"]}_seed{args["seed"]}'
seed = args['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# script backup
backup = os.path.join(args['result'], args['project'], args['name'])
if os.path.exists(backup):
    shutil.rmtree(backup)
shutil.copytree(os.path.dirname(os.path.realpath(__file__)), f"{backup}/scripts")
for root, dirs, _ in os.walk(backup):
    for dir in dirs:
        if dir == '__pycache__':
            shutil.rmtree(f'{root}/{dir}')


# ---------------------
#   [1] Logging
# ---------------------
# Configure Neptune logger
neptune_logger = pl.loggers.NeptuneLogger(
    project=args['project'],
    api_key=args['neptune_token'],
    tags=args['tags'],
    name=args['name'],
    log_model_checkpoints=False,
    description=args['description'],
)

csv_logger = pl.loggers.CSVLogger(
    save_dir=backup
)


# ---------------------
#   [2] Data
# ---------------------
# Create the Trainer with DDP strategy and configured callbacks and logger
data_module = VoxCelebDataModule(args)

# ---------------------
#   [3] Model
# ---------------------
# Load teacher model configuration and weights
config = AutoConfig.from_pretrained(
    args['huggingface_url'],
    finetuning_task="audio-classification",
    revision="main",
)

teacher_model =  WavLMModel.from_pretrained(
    args['huggingface_url'],
    config=config,
    revision="main",
    ignore_mismatched_sizes=False,
)
teacher_model = teacher_model.to("cuda")
_ = teacher_model(torch.randn(1, 8000).to("cuda"), output_hidden_states=True)
teacher_model = FrozenWrapper(teacher_model)

# Build the student model
student_model = SVMixer(
    args['num_hidden_layers'],
    args['seq_len'],
    args['hidden_size'],
)
macs, _ = profile(student_model, inputs=(torch.randn(1, 48000),))
num_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
neptune_logger.experiment["student_MACs"].log(format(int(macs), ','))
neptune_logger.experiment["student_size"].log(format(num_params, ','))

# Build the student model
classifier = ECAPA_TDNN(
    args['num_hidden_layers'],
    args['hidden_size'],
    args['channel'],
    args['embedding_size']
)
macs, _ = profile(classifier, inputs=(torch.randn(1, args['num_hidden_layers'] + 1, args['seq_len'], args['hidden_size']),))
num_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
neptune_logger.experiment["classifier_MACs"].log(format(int(macs), ','))
neptune_logger.experiment["classifier_size"].log(format(num_params, ','))


# ---------------------
#   [4] Loss
# ---------------------
# Build the primary loss (AAMSoftmax) for speaker verification
class_weight = torch.FloatTensor(data_module.vox2.class_weight)
criterion_sv = AAMSoftmax(
    args['embedding_size'],
    len(data_module.vox2.class_weight),
    args['aam_margin'],
    args['aam_scale'],
    class_weight=class_weight,
    topk_panelty=args['topk_panalty']
)

# Build the Knowledge Distillation (KD) loss (MSE)
criterion_kd = DistilHubertKDLoss(
    args['cos_lambda'], 
    args['target_layer_idx'],
    args['teacher_hidden_size'],
    args['hidden_size'] 
)

# ---------------------
#   [5] Callback functions
# ---------------------
# Configure ModelCheckpoint callback: save the best model based on EER
checkpoint = pl.callbacks.ModelCheckpoint(
    dirpath=backup,
    filename="best-model-{epoch:02d}-{EER:.2f}",
    monitor="EER",
    mode="min",
    save_top_k=5,
    verbose=True
)

class OptimizerAndSchedulerSwitchCallback(pl.Callback):
    def __init__(self, switch_step):
        self.switch_step = switch_step

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        if trainer.global_step == self.switch_step:
            trainer.optimizers = [pl_module.sgd_optimizer]

            sch = pl_module.scheduler["scheduler"]
            sch.optimizer = pl_module.sgd_optimizer
            sch.base_lrs = [group["lr"] for group in pl_module.sgd_optimizer.param_groups]

switch_cb = OptimizerAndSchedulerSwitchCallback(switch_step=args['warmup_steps'] + args['rise_steps'])

# ---------------------
#   [6] Run
# ---------------------
# Create the Lightning model instance
model = ModelModule(
    args=args,
    teacher_model=teacher_model,
    student_model=student_model,
    classifier=classifier,
    criterion_sv=criterion_sv,
    criterion_kd=criterion_kd,
    trials=data_module.vox2.trials_O
)

trainer = pl.Trainer(
    max_steps=args['total_train_steps'],
    accelerator="gpu",
    devices=-1,
    logger=[neptune_logger, csv_logger],
    callbacks=[checkpoint, switch_cb],
    strategy="auto",
    gradient_clip_val=10.0,
    num_sanity_val_steps=0,
    check_val_every_n_epoch=None,
    val_check_interval=args['eval_interval_steps'],
)
trainer.fit(model, datamodule=data_module)