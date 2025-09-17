import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import compute_eer, compute_min_dcf, CosineAnnealingWarmUpRestarts

class ModelModule(pl.LightningModule):
    def __init__(self, args, teacher_model, student_model, classifier, criterion_sv, criterion_kd, trials):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.classifier = classifier
        self.criterion_sv = criterion_sv
        self.criterion_kd = criterion_kd
        self.trials = trials

        self.rise_steps = args['rise_steps']
        self.warmup_steps = args['warmup_steps']
        self.train_steps = args['train_steps']

        self.lr_max = args['lr_max']
        self.lr_min = args['lr_min']
        self.warmup_lr_max = args['warmup_lr_max']
        self.warmup_lr_min = args['warmup_lr_min']

        self.weight_decay = args['weight_decay']
        self.momentum = args['momentum']
        self.num_seg = args['num_seg']
        
        self._val_outputs = []
        self.current_EER = None

    def forward(self, x):
        x = self.student_model(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, x_wavlm, labels = batch
        x = x.to(dtype=torch.float32, device=self.device)
        x_wavlm = x_wavlm.to(dtype=torch.float32, device=self.device)
        labels = labels.to(self.device)
        
        with torch.no_grad():
            teacher_out = self.teacher_model(x_wavlm, output_hidden_states=True).hidden_states
            teacher_out = torch.stack(teacher_out, dim=1)
            
        student_output = self.student_model(x)
        embeddings = self.classifier(student_output)
        loss_sv = self.criterion_sv(embeddings, labels)
        loss_kd = self.criterion_kd(student_output, teacher_out)
        loss = loss_sv + loss_kd

        self.log("Loss (SV)", loss_sv, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("Loss (KD)", loss_kd, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.current_EER = None
        return loss

    def on_after_backward(self):
        total_norm = torch.norm(torch.stack([
            p.grad.detach().data.norm(2)
            for p in self.parameters()
            if p.grad is not None
        ]), 2)
        self.log('grad_norm', total_norm, prog_bar=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr)

    def on_validation_epoch_start(self):
        self._val_outputs = []

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x_seg, keys = batch
        B = x_seg.size(0)
        x_seg = x_seg.to(dtype=torch.float32, device=self.device, non_blocking=True)
        x_seg = x_seg.view(B * self.num_seg, -1)
        x_seg = self.student_model(x_seg)
        emb = self.classifier(x_seg)
        emb = emb.view(B, self.num_seg, -1)
        self._val_outputs.append({"keys": keys, "embeddings": emb})

    def on_validation_epoch_end(self):
        keys_list = []
        embeddings_list = []
        for batch_out in self._val_outputs:
            keys_list.extend(batch_out["keys"])
            embeddings_list.append(batch_out["embeddings"])

        all_keys_tensor = torch.stack(keys_list, dim=0)
        all_embeddings_tensor = torch.cat(embeddings_list, dim=0)

        if self.trainer.world_size > 1:
            gathered_keys = self.all_gather(all_keys_tensor)
            gathered_embeddings = self.all_gather(all_embeddings_tensor)
        else:
            gathered_keys = all_keys_tensor
            gathered_embeddings = all_embeddings_tensor

        gathered_keys = gathered_keys.cpu().view(-1)
        gathered_embeddings = gathered_embeddings.cpu().view(-1, self.num_seg, gathered_embeddings.size(-1))

        max_key = int(gathered_keys.max().item())
        embedding_list = [None] * (max_key + 1)

        for k, emb in zip(gathered_keys, gathered_embeddings):
            embedding_list[int(k.item())] = emb

        labels = []
        cos_sims_seg = [[], []]
        for trial in self.trials:
            cos_sims_seg[0].append(embedding_list[trial.key1])
            cos_sims_seg[1].append(embedding_list[trial.key2])
            labels.append(trial.label)

        batch_size = len(labels)
        buffer_seg_1 = torch.stack(cos_sims_seg[0], dim=0).view(batch_size, self.num_seg, -1)
        buffer_seg_2 = torch.stack(cos_sims_seg[1], dim=0).view(batch_size, self.num_seg, -1)

        buffer_seg_1 = buffer_seg_1.repeat(1, self.num_seg, 1).view(batch_size * self.num_seg * self.num_seg, -1)
        buffer_seg_2 = buffer_seg_2.repeat(1, 1, self.num_seg).view(batch_size * self.num_seg * self.num_seg, -1)
        cosine_seg = F.cosine_similarity(buffer_seg_1, buffer_seg_2)
        scores = cosine_seg.view(batch_size, self.num_seg * self.num_seg).mean(dim=1)

        eer = compute_eer(scores, labels)
        min_dcf = compute_min_dcf(scores, labels)

        self.log("EER", eer, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("minDCF", min_dcf, prog_bar=True, on_epoch=True, sync_dist=True)
        self.current_EER = eer

    def configure_optimizers(self):
        params = list(self.student_model.parameters()) + list(self.classifier.parameters())
        if isinstance(self.criterion_sv, torch.nn.Module):
            params += list(self.criterion_sv.parameters())
        if isinstance(self.criterion_kd, torch.nn.Module):
            params += list(self.criterion_kd.parameters())

        self.adam_optimizer = torch.optim.AdamW(
            params,
            lr=self.warmup_lr_min,
            weight_decay=self.weight_decay
        )

        self.sgd_optimizer = torch.optim.SGD(
            params,
            lr=self.lr_min,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            nesterov=True
        )

        self.scheduler = {
            'scheduler': CosineAnnealingWarmUpRestarts(
                self.adam_optimizer,
                self.warmup_steps + self.rise_steps,
                T_mult=self.train_steps // self.warmup_steps,
                eta_max=self.warmup_lr_max,
                T_up=self.rise_steps,
                gamma=self.lr_max / self.warmup_lr_max,
                last_epoch=-1
            ),
            'interval': 'step',
            'frequency': 1
        }

        return [self.adam_optimizer], [self.scheduler]