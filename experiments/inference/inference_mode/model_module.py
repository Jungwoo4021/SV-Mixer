import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from thop import profile
from utils import compute_eer, compute_min_dcf

class ModelModule(pl.LightningModule):
    def __init__(self, args, student_model, classifier):
        super().__init__()
        self.student_model = student_model
        self.classifier = classifier

        self.num_seg = args['num_seg']

        self._val_outputs = []
    
    def set_trials(self, name, trials):
        self.test_name = name
        self.trials = trials

    def forward(self, x):
        x = self.student_model(x)
        x = self.classifier(x)
        return x

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
        # 1. Collect all keys and embeddings
        keys_list = []
        embeddings_list = []
        for batch_out in self._val_outputs:
            keys_list.extend(batch_out["keys"])
            embeddings_list.append(batch_out["embeddings"])

        all_keys_tensor = torch.stack(keys_list, dim=0)
        all_embeddings_tensor = torch.cat(embeddings_list, dim=0)

        # 2. If multi-GPU, gather results across all processes; otherwise keep as is
        if self.trainer.world_size > 1:
            gathered_keys = self.all_gather(all_keys_tensor)
            gathered_embeddings = self.all_gather(all_embeddings_tensor)
        else:
            gathered_keys = all_keys_tensor
            gathered_embeddings = all_embeddings_tensor

        gathered_keys = gathered_keys.cpu().view(-1)
        gathered_embeddings = gathered_embeddings.cpu().view(-1, self.num_seg, gathered_embeddings.size(-1))

        # 3. Store embeddings corresponding to each key (e.g., speaker ID)
        max_key = int(gathered_keys.max().item())
        embedding_list = [None] * (max_key + 1)
        for k, emb in zip(gathered_keys, gathered_embeddings):
            embedding_list[int(k.item())] = emb

        # 4. Process trials in chunks (e.g., 100,000 at a time)
        chunk_size = 100000
        all_scores = []
        all_labels = []
        trial_chunk = []
        for trial in self.trials:
            # Each trial assumed to have attributes: key1, key2, label
            trial_chunk.append((trial.key1, trial.key2, trial.label))
            if len(trial_chunk) >= chunk_size:
                scores_chunk, labels_chunk = self.process_trial_chunk(trial_chunk, embedding_list)
                all_scores.append(scores_chunk)
                all_labels.extend(labels_chunk)
                trial_chunk = []
        # Process the remainder
        if len(trial_chunk) > 0:
            scores_chunk, labels_chunk = self.process_trial_chunk(trial_chunk, embedding_list)
            all_scores.append(scores_chunk)
            all_labels.extend(labels_chunk)

        # 5. Concatenate results from all chunks
        scores = torch.cat(all_scores, dim=0)
        # scores: (total number of trials, )

        # 6. Compute evaluation metrics
        eer = compute_eer(scores, all_labels)
        min_dcf = compute_min_dcf(scores, all_labels)

        self.log(f"{self.test_name}_EER", eer, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{self.test_name}_minDCF", min_dcf, prog_bar=True, on_epoch=True, sync_dist=True)


    def process_trial_chunk(self, trial_chunk, embedding_list):
        """
        trial_chunk: list of tuples (key1, key2, label)
        embedding_list: full embedding list (indices correspond to keys)
        
        For each trial, retrieve embeddings corresponding to the keys,
        compute cosine similarity across self.num_seg segments,
        and average them to return a final score per trial.
        """
        batch_chunk = len(trial_chunk)
        # Extract key1, key2, label
        cos_sims_1 = [embedding_list[key1] for key1, key2, label in trial_chunk]
        cos_sims_2 = [embedding_list[key2] for key1, key2, label in trial_chunk]
        labels = [label for key1, key2, label in trial_chunk]
        
        # Stack into shape (batch_chunk, num_seg, D)
        buffer_seg_1 = torch.stack(cos_sims_1, dim=0).view(batch_chunk, self.num_seg, -1)
        buffer_seg_2 = torch.stack(cos_sims_2, dim=0).view(batch_chunk, self.num_seg, -1)

        # For each trial, compute self.num_seg x self.num_seg cosine similarities
        buffer_seg_1 = buffer_seg_1.repeat(1, self.num_seg, 1).view(batch_chunk * self.num_seg * self.num_seg, -1)
        buffer_seg_2 = buffer_seg_2.repeat(1, 1, self.num_seg).view(batch_chunk * self.num_seg * self.num_seg, -1)
        cosine_seg = F.cosine_similarity(buffer_seg_1, buffer_seg_2)
        # Average to obtain one score per trial
        scores_chunk = cosine_seg.view(batch_chunk, self.num_seg * self.num_seg).mean(dim=1)
        return scores_chunk, labels

    def configure_optimizers(self):
        params = list(self.student_model.parameters())
        if isinstance(self.criterion_sv, torch.nn.Module):
            params += list(self.criterion_sv.parameters())

        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode="min", factor=self.lr_gamma, patience=self.lr_patience, verbose=True),
            'monitor': 'EER',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]