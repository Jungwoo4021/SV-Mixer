import torch
import torch.nn as nn

class DistilHubertKDLoss(nn.Module):
    """
    SSL KD loss function used in the paper:
    "Distilhubert: Speech representation learning by layer-wise distillation of hidden-unit bert".

    This loss function computes a loss by combining an L1 loss and a cosine similarity loss
    between the student network's predictions and the teacher network's hidden representations.
    """

    def __init__(self, cos_lambda, target_layer_idx, student_hidden_size, teacher_hidden_size):
        """
        Args:
            cos_lambda (float): Weighting factor for the cosine similarity term.
            target_layer_idx (list of int): List of teacher layer indices to distill.
            student_hidden_size (int): Hidden size of the student SSL model.
            teacher_hidden_size (int): Hidden size of the teacher SSL model.
        """
        super(DistilHubertKDLoss, self).__init__()
        self.cos_lambda = cos_lambda
        self.target_layer_idx = target_layer_idx
        self.num_heads = len(target_layer_idx)

        # Modules used in the loss computation
        self.log_sigmoid = nn.LogSigmoid()
        self.cos_sim = nn.CosineSimilarity(dim=1)
        
        # Create a prediction head for each teacher layer
        self.predict_layer = nn.ModuleList([
            self._make_prediction_layer(student_hidden_size, teacher_hidden_size)
            for _ in range(self.num_heads)
        ])

    def forward(self, x, teacher_output):
        """
        Computes the KD loss between the student's and teacher's hidden representations.

        Args:
            x (torch.Tensor): Student network's output, shape: [batch, time, seq, hidden]
                              (we use the last time step, x[:, -1, :, :])
            teacher_output (torch.Tensor): Teacher network's output, shape: [batch, layers, seq, hidden]

        Returns:
            torch.Tensor: Scalar value representing the final loss.
        """
        # Apply each prediction head on the last time step of the student output
        pred_heads = [self.predict_layer[i](x[:, -1, :, :]) for i in range(self.num_heads)]
        pred_heads = torch.stack(pred_heads, dim=1)  # shape: [batch, num_heads, seq, hidden]

        # Calculate L1 loss
        batch, num_heads, seq, hidden = pred_heads.size()
        if len(self.target_layer_idx) < teacher_output.size(1):
            teacher_hidden = teacher_output[:, self.target_layer_idx, :, :]  # Select teacher layers
        else:
            teacher_hidden = teacher_output
        # Reshape sequence and hidden dimensions to compute L1 loss
        l1_loss = torch.abs(
            pred_heads.reshape(batch, num_heads, seq * hidden) - 
            teacher_hidden.reshape(batch, num_heads, seq * hidden)
        )
        l1_loss = torch.mean(l1_loss, dim=-1)  # shape: [batch, num_heads]

        # Calculate cosine similarity loss
        pred_heads_flat = pred_heads.reshape(batch * num_heads * seq, hidden)
        teacher_flat = teacher_hidden.reshape(batch * num_heads * seq, hidden)
        cos_loss = self.cos_sim(pred_heads_flat, teacher_flat)
        # Apply log-sigmoid and reshape back to original dimensions
        cos_loss = self.log_sigmoid(cos_loss).view(batch, num_heads, seq)
        cos_loss = cos_loss.sum(dim=2)  # Sum over the sequence dimension, shape: [batch, num_heads]

        # Combine L1 loss and cosine similarity loss into the final loss
        loss = l1_loss - self.cos_lambda * cos_loss
        loss = loss.sum(dim=1)  # Sum over heads
        loss = loss.mean()     # Average over the batch

        return loss

    def _make_prediction_layer(self, hidden_size, target_size):
        layer = nn.Sequential(
            nn.Linear(hidden_size, target_size),
            nn.GELU(),
            nn.Linear(target_size, target_size),
        )
        return layer
