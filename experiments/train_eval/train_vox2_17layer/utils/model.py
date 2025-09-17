import torch

class FrozenWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)