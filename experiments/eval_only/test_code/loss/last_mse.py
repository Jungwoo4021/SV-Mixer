import torch.nn as nn

class LastMSE(nn.Module):
    '''KD loss simply comparing the last outputs of teachers and students using MSE. 
    '''
    def __init__(self, scale=1):
        super(LastMSE, self).__init__()
        self.scale = scale
        self.mse = nn.MSELoss()
        
    def forward(self, x, label):
        return self.scale * self.mse(x[:, -1, :, :], label[:, -1, :, :])