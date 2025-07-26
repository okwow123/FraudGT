import torch.nn as nn
from fraudGT.layer.gt_layer import GTLayer

class GTModel(nn.Module):
    def __init__(self, dim_in, dim_out, dataset=None):
        super().__init__()
        self.encoder = GTLayerMEM(dim_in, dim_out, dataset)
    
    def forward(self, batch):
        return self.encoder(batch)
