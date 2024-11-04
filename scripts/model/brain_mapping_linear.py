import torch
import torch.nn as nn


class LinearMapping(nn.Module):
    def __init__(self, input_d, output_d):
        super(LinearMapping, self).__init__()
        self.input_d = input_d
        self.output_d = output_d
        self.linear_mapping = nn.Linear(self.input_d, self.output_d)

    def forward(self, x):
        y = self.linear_mapping(x)
        return y
