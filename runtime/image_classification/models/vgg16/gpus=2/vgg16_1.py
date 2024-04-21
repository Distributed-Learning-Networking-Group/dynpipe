# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from .stage2 import Stage2
from .stage3 import Stage3

class VGG16Partitioned_1(torch.nn.Module):
    def __init__(self):
        super(VGG16Partitioned_1, self).__init__()
        self.stage0 = Stage2()
        self.stage1 = Stage3()

    def forward(self, input0):
        out0 = self.stage0(input0)
        out1 = self.stage1(out0)
        return out1