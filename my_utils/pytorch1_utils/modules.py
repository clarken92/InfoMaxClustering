import torch.nn as nn


class CustomModuleList(nn.Module):
    def __init__(self, modules):
        super().__init__()

        assert isinstance(modules, (list, tuple)), f"type(modules)={type(modules)}"
        self.module_list = nn.ModuleList(modules)

    def forward(self, inp):
        outs = []
        for m in self.module_list:
            outs.append(m(inp))

        return outs