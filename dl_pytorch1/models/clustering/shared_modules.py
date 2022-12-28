import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderProjClass(nn.Module):
    def __init__(self, encoder, proj_head, class_head,
                 freeze_encoder=False, normalize_proj_head=True):
        super(EncoderProjClass, self).__init__()
        self.encoder = encoder
        self.proj_head = proj_head
        self.class_head = class_head

        self._freeze_encoder = freeze_encoder
        self._normalize_proj_head = normalize_proj_head

    def forward(self, x, return_hidden=False):
        if self._freeze_encoder:
            enc_train = self.encoder.training
            self.encoder.eval()
            with torch.no_grad():
                h = self.encoder(x)
                h = h.detach()
            self.encoder.train(enc_train)
        else:
            h = self.encoder(x)

        z = self.proj_head(h)

        if self._normalize_proj_head:
            z = F.normalize(z, p=2, dim=-1)

        ly = self.class_head(h)

        if return_hidden:
            return h, z, ly
        else:
            return z, ly