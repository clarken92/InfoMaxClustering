import torch
import torch.nn as nn
import torch.nn.functional as F


class ConInstContrast:
    def __init__(self, num_samples, temperature, device, reduction="mean"):
        super(ConInstContrast, self).__init__()
        self.num_samples = num_samples
        self.temperature = temperature
        self.device = device
        self.reduction = reduction

        # This mask is different from the mask in version 2
        # It also mask 2 sub-diagonals
        self.mask = self.get_mask(num_samples)
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def get_mask(self, num_samples):
        B = num_samples

        mask = torch.full((2 * B, 2 * B), True, dtype=torch.bool,
                          device=self.device, requires_grad=False)
        mask = mask.fill_diagonal_(False)

        for i in range(B):
            mask[i, B + i] = False
            mask[B + i, i] = False

        return mask

    def critic(self, z, normalized):
        if normalized:
            return torch.matmul(z, z.t()) / self.temperature
        else:
            return F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1) / self.temperature

    def compute_loss(self, sim, *args, **kwargs):
        B = self.num_samples
        assert tuple(sim.size()) == (2*B, 2*B), \
            f"sim.size()={sim.size()}"

        # (B, ): (i, i+B)
        sim_pos_1 = torch.diag(sim, B)
        # (B, ): (i+B, i)
        sim_pos_2 = torch.diag(sim, -B)

        # (2B, 1)
        sim_pos = torch.cat((sim_pos_1, sim_pos_2), dim=0).reshape(2 * B, 1)
        # (2B, 2B - 2)
        sim_neg = sim[self.mask].reshape(2 * B, -1)

        # (2B,) of 0s (positive samples are at index 0)
        labels = torch.zeros([2 * B], dtype=torch.long, device=sim_pos.device)
        # (2B, 2B - 1)
        logits = torch.cat((sim_pos, sim_neg), dim=1)

        # Maximize index 0 and minimize other index
        loss = self.criterion(logits, labels)

        return loss

    def __call__(self, z_1, z_2, normalized=True, return_sim_matrix=False):
        z = torch.cat((z_1, z_2), dim=0)

        # (2B, 2B)
        sim = self.critic(z, normalized)

        loss = self.compute_loss(sim)

        if return_sim_matrix:
            return loss, sim
        else:
            return loss


class ConInstContrast_v2(ConInstContrast):
    def __init__(self, num_samples, temperature, device, reduction="mean"):
        super(ConInstContrast_v2, self).__init__(
            num_samples, temperature, device, reduction=reduction)

    # We still keep the upper (i, i+B) and lower (i+B, i) diagonals as 1
    # since we still need to compute these values in the denominator
    def get_mask(self, num_samples):
        B = num_samples
        # All entries are 1 with the main diagonal entries are 0
        mask = torch.full((2 * B, 2 * B), True, dtype=torch.bool,
                          device=self.device, requires_grad=False)
        mask = mask.fill_diagonal_(False)
        return mask

    def compute_loss(self, sim, *args, **kwargs):
        B = self.num_samples
        assert tuple(sim.size()) == (2 * B, 2 * B), \
            f"sim.size()={sim.size()}"

        # (2B, 1)
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)

        # (2B, 2B)
        sim_b = sim - sim_max.detach()

        # (B, ): (i, i+B)
        sim_pos_1 = torch.diag(sim_b, B)
        # (B, ): (i+B, i)
        sim_pos_2 = torch.diag(sim_b, -B)

        # (2B,)
        sim_pos = torch.cat((sim_pos_1, sim_pos_2), dim=0)

        # (2B, 2B) * (2B, 2B) => (2B, 2B) the sum over dim=1 => (2B,)
        logsumexp_sim = (sim_b.exp() * self.mask).sum(1).log()

        loss = (-sim_pos + logsumexp_sim)
        if self.reduction == "mean":
            loss = loss.mean(0)
        elif self.reduction == "sum":
            loss = loss.sum(0)

        return loss


class ConCompContrast(ConInstContrast):
    def __init__(self, num_components, temperature, device, reduction="mean"):
        super(ConCompContrast, self).__init__(
            num_components, temperature, device, reduction=reduction)

    def __call__(self, c_1, c_2, normalized=False, return_sim_matrix=False):
        return super(ConCompContrast, self).__call__(
            c_1.t(), c_2.t(), normalized=normalized,
            return_sim_matrix=return_sim_matrix)


class ConCompContrast_v2(ConInstContrast_v2):
    def __init__(self, num_components, temperature, device, reduction="mean"):
        super(ConCompContrast_v2, self).__init__(
            num_components, temperature, device, reduction=reduction)

    def __call__(self, c_1, c_2, normalized=False, return_sim_matrix=False):
        return super(ConCompContrast_v2, self).__call__(
            c_1.t(), c_2.t(), normalized=normalized,
            return_sim_matrix=return_sim_matrix)


class CatInstContrast(ConInstContrast):
    def __init__(self, num_samples, device, reduction="mean",
                 critic_type="log_dot_prod"):
        super(CatInstContrast, self).__init__(num_samples, None, device, reduction)
        self.critic_type = critic_type

    def critic(self, p, normalized):
        if self.critic_type == "log_dot_prod":
            return torch.matmul(p, p.t()).log()

        elif self.critic_type == "dot_prod":
            return torch.matmul(p, p.t())

        elif self.critic_type == "neg_l2" or self.critic_type == "nsse":
            return -(p.unsqueeze(1) - p.unsqueeze(0)).pow(2).sum(-1)

        elif self.critic_type == "neg_jsd":
            p1 = p.unsqueeze(1)
            p2 = p.unsqueeze(0)
            p_avg = 0.5 * (p1 + p2)

            out = -0.5 * ((p1 * (p1.log() - p_avg.log())).sum(-1) +
                          (p2 * (p2.log() - p_avg.log())).sum(-1))

            return out

        else:
            raise ValueError(f"Do not support critic_type={self.critic_type}!")

    def __call__(self, p_1, p_2, return_sim_matrix=False):
        return super(CatInstContrast, self).__call__(
            p_1, p_2, normalized=True,
            return_sim_matrix=return_sim_matrix)


class CatInstContrast_v2:
    def __init__(self, num_samples, device, reduction="mean"):
        self.num_samples = num_samples
        self.device = device
        assert reduction in ("none", "mean", "sum"), f"reduction={reduction}!"
        self.reduction = reduction

        self.mask = self.get_mask(num_samples, device)

    def get_mask(self, num_samples, device):
        # All entries are 1 with the main diagonal entries are 0
        mask = torch.ones((2 * num_samples, 2 * num_samples), dtype=torch.float32,
                          device=device, requires_grad=False)
        mask = mask.fill_diagonal_(0)
        return mask

    def exp_critic(self, p):
        return torch.matmul(p, p.t())

    def compute_loss(self, exp_sim):
        B = self.num_samples
        assert tuple(exp_sim.size()) == (2 * B, 2 * B), \
            f"sim.size()={exp_sim.size()}"

        # (B, ): (i, i+B)
        exp_sim_pos_1 = torch.diag(exp_sim, B)
        # (B, ): (i+B, i)
        exp_sim_pos_2 = torch.diag(exp_sim, -B)

        # (2B,)
        sim_pos = torch.cat((exp_sim_pos_1, exp_sim_pos_2), dim=0).log()
        # (2B, 2B) * (2B, 2B) => (2B, 2B) the sum over dim=1 => (2B,)
        logsumexp_sim = (exp_sim * self.mask).sum(1).log()

        loss = (-sim_pos + logsumexp_sim)
        if self.reduction == "mean":
            loss = loss.mean(0)
        elif self.reduction == "sum":
            loss = loss.sum(0)

        return loss

    def __call__(self, p_1, p_2, return_exp_sim_matrix=False):
        # (2B, C)
        p = torch.cat((p_1, p_2), dim=0)
        # (2B, 2B)
        exp_sim = self.exp_critic(p)

        loss = self.compute_loss(exp_sim)

        if return_exp_sim_matrix:
            return loss, exp_sim
        else:
            return loss


class CatInstConsistency:
    def __init__(self, reduction="mean", cons_type="neg_log_dot_prod"):
        assert reduction in ("none", "mean", "sum"), f"reduction={reduction}!"
        self.reduction = reduction

        possible_cons_types = ("xent", "jsd", "l2", "l1",
                               "neg_dot_prod", "neg_log_dot_prod")
        assert cons_type in possible_cons_types, \
            f"cons_type must be in {possible_cons_types}. Found {cons_type}!"
        self.cons_type = cons_type

    def get_cons(self, p_1, p_2):
        cons_type = self.cons_type

        if cons_type == "neg_log_dot_prod":
            cons = -(p_1 * p_2).sum(-1).log()
        elif cons_type == "neg_dot_prod":
            cons = -(p_1 * p_2).sum(-1)
        elif cons_type == "xent":
            cons = -(p_1.detach() * p_2.log()).sum(-1) \
                   -(p_2.detach() * p_1.log()).sum(-1)
            cons = cons * 0.5
        elif cons_type == "jsd":
            p_avg = 0.5 * (p_1 + p_2)

            cons = 0.5 * ((p_1 * (p_1.log() - p_avg.log())).sum(-1) +
                          (p_2 * (p_2.log() - p_avg.log())).sum(-1))
        elif cons_type == "l2":
            cons = (p_1 - p_2).pow(2).sum(-1)
        elif cons_type == "l1":
            cons = (p_1 - p_2).abs().sum(-1)
        else:
            raise ValueError(f"Do not support cons_type={cons_type}!")

        return cons

    def __call__(self, p_1, p_2):
        loss = self.get_cons(p_1, p_2)
        if self.reduction == "mean":
            loss = loss.mean(0)
        elif self.reduction == "sum":
            loss = loss.sum(0)

        return loss
