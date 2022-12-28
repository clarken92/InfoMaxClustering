import torch
import torch.nn.functional as F

from my_utils.pytorch1_utils.models import LiteBaseModel


from .shared_modules import EncoderProjClass
from ...utils.contrastive.online_criteria import \
    ConInstContrast_v2, ConCompContrast_v2, CatInstContrast_v2, \
    CatInstConsistency, CatInstContrast


class CRLC(LiteBaseModel):
    def __init__(self, num_clusters, feat_dim,
                 num_class_subheads, batch_size,
                 encoder, proj_head, class_head,

                 critic_type="log_dot_prod",
                 cons_type="neg_log_dot_prod",

                 normalize_proj_head=True,
                 freeze_encoder=False,
                 use_pretrained_net=False,

                 temperature=0.1,
                 cluster_temperature=1.0,

                 smooth_prob=True, smooth_coeff=0.1,
                 min_rate=0.7, max_rate=1.3,
                 max_abs_logit=20.0,

                 device='cuda'):

        LiteBaseModel.__init__(self, device)
        self.num_clusters = num_clusters
        self.feat_dim = feat_dim
        self.num_class_subheads = num_class_subheads

        self.add_module('encoder', encoder)
        self.add_module('proj_head', proj_head)
        self.add_module('class_head', class_head)
        self.encoder_proj_class = EncoderProjClass(
            encoder, proj_head, class_head,
            freeze_encoder=freeze_encoder,
            normalize_proj_head=normalize_proj_head)

        self.normalize_proj_head = normalize_proj_head

        if freeze_encoder:
            assert use_pretrained_net, "'use_pretrained_net' must be True when 'freeze_encoder'=True!"
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.freeze_encoder = freeze_encoder
        self.use_pretrained_net = use_pretrained_net

        self.smooth_prob = smooth_prob
        assert 0.0 <= smooth_coeff <= 1.0, f"smooth_coeff={smooth_coeff}!"
        self.smooth_coeff = smooth_coeff

        assert 0 < temperature, f"temperature={temperature}!"
        self.temperature = temperature
        assert 0 < cluster_temperature, f"cluster_temperature={cluster_temperature}!"
        self.cluster_temperature = cluster_temperature

        assert 0.0 <= min_rate <= 1.0, f"min_rate={min_rate}!"
        assert 1.0 <= max_rate <= 2.0, f"max_rate={max_rate}!"
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.max_abs_logit = max_abs_logit

        self._encoder_is_loaded = False
        self._proj_head_is_loaded = False

        self.batch_size = batch_size
        self._contrast_z_criterion = ConInstContrast_v2(
            batch_size, temperature, device, reduction='none')
        self._contrast_cluster_criterion = ConCompContrast_v2(
            num_clusters, cluster_temperature, device, reduction='none')
        # self._contrast_y_criterion = CatInstContrast_v2(
        #     batch_size, device, reduction='none')
        self._contrast_y_criterion = CatInstContrast(
            batch_size, device, reduction='none', critic_type=critic_type)
        self._cons_y_criterion = CatInstConsistency(reduction='none', cons_type=cons_type)

        print(f"In class [{self.__class__}]")
        print(f"num_clusters: {self.num_clusters}")
        print(f"feat_dim: {self.feat_dim}")
        print(f"num_class_subheads: {self.num_class_subheads}")
        print(f"batch_size: {self.batch_size}")

        print(f"critic_type: {critic_type}")
        print(f"cons_type: {cons_type}")

        print(f"freeze_encoder: {self.freeze_encoder}")
        print(f"use_pretrained_net: {self.use_pretrained_net}")

        print(f"temperature: {self.temperature}")
        print(f"cluster_temperature: {self.cluster_temperature}")

        print(f"smooth_prob: {self.smooth_prob}")
        print(f"smooth_coeff: {self.smooth_coeff}")
        print(f"min_rate: {self.min_rate}")
        print(f"max_rate: {self.max_rate}")
        print(f"max_abs_logit: {self.max_abs_logit}")

        print(f"device: {self.device}")

    def cluster(self, x, stack_results=True):
        with torch.no_grad():
            z, lys = self.encoder_proj_class(x)
            if not self.normalize_proj_head:
                z = F.normalize(z, p=2, dim=-1)

            ys_logit = []
            ys_prob = []
            ys_pred = []

            for i in range(self.num_class_subheads):
                py = torch.softmax(lys[i], dim=-1)
                _, y_pred = torch.max(py, dim=-1)

                ys_logit.append(lys[i])
                ys_prob.append(py)
                ys_pred.append(y_pred)

            if stack_results:
                ys_logit = torch.stack(ys_logit, dim=1)
                ys_prob = torch.stack(ys_prob, dim=1)
                ys_pred = torch.stack(ys_pred, dim=1)

        return {
            "z": z.data,
            "ys_logit": ys_logit.data,
            "ys_prob": ys_prob.data,
            "ys_pred": ys_pred.data,
        }

    def get_all_train_params(self):
        params = list(self.proj_head.parameters()) + list(self.class_head.parameters())

        if not self.freeze_encoder:
            params += list(self.encoder.parameters())

        return params

    def _loss_of_1_subhead(self, ly_1, ly_2, loss_coeffs):
        lc = loss_coeffs

        C = self.num_clusters
        min_rate = self.min_rate
        max_rate = self.max_rate
        zero = torch.zeros([], dtype=torch.float32,
                           device=self.device, requires_grad=False)

        py_1 = torch.softmax(ly_1, dim=-1)
        py_2 = torch.softmax(ly_2, dim=-1)

        if self.smooth_prob:
            # Smoothing
            alpha = self.smooth_coeff
            py_1_smt = (1.0 - alpha) * py_1 + alpha * (1.0 / C)
            py_2_smt = (1.0 - alpha) * py_2 + alpha * (1.0 / C)
        else:
            py_1_smt = torch.clamp(py_1, 1e-6, 1e6)
            py_2_smt = torch.clamp(py_2, 1e-6, 1e6)

        # py_avg = py_smt.mean(0)
        # py_avg_b = py_b_smt.mean(0)
        py_avg_1 = py_1.mean(0)
        py_avg_2 = py_2.mean(0)

        # Contrast cluster
        # ------------------------------- #
        if lc['contrast_cluster'] > 0:
            contrast_cluster = self._contrast_cluster_criterion(
                py_1_smt, py_2_smt, normalized=False).mean(0)
        else:
            contrast_cluster = zero
        # ------------------------------- #

        # Contrast y
        # ------------------------------- #
        if lc['contrast_y'] > 0:
            contrast_y = self._contrast_y_criterion(
                py_1_smt, py_2_smt).mean(0)
        else:
            contrast_y = zero
        # ------------------------------- #

        # Contrast y
        # ------------------------------- #
        if lc['cons_y'] > 0:
            cons_y = self._cons_y_criterion(py_1_smt, py_2_smt).mean(0)
        else:
            cons_y = zero
        # ------------------------------- #

        # Entropy
        # ------------------------------- #
        if lc['entropy'] > 0:
            entropy = -0.5 * ((py_avg_1 * py_avg_1.log()).sum(0) +
                              (py_avg_2 * py_avg_2.log()).sum(0))
        else:
            entropy = zero
        # ------------------------------- #

        # Loss that prevents cluster collapse
        # ------------------------------- #
        if lc['lower_clamp'] > 0:
            lower_clamp_coeff_1 = torch.min(py_avg_1.data - min_rate / C, zero).detach()
            lower_clamp_coeff_2 = torch.min(py_avg_2.data - min_rate / C, zero).detach()
            lower_clamp = 0.5 * ((py_avg_1 * lower_clamp_coeff_1).sum(0) +
                                 (py_avg_2 * lower_clamp_coeff_2).sum(0))
            tracked_lower_clamp = 0.5 * (lower_clamp_coeff_1.pow(2).sum(0) +
                                         lower_clamp_coeff_2.pow(2).sum(0))
        else:
            lower_clamp = zero
            tracked_lower_clamp = zero

        if lc['upper_clamp'] > 0:
            upper_clamp_coeff_1 = torch.max(py_avg_1.data - max_rate / C, zero).detach()
            upper_clamp_coeff_2 = torch.max(py_avg_2.data - max_rate / C, zero).detach()
            upper_clamp = 0.5 * ((py_avg_1 * upper_clamp_coeff_1).sum(0) +
                                 (py_avg_2 * upper_clamp_coeff_2).sum(0))
            tracked_upper_clamp = 0.5 * (upper_clamp_coeff_1.pow(2).sum(0) +
                                         upper_clamp_coeff_2.pow(2).sum(0))
        else:
            upper_clamp = zero
            tracked_upper_clamp = zero
        # ------------------------------- #

        # Min/Max Logit clamping
        # ------------------------------- #
        max_abs_logit = self.max_abs_logit
        if max_abs_logit > 0:
            lower_logit_clamp_coeff = torch.min(ly_1.data + max_abs_logit, zero).detach()
            lower_logit_clamp_coeff_b = torch.min(ly_2.data + max_abs_logit, zero).detach()
            lower_logit_clamp = 0.5 * ((ly_1 * lower_logit_clamp_coeff).sum(1).mean(0) +
                                       (ly_2 * lower_logit_clamp_coeff_b).sum(1).mean(0))
            tracked_lower_logit_clamp = 0.5 * (lower_logit_clamp_coeff.pow(2).sum(1).mean(0) +
                                               lower_logit_clamp_coeff_b.pow(2).sum(1).mean(0))

            upper_logit_clamp_coeff = torch.max(ly_1.data - max_abs_logit, zero).detach()
            upper_logit_clamp_coeff_b = torch.max(ly_2.data - max_abs_logit, zero).detach()
            upper_logit_clamp = 0.5 * ((ly_1 * upper_logit_clamp_coeff).sum(1).mean(0) +
                                       (ly_2 * upper_logit_clamp_coeff_b).sum(1).mean(0))

            tracked_upper_logit_clamp = 0.5 * (upper_logit_clamp_coeff.pow(2).sum(1).mean(0) +
                                               upper_logit_clamp_coeff_b.pow(2).sum(1).mean(0))

        else:
            lower_logit_clamp = zero
            tracked_lower_logit_clamp = zero

            upper_logit_clamp = zero
            tracked_upper_logit_clamp = zero
        # ------------------------------- #

        loss_subhead = lc['contrast_cluster'] * contrast_cluster + \
                       lc['contrast_y'] * contrast_y + \
                       lc['cons_y'] * cons_y + \
                       lc['entropy'] * (-entropy) + \
                       lc['lower_clamp'] * lower_clamp + \
                       lc['upper_clamp'] * upper_clamp + \
                       lc['lower_logit_clamp'] * lower_logit_clamp + \
                       lc['upper_logit_clamp'] * upper_logit_clamp

        return loss_subhead, \
               contrast_cluster.data.cpu().item(), \
               contrast_y.data.cpu().item(), \
               cons_y.data.cpu().item(), \
               entropy.data.cpu().item(), \
               tracked_lower_clamp.data.cpu().item(), \
               tracked_upper_clamp.data.cpu().item(), \
               tracked_lower_logit_clamp.data.cpu().item(), \
               tracked_upper_logit_clamp.data.cpu().item(), \
               py_1.data.cpu(), py_2.data.cpu()

    def _cluster_loss(self, z_1, z_2, lys_1, lys_2, loss_coeffs):
        lc = loss_coeffs

        nh = self.num_class_subheads
        zero = torch.zeros([], dtype=torch.float32,
                           device=self.device, requires_grad=False)

        # Contrast z
        # ------------------------------- #
        # NOTE: We need to use 'z' and 'z_b' to compute the loss
        if lc['contrast_z'] > 0:
            contrast_z = self._contrast_z_criterion(
                z_1, z_2, normalized=self.normalize_proj_head).mean(0)
        else:
            contrast_z = zero
        # ------------------------------- #

        loss_subheads = []
        contrast_cluster = zero
        contrast_y = zero
        cons_y = zero
        entropy = zero
        tracked_lower_clamp = zero
        tracked_upper_clamp = zero
        tracked_lower_logit_clamp = zero
        tracked_upper_logit_clamp = zero

        pys_1 = []
        pys_2 = []

        for i in range(nh):
            loss_subhead, contrast_cluster_, contrast_y_, cons_y_, entropy_, \
            tracked_lower_clamp_, tracked_upper_clamp_, \
            tracked_lower_logit_clamp_, tracked_upper_logit_clamp_, \
            py_1, py_2 = \
                self._loss_of_1_subhead(lys_1[i], lys_2[i], loss_coeffs)

            contrast_cluster = contrast_cluster + contrast_cluster_
            contrast_y = contrast_y + contrast_y_
            cons_y = cons_y + cons_y_
            entropy = entropy + entropy_
            tracked_lower_clamp = tracked_lower_clamp + tracked_lower_clamp_
            tracked_upper_clamp = tracked_upper_clamp + tracked_upper_clamp_
            tracked_lower_logit_clamp = tracked_lower_logit_clamp + tracked_lower_logit_clamp_
            tracked_upper_logit_clamp = tracked_upper_logit_clamp + tracked_upper_logit_clamp_

            loss_subheads.append(loss_subhead)
            pys_1.append(py_1)
            pys_2.append(py_2)

        contrast_cluster = contrast_cluster / nh
        contrast_y = contrast_y / nh
        cons_y = cons_y / nh
        entropy = entropy / nh
        tracked_lower_clamp = tracked_lower_clamp / nh
        tracked_upper_clamp = tracked_upper_clamp / nh
        tracked_lower_logit_clamp = tracked_lower_logit_clamp / nh
        tracked_upper_logit_clamp = tracked_upper_logit_clamp / nh

        loss = lc['contrast_z'] * contrast_z + (sum(loss_subheads) / nh)

        return {
            'loss': loss,
            'contrast_z': contrast_z.data.cpu().item(),
            'contrast_cluster': contrast_cluster,
            'contrast_y': contrast_y,
            'cons_y': cons_y,
            'entropy': entropy,
            'lower_clamp': tracked_lower_clamp,
            'upper_clamp': tracked_upper_clamp,
            'lower_logit_clamp': tracked_lower_logit_clamp,
            'upper_logit_clamp': tracked_upper_logit_clamp,
            'pys_1': pys_1,
            'pys_2': pys_2,
        }

    def train_step(self, x_1, x_2, loss_coeffs, optimizer):
        if self.use_pretrained_net:
            assert self._encoder_is_loaded, "Encoder must be loaded!"
            assert self._proj_head_is_loaded, "Proj head must be loaded!"

        z_1, lys_1 = self.encoder_proj_class(x_1)
        z_2, lys_2 = self.encoder_proj_class(x_2)

        cluster_results = self._cluster_loss(
            z_1=z_1, z_2=z_2, lys_1=lys_1, lys_2=lys_2,
            loss_coeffs=loss_coeffs)
        loss = cluster_results['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cluster_results['loss'] = loss.data.cpu().item()

        py = cluster_results['pys_1'][0]
        # (num_classes,)
        y_prob_avg = py.mean(0)
        y_prob_min, _ = torch.min(py, dim=0)
        y_prob_max, _ = torch.max(py, dim=0)

        y_logit_min, _ = torch.min(lys_1[0], dim=0)
        y_logit_max, _ = torch.max(lys_1[0], dim=0)

        outputs = cluster_results
        outputs.update({
            'y_prob_avg': y_prob_avg.data.cpu(),
            'y_prob_min': y_prob_min.data.cpu(),
            'y_prob_max': y_prob_max.data.cpu(),

            'y_logit_min': y_logit_min.data.cpu(),
            'y_logit_max': y_logit_max.data.cpu(),
        })

        return outputs

    def load_encoder_n_proj_head(
        self, file_path, encoder_key="encoder_state_dict",
        proj_head_key="proj_head_state_dict"):

        print(f"Load 'encoder' and 'proj_head' from : '{file_path}'!")
        save_obj = torch.load(file_path)
        self.encoder.load_state_dict(save_obj[encoder_key])
        self.proj_head.load_state_dict(save_obj[proj_head_key])
        self._encoder_is_loaded = True
        self._proj_head_is_loaded = True