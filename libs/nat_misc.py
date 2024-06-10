import numpy as np
import torch
import math
from einops import rearrange
from torch.nn import functional as F

def add_gumbel_noise(t, temperature, device):
    return (t + torch.Tensor(temperature * np.random.gumbel(size=t.shape)).to(device))


class NATSchedule(object):
    def __init__(self, codebook_size, device, ignore_ind=-1, smoothing=0., beta_alpha_beta=(12, 3)):
        self.mask_ind = codebook_size  # for input masking
        self.ignore_ind = ignore_ind  # for ce loss, excluding visible
        self.device = device
        self.smoothing = smoothing
        self.beta_a, self.beta_b = beta_alpha_beta

    @staticmethod
    def cosine_schedule(t):
        return torch.cos(t * math.pi * 0.5)

    def sample(self, x0):
        N, L, device = *x0.shape, self.device
        beta_dist = torch.distributions.Beta(self.beta_a, self.beta_b)
        rand_mask_probs = beta_dist.sample((N,)).to(device).float()
        num_token_masked = (L * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand(N, L, device=device).argsort(dim=-1)
        mask = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
        masked_ids = torch.where(mask, self.mask_ind, x0)
        labels = torch.where(mask, x0, self.ignore_ind)
        return None, labels, masked_ids  # timestep is not needed for nnet

    def loss(self, pred, label):  # pred: N, L, C
        return F.cross_entropy(pred.transpose(1, 2), label.long(),
                               ignore_index=self.ignore_ind, label_smoothing=self.smoothing)

    @torch.no_grad()
    def generate(self, gen_steps, _n_samples, nnet, decode_fn,
                 manual_ratios, manual_temp, manual_samp_temp, manual_cfg, **kwargs):
        device = self.device

        fmap_size = 16
        seq_len = fmap_size * fmap_size

        ids = torch.full((_n_samples, seq_len), self.mask_ind, dtype=torch.long, device=device)

        for step in range(gen_steps):
            mask_ratio = manual_ratios[step] if step < gen_steps - 1 else 0
            # scaling temp
            annealed_temp = manual_temp[step] if step < gen_steps - 1 else 0
            # scaling cfg
            cfg_scale = manual_cfg[step]
            samp_temp = manual_samp_temp[step]
            # sampling & scoring
            is_mask = (ids == self.mask_ind)
            logits = nnet(ids, **kwargs, scale=cfg_scale)
            sampled_ids = torch.distributions.Categorical(logits=logits/max(samp_temp, 1e-4)).sample()
            logits = torch.log_softmax(logits, dim=-1)
            sampled_logits = torch.squeeze(
                torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)
            sampled_ids = torch.where(is_mask, sampled_ids, ids)
            sampled_logits = torch.where(is_mask, sampled_logits, +np.inf).float()
            # masking
            mask_len = torch.Tensor([np.floor(seq_len * mask_ratio)]).to(device)
            mask_len = torch.maximum(torch.Tensor([1]).to(device),
                                     torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1,
                                                   mask_len))[0].squeeze()
            confidence = add_gumbel_noise(sampled_logits, annealed_temp, device)
            sorted_confidence, _ = torch.sort(confidence, axis=-1)
            cut_off = sorted_confidence[:, mask_len.long() - 1:mask_len.long()]
            masking = (confidence <= cut_off)
            ids = torch.where(masking, self.mask_ind, sampled_ids)

        _z = rearrange(sampled_ids, 'b (i j) -> b i j', i=fmap_size, j=fmap_size)
        out = decode_fn(_z)

        return out
