import datetime
import math
import os
import time
import argparse

import numpy as np
import pandas as pd
import torch
import yaml
from torch import optim
from torch.nn import Parameter
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader
from torch.utils._pytree import tree_map
from tqdm.auto import tqdm
import accelerate
import einops
from omegaconf import OmegaConf
from loguru import logger
import taming.models.vqgan
from libs.nat_misc import NATSchedule

import utils
from libs.inception import InceptionV3
from dataset import get_dataset

from torch._C import _distributed_c10d

_distributed_c10d.set_debug_level(_distributed_c10d.DebugLevel.INFO)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--benchmark', type=int, default=0)
    parser.add_argument('--mode', type=str, choices=['pretrain', 'search', 'eval'])
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--ignored_keys', type=str, default=[], nargs='+')
    parser.add_argument('--pretrained_path', type=str)
    parser.add_argument('--gen_steps', type=int, default=8)
    parser.add_argument('--eval_n', type=int, default=50000)
    # AutoNAT parameters
    parser.add_argument('--beta_alpha_beta', type=float, nargs='+', default=(12, 3))
    parser.add_argument('--test_bsz', type=int, default=125)
    parser.add_argument('--reference_image_path', type=str,
                        default='assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz')
    parser.add_argument('--searched_strategy', type=str)
    parser.add_argument('--sche_lr', type=float, default=1e-3)
    parser.add_argument('--sche_moment', type=float, default=0.5)
    parser.add_argument('--grad_estimate_step_size', type=float, default=1e-2)
    parser.add_argument('--unb_grad_estimate_step_size', type=float, default=0.1)
    parser.add_argument('--unb_lr', type=float, default=0.1)
    parser.add_argument('--max_upd', type=int, default=100)
    parser.add_argument('--resume_upd', type=str, nargs='+')
    parser.add_argument('--clip_grad', type=float)
    args = parser.parse_args()
    return args


def LSimple(x0, nnet, schedule, **kwargs):
    timesteps, labels, xn = schedule.sample(x0)
    pred = nnet(xn, timesteps=timesteps, **kwargs)
    loss = schedule.loss(pred, labels)
    masked_token_ratio = xn.eq(schedule.mask_ind).sum().item() / xn.shape[0] / xn.shape[1]
    return loss, masked_token_ratio


@logger.catch()
def train(config, args):
    logger.add(os.path.join(args.output_dir, 'output.log'), level='INFO')
    if args.benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(args.seed, device_specific=True)
    if accelerator.is_main_process:
        logger.info('Setting seed: {}'.format(args.seed))
    logger.info(f'Process {accelerator.process_index} using device: {device}')

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes
    args.batch_size = mini_batch_size
    logger.info(f'Using mini-batch size {mini_batch_size} per device')

    config.ckpt_root = os.path.join(args.output_dir, 'ckpts')
    config.searched_strategies_dir = os.path.join(args.output_dir, 'searched_strategies')
    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.searched_strategies_dir, exist_ok=True)

    accelerator.wait_for_everyone()
    logger.info(f'Run on {accelerator.num_processes} devices')

    # prepare for fid calc
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device)
    inception.eval()
    inception.requires_grad_(False)
    # load npz file
    with np.load(args.reference_image_path) as f:
        m2, s2 = f['mu'][:], f['sigma'][:]
        m2, s2 = torch.from_numpy(m2).to(device), torch.from_numpy(s2).to(device)

    autoencoder = taming.models.vqgan.get_model()
    codebook_size = autoencoder.n_embed
    config.nnet.codebook_size = codebook_size
    autoencoder.to(device)

    # load npy dataset
    dataset = get_dataset(**config.dataset)
    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=8, pin_memory=True, persistent_workers=True
                                      )
    # for cfg:
    empty_ctx = torch.from_numpy(np.array([[1000]], dtype=np.longlong)).to(device)

    train_state = utils.initialize_train_state(config, device, args)

    nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader)

    assert len(optimizer.param_groups) == 1
    lr_scheduler = train_state.lr_scheduler
    if args.resume and not bool(os.listdir(config.ckpt_root)):
        train_state.resume(args.resume, ignored_keys=args.ignored_keys)
    else:
        train_state.resume(config.ckpt_root)

    @torch.cuda.amp.autocast(enabled=True)
    def encode(_batch):
        return autoencoder.encode(_batch)[-1][-1].reshape(len(_batch), -1)

    @torch.cuda.amp.autocast(enabled=True)
    def decode(_batch):
        return autoencoder.decode_code(_batch)

    def get_data_generator():
        while True:
            for data in train_dataset_loader:
                yield data

    data_generator = get_data_generator()

    def get_test_generator():
        while True:
            yield torch.randint(0, 1000, (args.test_bsz, 1), device=device)

    schedule = NATSchedule(codebook_size=codebook_size, device=device, **config.muse,
                           beta_alpha_beta=args.beta_alpha_beta)

    def cfg_nnet(x, scale, **kwargs):
        _cond = nnet_ema(x, **kwargs)
        kwargs['context'] = einops.repeat(empty_ctx, '1 ... -> B ...', B=x.size(0))
        _uncond = nnet_ema(x, **kwargs)
        res = _cond + scale * (_cond - _uncond)
        return res

    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()
        with torch.no_grad():
            _z = _batch[0]
            context = _batch[1]
        loss, masked_token_ratio = LSimple(_z, nnet, schedule, context=context)
        metric_logger.update(loss=accelerator.gather(loss.detach()).mean())
        metric_logger.update(masked_token_ratio=masked_token_ratio)
        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step(train_state.step)
        train_state.ema_update(config.get('ema_rate', 0.9999))
        metric_logger.update(loss_scaler=accelerator.scaler.get_scale() if accelerator.scaler is not None else 1.)
        metric_logger.update(grad_norm=utils.get_grad_norm_(optimizer.param_groups[0]['params']))

        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'],
                    **{k: v.value for k, v in metric_logger.meters.items()})

    @torch.no_grad()
    def eval_step(n_samples, **nat_conf):
        # logger.info(f'evaluating with {nat_conf}')
        test_generator = get_test_generator()

        batch_size = args.test_bsz * accelerator.num_processes

        idx = 0

        pred_tensor = torch.empty((n_samples, 2048), device=device)
        for _batch_size in tqdm(utils.amortize(n_samples, batch_size), disable=not accelerator.is_main_process,
                                desc='sample2dir'):
            contexts = next(test_generator)
            samples = schedule.generate(args.gen_steps, len(contexts), cfg_nnet, decode, context=contexts,
                                        **nat_conf)
            samples = samples.clamp_(0., 1.)

            pred = inception(samples.float())[0]

            # Apply global spatial average pooling if needed
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2)
            pred_tensor[idx:idx + pred.shape[0]] = pred

            idx = idx + pred.shape[0]

        pred_tensor = pred_tensor[:idx].to(device)
        pred_tensor = accelerator.gather(pred_tensor)

        pred_tensor = pred_tensor[:n_samples]

        m1 = torch.mean(pred_tensor, dim=0)
        pred_centered = pred_tensor - pred_tensor.mean(dim=0)
        s1 = torch.mm(pred_centered.T, pred_centered) / (pred_tensor.size(0) - 1)

        m1 = m1.double()
        s1 = s1.double()

        fid = utils.calc_fid(m1, s1, m2, s2)

        if accelerator.is_main_process:
            logger.info(f'FID{n_samples}={fid}')
        return {f'fid{n_samples}': fid}

    class DisPatcher:
        def __init__(self, grad_estimate_step, unb_grad_estimate_step_size, lr, unb_lr, moment,
                     max_upd=100):
            init_values = {
                'manual_ratios': torch.tensor(np.cos(np.linspace(0, math.pi / 2, args.gen_steps + 1)))[1:-1],
                'manual_temp': torch.tensor(1 - np.linspace(0, 1, args.gen_steps + 1))[:-2],
                'manual_samp_temp': torch.tensor([1 for _ in range(args.gen_steps)]),
                'manual_cfg': torch.tensor(np.linspace(0, 1, args.gen_steps + 1))[1:],
            }
            clip_option = {  # True: clip to [0, 0.99], False: clip to [1e-4, +inf)
                'manual_ratios': [True] * len(init_values['manual_ratios']),
                'manual_temp': [False] * len(init_values['manual_temp']),
                'manual_samp_temp': [False] * len(init_values['manual_samp_temp']),
                'manual_cfg': [False] * len(init_values['manual_cfg']),
            }

            self.upd_idx = 0

            self.splits = {k: len(init_values[k]) for k in config_set}
            self.clip_option = torch.cat([torch.tensor(clip_option[k], dtype=torch.bool) for k in config_set])
            params = torch.cat([init_values[k] for k in config_set])

            self.params_clipped = Parameter(params[self.clip_option])
            self.params_unclipped = Parameter(params[~self.clip_option])

            # ensure cat is equal to params
            cumsum = torch.cumsum(self.clip_option.int(), dim=0)
            assert torch.all(cumsum[:-1] <= cumsum[-1]) and cumsum[-1] == torch.sum(self.clip_option).item()

            logger.info(f'self.params: {self.params_clipped}\n{self.params_unclipped}\n{self.params}')
            logger.info(f'self.splits: {self.splits}')

            self.grad_estimate_step = grad_estimate_step
            self.unb_grad_estimate_step_size = unb_grad_estimate_step_size
            self.max_upd = max_upd
            self.history_records = []
            self.record_time = datetime.datetime.now().strftime("%m%d_%H%M%S")

            param_groups = [
                {"params": self.params_clipped, "lr": lr},
                {"params": self.params_unclipped, "lr": unb_lr},
            ]

            self.optimizer = optim.SGD(param_groups, momentum=moment)

            self.best_fid = 1e9
            self.base_fid = self.eval(phase=f'stp_{self.upd_idx}', write=True)
            self.save()

        def run(self):
            for upd in range(self.max_upd):
                self.update()
                if self.base_fid < self.best_fid:
                    self.best_fid = self.base_fid
                    logger.info(f'Best FID: {self.best_fid} at step {self.upd_idx}')

        def save(self):
            if accelerator.is_main_process:
                save_path = os.path.join(config.searched_strategies_dir, f'stp_{self.upd_idx}.yaml')
                state_dict = self.params2sd(self.params)
                state_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in state_dict.items()}
                with open(save_path, 'w') as f:
                    yaml.dump(state_dict, f, sort_keys=False, default_flow_style=None)

        def update(self):
            self.optimizer.zero_grad()  # Zero out the gradients
            self.backward()
            logger.info(
                f'[Step {self.upd_idx}] Estimated Grad: {self.params_clipped.grad}\n{self.params_unclipped.grad}')
            logger.info(
                f'[Step {self.upd_idx}] Estimated Grad: {self.params_clipped.grad}\n{self.params_unclipped.grad}')
            if args.clip_grad is not None:
                self.params_clipped.grad = torch.clamp(self.params_clipped.grad, -args.clip_grad, args.clip_grad)
                self.params_unclipped.grad = torch.clamp(self.params_unclipped.grad, -args.clip_grad, args.clip_grad)
                logger.info(f'Grad after clipping: {self.params_clipped.grad}\n{self.params_unclipped.grad}')
            self.optimizer.step()  # Update parameters using the optimizer
            logger.info(f'updated params: {self.params_clipped}\n{self.params_unclipped}')
            self.params_clipped.data = torch.clamp(self.params_clipped, 0, 0.99).data
            self.params_unclipped.data = torch.clamp(self.params_unclipped, min=1e-4).data
            self.upd_idx += 1
            self.base_fid = self.eval(phase=f'stp_{self.upd_idx}', write=True)
            self.save()

        def params2sd(self, params):
            res = dict(zip(self.splits.keys(), torch.split(params, tuple(self.splits.values()))))
            res = {k: v[0] if len(v) == 1 else v for k, v in res.items()}
            res = {k: v.detach().numpy() for k, v in res.items()}
            return res

        @property
        def params(self):
            return torch.cat([self.params_clipped, self.params_unclipped])

        def eval(self, phase, params=None, write=False):
            params = self.params if params is None else params
            state_dict = self.params2sd(params)
            res = eval_step(n_samples=args.eval_n, **state_dict)
            state_dict = {k: tuple(v) if len(v.shape) != 0 else v.item() for k, v in state_dict.items()}
            result = {'phase': phase, **state_dict, **res, 'lr': (self.optimizer.param_groups[0]['lr'],
                                                                  self.optimizer.param_groups[1]['lr']),
                      'iter': self.upd_idx}
            self.history_records.append(result)
            if accelerator.is_main_process:
                pd.DataFrame(self.history_records).to_csv(
                    os.path.join(args.output_dir, f'dispatcher_{self.record_time}.csv'), index=False)
                if write:
                    logger.info(str(result))
            return res[f'fid{args.eval_n}']

        def backward(self):
            logger.info('Estimating grad')
            grad = torch.zeros_like(self.params)
            for i in range(len(self.params)):
                params = self.params.clone().detach()
                if self.clip_option[i]:
                    param_perturbed = params[i] + self.grad_estimate_step
                    param_perturbed = torch.clamp(param_perturbed, 0, 1)
                else:
                    param_perturbed = params[i] + self.unb_grad_estimate_step_size
                    param_perturbed = torch.clamp(param_perturbed, min=1e-4)
                params[i] = param_perturbed
                fid = self.eval(phase=f'grad_est_{i}_stp_{self.upd_idx}', params=params)
                grad[i] = (fid - self.base_fid) / (param_perturbed - self.params[i])
            self.params_clipped.grad = grad[self.clip_option]
            self.params_unclipped.grad = grad[~self.clip_option]

    if args.mode == 'pretrain':
        logger.info(f'Start fitting, step={train_state.step}, mixed_precision={accelerator.mixed_precision}')
        metric_logger = utils.MetricLogger()
        while train_state.step < config.train.n_steps:
            nnet.train()
            data_time_start = time.time()
            batch = next(data_generator)
            if isinstance(batch, list):
                batch = tree_map(lambda x: x.to(device), next(data_generator))
            else:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            metric_logger.update(data_time=time.time() - data_time_start)
            metrics = train_step(batch)

            if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
                torch.cuda.empty_cache()
                logger.info(f'Save checkpoint {train_state.step}...')
                if accelerator.local_process_index == 0:
                    train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))

            accelerator.wait_for_everyone()
            if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
                logger.info(f'[step {train_state.step}]: {metrics}')

        logger.info(f'Finish fitting, step={train_state.step}')
        del metrics
    else:
        config_set = ['manual_ratios', 'manual_temp', 'manual_samp_temp', 'manual_cfg']
        ckpt = torch.load(args.pretrained_path, map_location='cpu')
        nnet_ema.module.load_state_dict(ckpt)
        if args.mode == 'search':
            dispatcher = DisPatcher(grad_estimate_step=args.grad_estimate_step_size, lr=args.sche_lr,
                                    unb_lr=args.unb_lr,
                                    moment=args.sche_moment,
                                    unb_grad_estimate_step_size=args.unb_grad_estimate_step_size,
                                    max_upd=args.max_upd)
            dispatcher.run()
        elif args.mode == 'eval':
            prev_sd = OmegaConf.load(args.searched_strategy)
            res = eval_step(n_samples=args.eval_n,
                            **prev_sd)
            if accelerator.is_main_process:
                logger.info(f'Evaluated {args.eval_n} samples with strategy {args.searched_strategy}: {res}')


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    args = get_args()
    config = OmegaConf.load(args.config)
    train(config, args)
