import torch
import torch.nn as nn
import os

from loguru import logger
from collections import defaultdict, deque

logging = logger


def get_nnet(**kwargs):
    from libs.nat_model import UViT
    return UViT(**kwargs)


def get_optimizer(params, name, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1

    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name='customized', args=None, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None, args=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema
        self.args = args

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key not in ['step', 'args'] and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load(self, path, ignored_keys=tuple()):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'), map_location='cpu')
        ignored_keys += ('step', 'args')
        for key, val in self.__dict__.items():
            if key not in ignored_keys and val is not None:
                ckpt = torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu')
                if key in ['nnet', 'nnet_ema']:
                    ckpt = {k: v for k, v in ckpt.items() if 'position_ids' not in k}
                val.load_state_dict(ckpt)
            else:
                logging.info(f'<<load state dict>>: ignore {key}')

    def resume(self, ckpt_root, step=None, ignored_keys=tuple()):
        if not os.path.exists(ckpt_root):
            logger.warning('ckpt_root does not exist when resuming.')
            return
        if ckpt_root.endswith('.ckpt'):
            ckpt_path = ckpt_root
        else:
            if step is None:
                ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
                if not ckpts:
                    return
                steps = map(lambda x: int(x.split(".")[0]), ckpts)
                step = max(steps)
            ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path, ignored_keys=ignored_keys)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def initialize_train_state(config, device, args):
    params = []

    nnet = get_nnet(**config.nnet, args=args)

    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet, args=args)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, args=args, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema, args=args)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter=" "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if torch.__version__.startswith('2'):
        from torch import inf
    else:
        from torch._six import inf
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def calc_fid(m1, s1, m2, s2):
    a = (m1 - m2).square().sum(dim=-1)
    b = s1.trace() + s2.trace()
    c = torch.linalg.eigvals(s1 @ s2).sqrt().real.sum(dim=-1)

    _fid = (a + b - 2 * c).item()
    return _fid
