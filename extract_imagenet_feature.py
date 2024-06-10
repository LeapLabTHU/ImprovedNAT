import os
import torch.nn as nn
import numpy as np
import torch
from dataset import ImageNet
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)


def main(resolution=256, split='train'):
    dataset = ImageNet(path=args.path, resolution=resolution, random_flip=False, split=split).train
    dataset_loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False,
                                num_workers=8, pin_memory=True, persistent_workers=True)

    import taming.models.vqgan
    model = taming.models.vqgan.get_model('vq-f16-jax.yaml')

    model = nn.DataParallel(model)
    model.eval()
    model.requires_grad_(False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    feat_all = []
    with torch.no_grad():
        for batch in tqdm(dataset_loader):
            img, label = batch
            img = img.to(device)
            if split == 'train':
                img = torch.cat([img, img.flip(dims=[-1])], dim=0)
                label = torch.cat([label, label], dim=0)
            label = label.detach().cpu().numpy()
            N = len(label)
            batch = model(img)
            feat = batch[-1][-1].detach().cpu().numpy()
            feat_all.append(np.concatenate((label[:, None], feat.reshape(N, -1)), axis=-1))
    feat_all = np.concatenate(feat_all)

    out_dir = f'assets/imagenet{resolution}_vq_features'
    os.makedirs(out_dir, exist_ok=True)
    np.save(f'{out_dir}/{split}.npy', feat_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()
    main(resolution=256, split=args.split)
