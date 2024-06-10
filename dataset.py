from pathlib import Path

from torch.utils.data import Dataset
import numpy as np
import torch
import math
import random
from PIL import Image
import os
from loguru import logger

print = logger.info

class CFGDataset(Dataset):  # for classifier free guidance
    def __init__(self, dataset, p_uncond, empty_token):
        self.dataset = dataset
        self.p_uncond = p_uncond
        self.empty_token = empty_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        if random.random() < self.p_uncond:
            y = self.empty_token
        return x, y


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError

class FeatureDataset(Dataset):
    def __init__(self, path, split, n_extras):
        super().__init__()
        self.path = path
        self.split = split
        self.n_extras = n_extras
        data_path = f'{path}/{split}.npy'
        self.data = np.load(data_path)

    def __len__(self):
        return len(self.data)  # consider the random flip

    def __getitem__(self, idx):
        return self.data[idx][self.n_extras:], self.data[idx][0:self.n_extras]


class ImageNet256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = FeatureDataset(path, split='train', n_extras=1)
        print('Prepare dataset ok')
        self.K = 1000
        self.empty_context = np.array([self.K], dtype=np.longlong)

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.empty_context)


class ImageNet(DatasetFactory):
    def __init__(self, path, resolution, random_crop=False, random_flip=True, split='train'):
        super().__init__()

        print(f'Counting ImageNet files from {path}')
        train_files = _list_image_files_recursively(os.path.join(path, split))
        # class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        class_names = [Path(path).parent.name for path in train_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting ImageNet files')

        self.train = ImageDataset(resolution, train_files, labels=train_labels, random_crop=random_crop,
                                  random_flip=random_flip)
        self.resolution = resolution
        if len(self.train) != 1_281_167:
            print(f'Missing train samples: {len(self.train)} < 1281167')

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif Path(full_path).is_dir():
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
            self,
            resolution,
            image_paths,
            labels,
            random_crop=False,
            random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.image_paths = image_paths
        self.labels = labels
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        while True:
            try:
                pil_image = Image.open(path)
                pil_image.load()
                pil_image = pil_image.convert("RGB")
            except OSError as e:
                print(f"Catched exception: {str(e)}. Re-trying...")
                import time
                time.sleep(1)
            else:
                break

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 255.

        label = np.array(self.labels[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), label


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


def get_dataset(name, **kwargs):
    if name == 'imagenet':
        return ImageNet(**kwargs)
    elif name == 'imagenet256_features':
        return ImageNet256Features(**kwargs)
    else:
        raise NotImplementedError(name)


