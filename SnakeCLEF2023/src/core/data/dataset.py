import os
from typing import Iterable

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import PIL
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter

from src.utils import visualization as viz
from .sampling import ClassUniformSampler, oversample_dataset, augment_dataset

__all__ = ['ClassificationBaseDataset', 'ImageDataset',  'get_dataloader']


def truncate_str(x, maxlen=20):
    return f'{x[:maxlen].strip()}...' if len(x) > maxlen else x


class ClassificationBaseDataset(Dataset):
    def __init__(self, df, label_col, labels=None, encode_labels=True):
        assert label_col in df
        self.df = df
        self.label_col = label_col

        # set labels and ids
        self.labels = None
        if labels is not None:
            self.labels = labels
        elif (np.all([isinstance(x, str) for x in df[label_col]]) or
             pd.api.types.is_integer_dtype(df[label_col])):
            self.labels = np.unique(df[label_col])

        if encode_labels and self.labels is not None:
            self._label2id = {label: i for i, label in enumerate(self.labels)}
            self._id2label = {i: label for label, i in self._label2id.items()}
            self.ids = list(self._label2id.values())
        else:
            self._label2id = None
            self._id2label = None
            self.ids = self.labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raise NotImplementedError

    def label2id(self, x):
        if self._label2id is None:
            raise ValueError('The dataset does not encode labels.')

        if isinstance(x, Iterable) and not isinstance(x, str):
            out = np.array([self._label2id[itm] for itm in x])
        else:
            out = self._label2id[x]
        return out

    def id2label(self, x):
        if self._id2label is None:
            raise ValueError('The dataset does not encode labels.')

        if isinstance(x, Iterable) and not isinstance(x, str):
            out = np.array([self._id2label[itm] for itm in x])
        else:
            out = self._id2label[x]
        return out


class ImageDataset(ClassificationBaseDataset):
    def __init__(self, df, img_path_col, label_col, transforms=None,
                 path='.', bbox=False, labels=None, encode_labels=True):
        assert img_path_col in df
        super().__init__(df, label_col, labels, encode_labels)
        self.df = df
        self.img_path_col = img_path_col
        self.path = path
        self.transforms = transforms
        self.bbox=bbox

    def __getitem__(self, idx):
        img, label = self.get_item(idx)
        img = self.transform_image(img)
        if self._label2id:
            label = self._label2id[label]
        return img, label
        
    def transform_image(self, img: PIL.Image, ignore=[]):
        if self.transforms is not None:
            for t in self.transforms.transforms:
                if not isinstance(t, tuple(ignore)):
                    img = t(img)
        return img

    def get_item(self, idx):
       
        _file_path = self.df[self.img_path_col].values[idx]
        
        if _file_path[0] == '/':
            _file_path = _file_path[1:]
        file_path = os.path.join(self.path, _file_path)
        
        if self.bbox is not False:
            x_min = self.df['xmin'].values[idx]
            y_min=self.df['ymin'].values[idx]
            x_max=self.df['xmax'].values[idx]
            y_max=self.df['ymax'].values[idx]
            img = PIL.Image.open(file_path).convert('RGB')
            img=img.crop((x_min, y_min, x_max, y_max))
        else:
            img = PIL.Image.open(file_path).convert('RGB')

        label = self.df[self.label_col].values[idx]
        return img, label

    def show_item(self, idx=0, ax=None, apply_transforms=False):
        img, label = self.get_item(idx)
        if apply_transforms:
            img = self.transform_image(img, ignore=[T.Normalize])
            img = TF.to_pil_image(img)
        else:
            img = img.resize((224, 224), resample=PIL.Image.BILINEAR)
        img = np.array(img)
        ax = viz.imshow(
            img, title=truncate_str(f'{label}', maxlen=20),
            ax=ax, axis_off=True)
        return ax

    def show_items(self, idx=None, apply_transforms=False, *,
                   ncols=3, nrows=3, colsize=3, rowsize=3, **kwargs):
        if isinstance(idx, Iterable):
            params = dict(ntotal=len(idx))
        else:
            params = dict(nrows=nrows)
        fig, axs = viz.create_fig(ncols=ncols, colsize=colsize, rowsize=rowsize, **params)
        for i, ax in enumerate(axs):
            if idx is None:
                _idx = np.random.randint(len(self))
            elif isinstance(idx, Iterable):
                if i >= len(idx):
                    break
                _idx = idx[i]
            else:
                _idx = i + idx
            self.show_item(_idx, ax=ax, apply_transforms=apply_transforms, **kwargs)


def get_dataloader(df, img_path_col, label_col, path='.', transforms=None,
                   batch_size=32, shuffle=False, num_workers=4, sampler=None,
                   bbox=False, labels=None, encode_labels=True, **kwargs):
    if bbox is not False:
        dataset = ImageDataset(
            df, img_path_col, label_col, path=path, bbox=True, transforms=transforms,
            labels=labels, encode_labels=encode_labels)
    else:
        dataset = ImageDataset(
            df, img_path_col, label_col, path=path, bbox=False, transforms=transforms,
            labels=labels, encode_labels=encode_labels)

    if sampler=='weighted':
        
       dataloader = DataLoader(dataset, batch_size=batch_size, 
                           num_workers=num_workers, sampler=ClassUniformSampler(dataset), **kwargs)
        

    # elif sampler == 'augmentation':
    #     oversampled_dataset = oversample_dataset(dataset, label_col, desired_samples=100)
    #     augmented_dataset = augment_dataset(oversampled_dataset, transforms=[
    #          RandomHorizontalFlip(),
    #          RandomRotation(degrees=30),
    #          ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            
    #      ])
        
    #     dataloader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True,
    #                          num_workers=num_workers,  **kwargs)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, sampler=sampler, **kwargs)

    return dataloader
