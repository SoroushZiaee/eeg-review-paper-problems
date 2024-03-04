from typing import Type, Any, Dict, List, Tuple, Callable
from torch import Tensor
import torch

from lightning.fabric.utilities.apply_func import apply_to_collection

import numpy as np
import pickle

import logging

logger = logging.getLogger(__name__)


class _Transform(object):
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError


Transform = Type[_Transform]


class ToTensor(_Transform):
    def __init__(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

    def __call__(self, data, label):
        data = apply_to_collection(
            data,
            dtype=(np.ndarray, int, float, np.int64),
            function=lambda a: torch.tensor(a).float(),
        )
        label = apply_to_collection(
            label,
            dtype=(np.ndarray, int, float, np.int64),
            function=lambda a: torch.tensor(a).float(),
        )

        return data, label


class LabelToDict(_Transform):
    def __call__(self, data, label):
        return data, {"label": label}


class ZNorm(_Transform):
    def __init__(
        self,
        stats: str,
        mode: str = "min-max",
        max_clip_val: int = 0,
        min_clip_val: int = None,
    ):
        self.stats_name = stats
        self.mode = mode
        self.min_clip_val = min_clip_val if min_clip_val is not None else -max_clip_val
        self.max_clip_val = max_clip_val
        with open(stats, "rb") as stats_f:
            self.stats = pickle.load(stats_f)

    def __call__(self, pkg: Tuple[Dict[str, Tensor], List[int]], target: Any):
        for k, st in self.stats.items():
            if k in pkg:
                if self.mode == "min-max":
                    minx = st["min"].unsqueeze(0).to(pkg[k].device)
                    maxx = st["max"].unsqueeze(0).to(pkg[k].device)
                    pkg[k] = (pkg[k] - minx) / (maxx - minx)
                if self.mode == "mean-std":
                    mean = st["mean"].unsqueeze(0).to(pkg[k].device)
                    std = st["std"].unsqueeze(0).to(pkg[k].device)
                    pkg[k] = (pkg[k] - mean) / std
                if self.max_clip_val > 0 or self.min_clip_val is not None:
                    pkg[k] = torch.clip(
                        pkg[k], min=self.min_clip_val, max=self.max_clip_val
                    )
            else:
                raise ValueError(f"couldn't find stats key {k} in package")
        return pkg, target


class Compose(_Transform):
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, data: Any, target: Any):
        for t in self.transforms:
            data, target = t(data, target)
        return data, target

    def __repr__(self):
        return "\n".join([c.__class__.__name__ for c in self.transforms])


class ToCSP(_Transform):
    def __init__(
        self,
        stats: str = "/opt/ext/home/azureuser/workspace/pase-eeg/notebooks/University Thesis/weight_csp_no_window_no_freq.pkl",
        num_features: int = 8,
        name="csp",
    ):
        self.stats = stats
        self.num_features = num_features
        self.name = name
        self.idx = []
        self.template = "label-{}.0_filter-{}_window-{}"

        self._setup()

    def _setup(self):

        for i in range(self.num_features):
            self.idx.append(i)
        for i in reversed(self.idx):
            self.idx.append(-(i + 1))

        with open(self.stats, "rb") as fin:
            self.csp_w = pickle.load(fin)

    def _calc_feature(self, trial, W_bar):

        features = np.zeros((1, W_bar.shape[1]))

        part_1 = (W_bar.T).dot(trial)
        part_2 = (trial.T).dot(W_bar)

        tmp_element = part_1.dot(part_2)

        num = np.diag(tmp_element)
        den = np.trace(tmp_element)

        features[0, :] = np.log(num / den)

        return np.expand_dims(features[0, :], axis=0)

    def __call__(self, data, label):

        temp_data, temp_label = (
            np.transpose(
                np.vstack(
                    list(map(lambda a: np.expand_dims(a, axis=0), data.values()))
                ),
                axes=(1, 0, 2),
            ),
            label["label"],
        )

        W_bar = self.csp_w[self.template.format(temp_label, 0, 0)][:, self.idx]

        label[self.name] = self._calc_feature(temp_data[0], W_bar)

        return data, label
