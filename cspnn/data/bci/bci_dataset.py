from typing import Dict, Tuple, Union, List
import os
import random
import logging
import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .moabb_data import prepare_and_convert_data, Paradigm
from ..transforms import Transform

import warnings


class BCI2aDataset(Dataset):
    def __init__(
        self,
        eeg_electrode_positions: Dict[str, Tuple[int, int]],
        data_path: str,
        meta_data: pd.DataFrame = None,
        meta_data_path=None,
        transforms: Transform = None,
        in_mem: bool = False,
        patients: List[int] = None,
    ):
        self.eeg_electrode_positions = eeg_electrode_positions
        self.data_path = data_path

        if meta_data is not None:
            self.meta_data = meta_data
        elif meta_data_path:
            self.meta_data_path = meta_data_path
            self.meta_data = pd.read_csv(meta_data_path)
        else:
            meta_data_path = prepare_and_convert_data(
                data_path, Paradigm.FOUR_CLASSES, patients=patients
            )
            self.meta_data = pd.read_csv(meta_data_path)

        self.transforms = transforms
        self.in_mem = in_mem

        if self.in_mem:
            self._load_in_memory()

    def _load_in_memory(self):
        self.cache = []

        for i in range(len(self)):
            self.cache.append(self._getitem(i))

    def get_ptients(self):
        return self.meta_data["patient"].values

    def get_labels(self):
        return self.meta_data["label"].values

    def get_sampling_rate(self):
        return 250

    def get_resampling_rate(self):
        return 256

    def get_class_distribution(self):
        return self.meta_data["label"].value_counts()

    def __len__(self) -> int:
        return len(self.meta_data["file_name"])

    def __getitem__(self, idx: int) -> Union[dict, torch.Tensor]:
        if self.in_mem:
            return self.cache[idx]
        else:
            return self._getitem(idx=idx)

    def _getitem(self, idx: int) -> Union[dict, torch.Tensor]:
        # root_logger = logging.getLogger("mne")
        # root_logger.setLevel(logging.ERROR)
        # mne.set_log_level(verbose="ERROR")
        # warnings.simplefilter("ignore")

        meta_data = self.meta_data.iloc[idx]

        # shape -> (num_channels, n_times)
        eeg_data = np.load(os.path.join(self.data_path, meta_data["file_name"]))

        #         info = mne.create_info(
        #             list(self.eeg_electrode_positions.keys()),
        #             sfreq=self.get_sampling_rate(),
        #             ch_types="eeg",
        #             verbose=0,
        #         )

        #         eeg_data = (
        #             mne.io.RawArray(eeg_data, info, verbose=0)
        #             .filter(l_freq=2, h_freq=None, verbose=0)
        #             .resample(self.get_resampling_rate(), verbose=0)
        #         )

        #         eeg_data = eeg_data.get_data()

        label = int(meta_data["label"]) - 1

        wav = {
            key: np.expand_dims(eeg_data[i], axis=0)
            for i, key in enumerate(self.eeg_electrode_positions.keys())
        }

        if self.transforms is not None:
            wav, label = self.transforms(wav, label)

        return wav, label

    def subset(self, indices):
        return self.__class__(
            eeg_electrode_positions=self.eeg_electrode_positions,
            data_path=self.data_path,
            meta_data=self.meta_data.iloc[indices],
            transforms=self.transforms,
            in_mem=self.in_mem,
        )

    def get_train_test_subsets(self):
        t_ds = self.subset(
            np.argwhere(
                ["train" in item for item in self.meta_data["session"]]
            ).squeeze()
        )
        e_ds = self.subset(
            np.argwhere(
                ["test" in item for item in self.meta_data["session"]]
            ).squeeze()
        )

        return t_ds, e_ds

    @staticmethod
    def dict_to_2d_wave(dict_signals):
        return np.vstack([wav for wav in dict_signals.values()])

    @staticmethod
    def collate_fn(batch):
        imgs = {
            key: torch.vstack([item[0][key].unsqueeze(0) for item in batch])
            for key in batch[0][0].keys()
        }
        trgts = torch.vstack([item[1] for item in batch]).squeeze()

        return [imgs, trgts]
