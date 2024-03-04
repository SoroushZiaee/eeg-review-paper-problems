from typing import List
import os
import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from tqdm import tqdm
from enum import Enum


class Paradigm(Enum):
    FOUR_CLASSES = 1
    TWO_CLASSES = 2


_label_map_four_classes = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
_label_map_two_classes = {"left_hand": 1, "right_hand": 2}


class BCIDataset:
    def __init__(
        self,
        data_path=None,
        patients=list(range(1, 10)),
        paradigm: Paradigm = Paradigm.FOUR_CLASSES,
    ):
        assert data_path is not None, "you need to specify the dataset root directory"

        self.patients = patients
        self.data_path = data_path

        self.dataset = BNCI2014_001()
        self.download_patient()

        if paradigm == Paradigm.FOUR_CLASSES:
            self.paradigm = MotorImagery(n_classes=4)
            self.label_map = _label_map_four_classes
        if paradigm == Paradigm.TWO_CLASSES:
            self.paradigm = MotorImagery(n_classes=2)
            self.label_map = _label_map_two_classes

    def download_patient(self):
        self.dataset.download(
            subject_list=self.patients, path=os.path.join(self.data_path, "raw")
        )

    def get_data(self):
        X, labels, meta = self.paradigm.get_data(
            dataset=self.dataset, subjects=self.patients
        )

        return X, labels, meta


def prepare_and_convert_data(
    base_path: str,
    paradigm: Paradigm,
    patients: List[int] = None,
) -> str:
    if paradigm == Paradigm.FOUR_CLASSES:
        path_prefix = "four_classes"
    elif paradigm == Paradigm.TWO_CLASSES:
        path_prefix = "two_classes"
    meta_data_path = os.path.join(base_path, path_prefix, "metadata.csv")

    if os.path.exists(meta_data_path):
        return meta_data_path
    elif not os.path.exists(os.path.join(base_path, path_prefix)):
        os.makedirs(os.path.join(base_path, path_prefix))

    obj = (
        BCIDataset(data_path=base_path, paradigm=paradigm)
        if patients is None
        else BCIDataset(data_path=base_path, paradigm=paradigm, patients=patients)
    )
    X, labels, meta = obj.get_data()

    # mask = [
    #     any(tup)
    #     for tup in zip(*[labels == l for l in obj.label_map.keys()]
    #     )
    # ]

    y = labels.copy()
    for k, l in obj.label_map.items():
        y[y == k] = l
    y = y.astype(int)

    with open(meta_data_path, "w") as f:
        f.write("file_name,patient,session,run,label\n")
        for i in tqdm(range(len(meta)), desc="prepare patients"):
            patient = meta.iloc[i]["subject"]
            session = meta.iloc[i]["session"]
            run = meta.iloc[i]["run"]

            file_name = f"{path_prefix}/{patient}_{session}_{run}_{y[i]}_{i}.npy"
            f.write("{},{},{},{},{}\n".format(file_name, patient, session, run, y[i]))

            file = os.path.join(base_path, file_name)
            if os.path.exists(file):
                continue

            np.save(
                file,
                X[i, :, :],
            )

    return meta_data_path
