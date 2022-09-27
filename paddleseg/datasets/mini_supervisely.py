# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy

import cv2
import numpy as np

from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
from paddleseg.datasets import Dataset

URL = "https://paddleseg.bj.bcebos.com/dataset/Supervisely_face.zip"


@manager.DATASETS.add_component
class MiniSUPERVISELY(Dataset):
    """
    Supervise.ly dataset `https://supervise.ly/`.
    """

    def __init__(self, transforms, dataset_root, mode='train', edge=False):
        super().__init__(
            mode=mode,
            edge=edge,
            transforms=transforms,
            dataset_root=dataset_root,
            num_classes=2,
            val_path=os.path.join(dataset_root, 'val.txt'))
