# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os.path as osp

from uapi import PaddleModel, Config


def build_model(model_name):
    config = Config(model_name)
    # Set dataset params
    config.update({
        'train_dataset': {
            'num_classes': 2,
            'transforms': [{
                'type': 'Resize',
                'target_size': [398, 224]
            }, {
                'type': 'RandomHorizontalFlip'
            }, {
                'type': 'RandomDistort',
                'brightness_range': 0.4,
                'contrast_range': 0.4,
                'saturation_range': 0.4
            }, {
                'type': 'Normalize'
            }],
            'mode': 'train'
        },
        'val_dataset': {
            'num_classes': 2,
            'transforms': [{
                'type': 'Resize',
                'target_size': [398, 224]
            }, {
                'type': 'Normalize'
            }],
            'mode': 'val'
        },
    })

    model = PaddleModel(config=config)
    return model


def test_model(model):
    if isinstance(model, str):
        # In this case `model` is model name
        model_name = model
        model = build_model(model)
    else:
        model_name = model.name

    # Hard-code paths
    save_dir = f"uapi/tests/output/{model_name}_res"
    dataset_dir = "uapi/tests/data/mini_supervisely"
    infer_input_path = "uapi/tests/data/mini_supervisely/Images/baby-boy-hat-covered-101537.png"

    weight_path = osp.join(save_dir, "iter_10", "model.pdparams")
    export_dir = osp.join(save_dir, 'infer')
    pred_save_dir = osp.join(save_dir, 'pred_res')
    infer_save_dir = osp.join(save_dir, 'infer_res')
    compress_save_dir = osp.join(save_dir, 'compress')

    # Do test
    model.train(
        dataset=dataset_dir,
        batch_size=1,
        epochs_iters=10,
        device='gpu:0,1',
        amp='O1',
        save_dir=save_dir)

    model.predict(
        weight_path=weight_path,
        device='gpu',
        input_path=infer_input_path,
        save_dir=pred_save_dir)

    model.export(
        weight_path=weight_path,
        input_shape=[1, 3, 256, 256],
        save_dir=export_dir)

    model.infer(
        model_dir=export_dir,
        device='gpu',
        input_path=infer_input_path,
        save_dir=infer_save_dir)

    model.compression(
        dataset=dataset_dir,
        batch_size=2,
        learning_rate=0.1,
        epochs_iters=10,
        device='cpu',
        weight_path=weight_path,
        save_dir=compress_save_dir)
