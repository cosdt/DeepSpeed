# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
HiFloat8 + DeepSpeed 训练集成测试。

用法:
    deepspeed --num_gpus=1 tests/unit/hifloat8/test_hifloat8_train.py
"""

import torch
import torch.nn as nn
import deepspeed
import deepspeed.comm as dist

from deepspeed.runtime.hifloat8 import is_hifloat8_available


def build_model(in_features=64, hidden_features=128, out_features=10):
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, out_features),
    )


def get_ds_config():
    return {
        "train_batch_size": 8,
        "train_micro_batch_size_per_gpu": 8,
        "bf16": {"enabled": True},
        "hifloat8": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 1e-3},
        },
    }


def main():
    if not is_hifloat8_available():
        raise RuntimeError("HiFloat8 unavailable, run on Ascend NPU.")

    model = build_model().npu().bfloat16()

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=get_ds_config(),
    )

    x = torch.randn(8, 64, device="npu", dtype=torch.bfloat16)
    target = torch.randint(0, 10, (8,), device="npu")

    for step in range(200):
        model_engine.zero_grad()
        out = model_engine(x)
        loss = nn.functional.cross_entropy(out.float(), target)
        model_engine.backward(loss)
        model_engine.step()

        if step % 50 == 0 and dist.get_rank() == 0:
            print(f"Step {step:3d} | loss = {loss.item():.6f}")

    if dist.get_rank() == 0:
        print("HiFloat8 + DeepSpeed train passed.")


if __name__ == "__main__":
    main()
