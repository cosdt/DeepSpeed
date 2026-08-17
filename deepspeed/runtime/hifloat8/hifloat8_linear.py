# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
HiFloat8 Training Support for DeepSpeed.

Re-exports torch_npu's HiFloat8 training APIs. Conversion is triggered
automatically by ``deepspeed.initialize()`` when ``"hifloat8": {"enabled": true}``
is set in the DeepSpeed config.

For standalone use (without DeepSpeed engine):
    from deepspeed.runtime.hifloat8 import (
        is_hifloat8_available,
        convert_to_hifloat8_training,
    )
"""

from torch_npu.utils.hifloat8_train.hifloat8_linear import (
    HiFloat8Linear,
    convert_to_hifloat8_training,
)


def is_hifloat8_available() -> bool:
    """Return True if HiFloat8 training is available on the current system."""
    try:
        import torch_npu  # noqa: F401
        return True
    except ImportError:
        return False
