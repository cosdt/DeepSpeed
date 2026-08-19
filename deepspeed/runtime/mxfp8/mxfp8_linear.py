# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
MXFP8 Training Support for DeepSpeed.

Re-exports torch_npu's MXFP8 training APIs. Conversion is triggered
automatically by ``deepspeed.initialize()`` when ``"mxfp8": {"enabled": true}``
is set in the DeepSpeed config.

For standalone use (without DeepSpeed engine):
    from deepspeed.runtime.mxfp8 import (
        is_mxfp8_available,
        convert_to_mxfp8_training,
    )
"""

from torch_npu.utils.mxfloat8_train.mxfp8_linear import (
    MxFP8Linear,
    convert_to_mxfp8_training,
)


def is_mxfp8_available() -> bool:
    """Return True if MXFP8 training is available on the current system."""
    try:
        import torch_npu  # noqa: F401
        return True
    except ImportError:
        return False
