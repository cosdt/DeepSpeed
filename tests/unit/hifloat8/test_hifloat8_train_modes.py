# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
HiFloat8 + DeepSpeed 训练集成测试：Qwen3-0.6B 微调。
支持 普通 / 分布式 / 图模式 / 分布式+图模式 四种训练模式。

用法:
    # 1. 普通训练（单卡，无图模式）
    deepspeed --num_gpus=1 tests/unit/hifloat8/test_hifloat8_train_modes.py --mode plain

    # 2. 分布式训练（多卡）
    deepspeed --num_gpus=2 tests/unit/hifloat8/test_hifloat8_train_modes.py --mode distributed

    # 3. 图模式训练（单卡 + torch.compile）
    deepspeed --num_gpus=1 tests/unit/hifloat8/test_hifloat8_train_modes.py --mode compile

    # 4. 分布式 + 图模式训练
    deepspeed --num_gpus=2 tests/unit/hifloat8/test_hifloat8_train_modes.py --mode distributed_compile

"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import deepspeed
import deepspeed.comm as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspeed.runtime.hifloat8 import is_hifloat8_available

os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"

MODEL_PATH = "/shared/models/Qwen3-0.6B"
MODEL_DTYPE = torch.bfloat16
DS_COMPILE_BACKEND = os.getenv("DS_COMPILE_BACKEND", "npu")

# 微调用语料：几条中文文本，够 HiFloat8 训练链路验证即可
TRAIN_TEXTS = [
    "深度学习是一种基于人工神经网络的机器学习方法。",
    "HiFloat8 是昇腾上的一种低比特训练格式，可以降低显存占用。",
    "分布式训练通过数据并行来加速大规模模型的收敛。",
    "图模式把模型编译成计算图，以减少算子调度开销。",
    "大语言模型在海量文本上预训练，再通过微调适配具体任务。",
]


def build_dataset(tokenizer, seq_len):
    """把文本编码成定长序列，构造因果语言模型的 input_ids / labels。"""
    encodings = tokenizer(
        TRAIN_TEXTS,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"]
    labels = input_ids.clone()
    # 因果 LM 训练时忽略 padding 位置（置 -100，不参与 loss）
    labels[input_ids == tokenizer.pad_token_id] = -100
    return input_ids, labels


def get_ds_config(world_size, micro_batch_size, gradient_accumulation_steps, lr):
    return {
        # 全局 batch = 每卡 micro batch * 梯度累积步数 * 卡数
        "train_batch_size": micro_batch_size * gradient_accumulation_steps * world_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "bf16": {"enabled": True},
        "hifloat8": {"enabled": True},
        # 需要 ZeRO 分片时在这里硬编码，例如：
        # "zero_optimization": {"stage": 3},
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": lr},
        },
    }


def maybe_compile_module(module: nn.Module) -> nn.Module:
    """图模式：在 HiFloat8 层替换之后编译模型，失败时回退到 eager。"""
    if not hasattr(torch, "compile"):
        print("[COMPILE] torch.compile unavailable, skip.", flush=True)
        return module
    try:
        compiled_module = torch.compile(module, backend=DS_COMPILE_BACKEND)
        print(f"[COMPILE] torch.compile enabled, backend={DS_COMPILE_BACKEND}.", flush=True)
        return compiled_module
    except Exception as error:
        print(f"[COMPILE] skip compile, reason: {error}", flush=True)
        return module


def run_training(model_engine, input_ids, labels, steps, log_interval=10):
    micro_batch = model_engine.train_micro_batch_size_per_gpu()
    num_samples = input_ids.shape[0]
    device = "npu"

    for step in range(steps):
        # 每个 rank 用自己的随机顺序取 micro-batch，模拟真实数据并行
        order = torch.randperm(num_samples)[:micro_batch]
        batch_ids = input_ids[order].to(device)
        batch_labels = labels[order].to(device)

        model_engine.zero_grad()
        out = model_engine(input_ids=batch_ids, labels=batch_labels)
        loss = out.loss
        model_engine.backward(loss)
        model_engine.step()

        if step % log_interval == 0 and dist.get_rank() == 0:
            print(f"Step {step:3d} | loss = {loss.item():.6f}", flush=True)


def inspect_param_types(model_engine, tag=""):
    """按 rank 打印参数在本地存储的 tensor 类型，确认是否 DTensor / 是否分片。"""
    from collections import Counter

    rank = dist.get_rank()
    type_counts = Counter()
    storage_counts = Counter()
    dtensor_names = []
    total_numel = 0
    samples = []

    for name, param in model_engine.module.named_parameters():
        type_counts[type(param).__name__] += 1
        if "DTensor" in type(param).__name__:
            dtensor_names.append(name)
        # ZeRO stage 3 的分区存储挂在 param.ds_tensor 上（DDP / stage1/2 无）
        storage = getattr(param, "ds_tensor", None)
        storage_counts[type(storage).__name__ if storage is not None else "None"] += 1
        total_numel += param.numel()
        if len(samples) < 3:
            samples.append((name, type(param).__name__, tuple(param.shape), param.dtype, storage))

    print(f"[rank{rank}] === {tag} 参数存储类型 ===", flush=True)
    print(f"[rank{rank}] {tag} param 类型统计: {dict(type_counts)}", flush=True)
    print(f"[rank{rank}] {tag} ds_tensor(ZeRO3分区) 统计: {dict(storage_counts)}", flush=True)
    print(f"[rank{rank}] {tag} 本地参数量 numel={total_numel}  DTensor 数量={len(dtensor_names)}", flush=True)
    for name, ttype, shape, dtype, storage in samples:
        if storage is not None:
            print(f"[rank{rank}] {tag} 示例 {name}: type={ttype} shape={shape} dtype={dtype} "
                  f"ds_tensor={type(storage).__name__} 本地分片 shape={tuple(storage.shape)} numel={storage.numel()}",
                  flush=True)
        else:
            print(f"[rank{rank}] {tag} 示例 {name}: type={ttype} shape={shape} dtype={dtype} ds_tensor=None",
                  flush=True)
    if dtensor_names:
        print(f"[rank{rank}] {tag} DTensor 参数: {dtensor_names[:5]}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="HiFloat8 + DeepSpeed training modes")
    # DeepSpeed 自己的命令行参数（--deepspeed / --deepspeed_config 等）
    deepspeed.add_config_arguments(parser)
    # deepspeed launcher 默认会注入 --local_rank=<n>，必须接收，否则 argparse 报 unrecognized
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="deepspeed launcher 注入的本地 rank（分布式训练必需）",
    )
    parser.add_argument(
        "--mode",
        choices=["plain", "distributed", "compile", "distributed_compile"],
        default="plain",
        help="plain / distributed / compile / distributed_compile",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="预训练模型路径（默认 Qwen3-0.6B）",
    )
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=1,
        help="每卡 micro batch size（默认 1）",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="训练序列长度（默认 128）",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="梯度累积步数（默认 1）",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="训练步数（默认 200）",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="学习率（默认 1e-5）",
    )

    return parser.parse_args()


def validate_mode(mode, world_size):
    is_distributed_mode = mode in ("distributed", "distributed_compile")
    if is_distributed_mode and world_size == 1:
        raise RuntimeError(
            f"mode={mode} requires multiple NPUs, but world_size={world_size}. "
            "Launch with `deepspeed --num_gpus=N ...` (N >= 2)."
        )
    if not is_distributed_mode and world_size > 1:
        print(f"[WARN] mode={mode} launched with world_size={world_size}, "
              "this run is effectively distributed.", flush=True)


def main():
    if not is_hifloat8_available():
        raise RuntimeError("HiFloat8 unavailable, run on Ascend NPU.")

    args = parse_args()

    # 先初始化 DeepSpeed 通信后端（幂等），之后才能用 dist.get_world_size() 构造 config
    deepspeed.init_distributed()
    world_size = dist.get_world_size()
    is_compile_mode = args.mode in ("compile", "distributed_compile")

    validate_mode(args.mode, world_size)

    # 不同 rank 用不同随机种子，取不同训练样本
    torch.manual_seed(42 + dist.get_rank())

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=MODEL_DTYPE,
        trust_remote_code=True,
    ).train()

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=get_ds_config(
            world_size,
            args.micro_batch,
            args.gradient_accumulation_steps,
            args.lr,
        ),
    )

    if is_compile_mode:
        model_engine = maybe_compile_module(model_engine)

    # 确认每个 rank 本地存储的参数 tensor 类型（是否 DTensor / 是否 ZeRO 分片）
    inspect_param_types(model_engine, "after-init")

    input_ids, labels = build_dataset(tokenizer, args.seq_len)

    if dist.get_rank() == 0:
        print(f"[MODE] {args.mode} | world_size={world_size} | compile={is_compile_mode} | "
              f"model={args.model_path} | seq_len={args.seq_len}", flush=True)

    run_training(model_engine, input_ids, labels, args.steps)

    # 训练后再确认一次（ZeRO3 下分区存储会在训练过程中才被分配）
    inspect_param_types(model_engine, "after-train")

    if dist.get_rank() == 0:
        print(f"HiFloat8 + DeepSpeed [{args.mode}] passed.", flush=True)


if __name__ == "__main__":
    main()
