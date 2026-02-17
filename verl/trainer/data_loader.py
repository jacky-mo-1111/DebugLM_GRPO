# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from typing import Optional

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    def _build_dataset(data_path: str) -> RLHFDataset:
        return RLHFDataset(
            data_path=data_path,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
    )

    def _build_sampler(dataset: RLHFDataset, seed_offset: int = 0):
        if config.shuffle:
            generator = torch.Generator()
            generator.manual_seed(config.seed + seed_offset)
            return RandomSampler(data_source=dataset, generator=generator)
        return SequentialSampler(data_source=dataset)

    def _build_loader(dataset: RLHFDataset, seed_offset: int = 0) -> StatefulDataLoader:
        return StatefulDataLoader(
            dataset=dataset,
            batch_size=train_batch_size,
            sampler=_build_sampler(dataset, seed_offset=seed_offset),
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    if config.alt_update:
        if config.debug_train_files is None or config.normal_train_files is None:
            raise ValueError("When data.alt_update=True, both data.debug_train_files and data.normal_train_files are required.")

        debug_dataset = _build_dataset(config.debug_train_files)
        normal_dataset = _build_dataset(config.normal_train_files)

        debug_dataloader = _build_loader(debug_dataset, seed_offset=0)
        normal_dataloader = _build_loader(normal_dataset, seed_offset=1)
        train_dataloader = {"debug": debug_dataloader, "normal": normal_dataloader}
    else:
        train_dataset = _build_dataset(config.train_files)
        train_dataloader = _build_loader(train_dataset, seed_offset=0)

    val_dataset = RLHFDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
    )

    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    if config.alt_update:
        assert len(train_dataloader["debug"]) >= 1
        assert len(train_dataloader["normal"]) >= 1
        print(f"Size of debug train dataloader: {len(train_dataloader['debug'])}")
        print(f"Size of normal train dataloader: {len(train_dataloader['normal'])}")
    else:
        assert len(train_dataloader) >= 1
        print(f"Size of train dataloader: {len(train_dataloader)}")
    assert len(val_dataloader) >= 1
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
