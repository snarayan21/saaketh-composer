# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import DataLoader

from composer.callbacks import MemoryMonitor
from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


def test_memory_monitor_warnings_on_cpu_models():
    with pytest.warns(UserWarning, match='The memory monitor only works on CUDA devices'):
        Trainer(
            model=SimpleModel(),
            callbacks=MemoryMonitor(),
            device='cpu',
            train_dataloader=DataLoader(RandomClassificationDataset()),
            max_duration='1ba',
        )


@pytest.mark.gpu
def test_memory_monitor_gpu():
    # Construct the trainer
    memory_monitor = MemoryMonitor()
    in_memory_logger = InMemoryLogger()
    trainer = Trainer(
        model=SimpleModel(),
        callbacks=memory_monitor,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration='1ba',
    )
    trainer.fit()

    num_memory_monitor_calls = len(in_memory_logger.data['memory/peak_allocated_mem'])

    assert num_memory_monitor_calls == int(trainer.state.timestamp.batch)
