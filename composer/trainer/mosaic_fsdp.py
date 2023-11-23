# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Released under BSD 3-Clause License,
# Copyright (c) Facebook, Inc. and its affiliates.

"""Monkey patch FSDPs _auto_wrap to enable module_kwargs and custom process_group cache and ChunkShardingSpec to enable sharding over all gpus."""

import torch
from packaging import version
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed.fsdp import FullyShardedDataParallel
from typing import Callable, Optional
import functools

from composer.trainer.mosaic_fsdp_utils import (_sharded_pre_load_state_dict_hook, build_metadata,
                                                custom_auto_wrap_t1p13p1, CompressedCollective)


def patch_pytorch():
    """Monkey patches pytorch functions based on pytorch version."""
    if version.parse(torch.__version__) < version.parse('1.13.1'):
        raise NotImplementedError(f'Not supported for torch < 1.13.1')

    elif version.parse(torch.__version__) < version.parse('2.0.0'):
        # Monkey patch for torch < 2.0 ie torch == 1.13.1

        # Monkey patch _auto_wrap with _custom_auto_wrap fn
        FullyShardedDataParallel._auto_wrap = custom_auto_wrap_t1p13p1  # type: ignore

    elif version.parse(torch.__version__) < version.parse('2.0.1'):
        raise NotImplementedError(f'Not supported for torch == 2.0.0')

    elif version.parse(torch.__version__) < version.parse('2.0.2'):
        # Monkey patch for torch == 2.0.1

        # Monkey patch __init__ where __init__ calls the custom _auto_wrap fn
        from composer.trainer.mosaic_fsdp_utils import init_fn_t2p0p1

        FullyShardedDataParallel.__init__ = init_fn_t2p0p1  # type: ignore

        # Monkey patch sharding method
        ChunkShardingSpec.build_metadata = build_metadata

    elif version.parse(torch.__version__) < version.parse('2.1.1'):
        # Monkey path for torch < 2.1.1 ie torch == 2.1.0
        from torch.distributed.fsdp import _state_dict_utils

        # Monkey patch sharding method
        ChunkShardingSpec.build_metadata = build_metadata

        # Monkey patch partial state dict handling
        _state_dict_utils._sharded_pre_load_state_dict_hook = (_sharded_pre_load_state_dict_hook)

    elif version.parse(torch.__version__) >= version.parse('2.1.1'):
        raise NotImplementedError(f'FullyShardedDataParallel is not supported for torch >= 2.2.0')
    
def patch_compressed_collectives(compress_fn: Callable,
                                 decompress_fn: Callable,
                                 compress_kwargs: Optional[dict] = None,
                                 decompress_kwargs: Optional[dict] = None):
    """Monkey patches specific collective operations. Currently only all gather for FSDP."""
    if version.parse(torch.__version__) < version.parse('2.1.0'):
        raise NotImplementedError(f'8 bit all gather not supported for torch < 2.1.0')
    elif version.parse(torch.__version__) >= version.parse('2.1.1'):
        raise NotImplementedError(f'8 bit all gather not supported for torch >= 2.2.0')
    else:
        # Monkey patch _allgather_base in ProcessGroup to use 8 bits.
        from torch.distributed import ProcessGroup

        collectives_names = ['_allgather_base']
        for name in collectives_names:
            collective = getattr(ProcessGroup, name)

            # Compress before and decompress after using the provided functions.
            @functools.wraps(collective)
            def compressed_collective(*args, **kwargs):
                comp_coll = CompressedCollective(compress_fn,
                                        decompress_fn,
                                        compress_kwargs,
                                        decompress_kwargs)
                return comp_coll.call(collective, *args, **kwargs)
            
            setattr(ProcessGroup, name, compressed_collective)
