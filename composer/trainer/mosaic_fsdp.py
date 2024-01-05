# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Released under BSD 3-Clause License,
# Copyright (c) Facebook, Inc. and its affiliates.

"""Monkey patch FSDPs _auto_wrap to enable module_kwargs and custom process_group cache and ChunkShardingSpec to enable sharding over all gpus."""

import torch
from packaging import version
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed.fsdp import FullyShardedDataParallel
from typing import Callable
import functools
from typing import Optional
import warnings

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
        # Monkey patch for torch < 2.1.1 ie torch == 2.1.0

        # Monkey patch sharding method
        ChunkShardingSpec.build_metadata = build_metadata

        # Monkey patch partial state dict handling
        from torch.distributed.fsdp import _state_dict_utils
        _state_dict_utils._sharded_pre_load_state_dict_hook = (_sharded_pre_load_state_dict_hook)

    elif version.parse(torch.__version__) >= version.parse('2.1.1'):
        raise NotImplementedError(f'FullyShardedDataParallel is not supported for torch >= 2.2.0')
    
        # Allow 2D HSDP
        from torch.distributed.fsdp import _runtime_utils
        _runtime_utils._validate_and_get_hybrid_shard_state = lambda *args, **kwargs: None

    elif version.parse(torch.__version__) < version.parse('2.2.0'):
        # Monkey patch for torch < 2.2.0 ie torch == 2.1.1, 2.1.2

        # Allow 2D HSDP
        from torch.distributed.fsdp import _runtime_utils
        _runtime_utils._validate_and_get_hybrid_shard_state = lambda *args, **kwargs: None

        # Better overlap communication and computation
        from composer.trainer.mosaic_fsdp_utils import _share_state_and_init_handle_attrs_t2p1
        _runtime_utils._share_state_and_init_handle_attrs = _share_state_and_init_handle_attrs_t2p1

    elif version.parse(torch.__version__) < version.parse('2.2.1'):
        # Monkey patch for torch < 2.2.1 ie torch == 2.2.0

        # Better overlap communication and computation
        from torch.distributed.fsdp import _runtime_utils

        from composer.trainer.mosaic_fsdp_utils import _share_state_and_init_handle_attrs_t2p2
        _runtime_utils._share_state_and_init_handle_attrs = _share_state_and_init_handle_attrs_t2p2

def patch_compressed_collectives(compress_fn: Callable,
                                 decompress_fn: Callable,
                                 compress_kwargs: Optional[dict] = None,
                                 decompress_kwargs: Optional[dict] = None):
    """Monkey patches specific collective operations to use 8 bits.

    Currently implemented for all gather and all to all.
    """
    if version.parse(torch.__version__) < version.parse('2.1.0'):
        raise NotImplementedError(f'8 bit all gather not supported for torch < 2.1.0')
    else:
        # Monkey patch _allgather_base in ProcessGroup to use 8 bits.
        from torch.distributed import ProcessGroup

        warnings.warn("Using 8 bit communication is experimental!")

        collectives_names = ['_allgather_base', 'alltoall_base']
        for name in collectives_names:
            collective = getattr(ProcessGroup, name)

            # Compress before and decompress after using the provided functions.
            @functools.wraps(collective)
            def compressed_collective(*args: tuple,
                                    coll: Callable = collective,
                                    **kwargs: dict):
                comp_coll = CompressedCollective(compress_fn, decompress_fn,
                                                compress_kwargs, decompress_kwargs)
                return comp_coll.call(coll, *args, **kwargs)

            setattr(ProcessGroup, name, compressed_collective)
        
        # Monkey patch FSDP to ignore the low precision shard during the forwards pass, saving us
        # an intermediate dtype conversion.
        from torch.distributed.fsdp.flat_param import FlatParamHandle

        def _use_low_precision_shard(self: FlatParamHandle): # pyright: ignore
            return
        
        setattr(FlatParamHandle, '_use_low_precision_shard', _use_low_precision_shard)
