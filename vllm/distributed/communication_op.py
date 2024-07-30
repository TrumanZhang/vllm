from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_sp_group, get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> torch.Tensor:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(tensor_dict: Optional[Dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)


def broadcast_sp_tensor_dict(
        data_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None,
        src: int = 0):
    if not torch.distributed.is_initialized():
        return data_dict
    return get_sp_group(src).broadcast_tensor_dict(data_dict, src)


def send_sp_tensor(
        tensor: torch.Tensor = None,
        sp_group: int = 0,
        dst: int = 0):
    # `dst` is the local rank
    if not torch.distributed.is_initialized():
        return tensor_model_parallel_all_reduce
    return get_sp_group(sp_group).send_tensor(tensor, dst)


def recv_sp_tensor(
        tensor: torch.Tensor = None,
        sp_group: int = 0,
        src: int = 0):
    # `src` is the local rank.
    if not torch.distributed.is_initialized():
        return tensor
    return get_sp_group(sp_group).recv_tensor(tensor, src)
