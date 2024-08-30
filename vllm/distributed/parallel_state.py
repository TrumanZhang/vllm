# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""vLLM distributed state.
It takes over the control of the distributed environment from PyTorch.
The typical workflow is:

- call `init_distributed_environment` to initialize the distributed environment.
- call `initialize_model_parallel` or `ensure_model_parallel_initialized` to 
 initialize the model parallel groups.

- any code dealing with the distributed stuff

- call `destroy_model_parallel` to destroy the model parallel groups.
- call `destroy_distributed_environment` to destroy the distributed environment.

If you only need to use the distributed environment without model/pipeline
 parallelism, you can skip the model parallel initialization and destruction
 steps.
"""
import contextlib
import pickle
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import torch
import torch.distributed
from torch.distributed import Backend, ProcessGroup

import vllm.envs as envs
from vllm.logger import init_logger


@dataclass
class GraphCaptureContext:
    stream: torch.cuda.Stream


TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])


def _split_tensor_dict(
        tensor_dict: Dict[str, Union[torch.Tensor, Any]],
        prefix: str = "") -> Tuple[List[Tuple[str, Any]], List[torch.Tensor]]:
    """Split the tensor dictionary into two parts:
    1. A list of (key, value) pairs. If the value is a tensor, it is replaced
         by its metadata.
    2. A list of tensors.

    If the Tensor is nested under `tensor_dict["key1"]["key2"]`, the key of its
    metadata will be "key1%key2".
    """
    metadata_list: List[Tuple[str, Any]] = []
    tensor_list = []
    for key, value in tensor_dict.items():
        assert "%" not in key, (
            "Avoid having '%' in key "
            "as it is used as a separator for nested entries.")
        if isinstance(value, torch.Tensor):
            # Note: we cannot use `value.device` here,
            # because it contains not only the device type but also the device
            # index (e.g. "cuda:0"). We only need the device type.
            # receiving side will set the device index.
            device = value.device.type
            metadata_list.append(
                (prefix + key, TensorMetadata(device, value.dtype,
                                              value.size())))
            tensor_list.append(value)
        elif isinstance(value, dict):
            if len(value) == 0:
                metadata_list.append((prefix + key, value))
            inner_metadata_list, inner_tensor_list = _split_tensor_dict(
                value, prefix + key + "%")
            metadata_list.extend(inner_metadata_list)
            tensor_list.extend(inner_tensor_list)
        else:
            metadata_list.append((prefix + key, value))
    return metadata_list, tensor_list


def _update_nested_dict(nested_dict, flattened_key, value):
    key_splits = flattened_key.split("%")
    cur_dict = nested_dict
    for k in key_splits[:-1]:
        if k not in cur_dict:
            cur_dict[k] = {}
        cur_dict = cur_dict[k]
    cur_dict[key_splits[-1]] = value


class GroupCoordinator:
    """
    PyTorch ProcessGroup wrapper for a group of processes.
    PyTorch ProcessGroup is bound to one specific communication backend,
        e.g. NCCL, Gloo, MPI, etc.
    GroupCoordinator takes charge of all the communication operations among
        the processes in the group. It can route the communication to
        a specific implementation (e.g. switch allreduce implementation
        based on the tensor size and cuda graph mode).
    """

    # available attributes:
    rank: int  # global rank
    ranks: List[int]  # global ranks in the group
    world_size: int  # size of the group
    # difference between `local_rank` and `rank_in_group`:
    # if we have a group of size 4 across two nodes:
    # Process | Node | Rank | Local Rank | Rank in Group
    #   0     |   0  |  0   |     0      |       0
    #   1     |   0  |  1   |     1      |       1
    #   2     |   1  |  2   |     0      |       2
    #   3     |   1  |  3   |     1      |       3
    local_rank: int  # local rank used to assign devices
    rank_in_group: int  # rank inside the group
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication
    use_pynccl: bool  # a hint of whether to use PyNccl
    use_custom_allreduce: bool  # a hint of whether to use CustomAllreduce
    # communicators are only created for world size > 1
    pynccl_comm: Optional[Any]  # PyNccl communicator
    ca_comm: Optional[Any]  # Custom allreduce communicator
    shm_broadcaster: Optional[Any]  # shared memory broadcaster

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        use_pynccl: bool,
        use_custom_allreduce: bool,
    ):

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None
        logger.info("init_groupCoordinator enter")
        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend)
            logger.info("init_groupCoordinator created device_group")
            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            logger.info("init_groupCoordinator created cpu_group")
            if self.rank in ranks:
                logger.info("init_groupCoordinator init ranks,world_size.....")
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group
                self.cpu_group = cpu_group

        assert self.cpu_group is not None
        assert self.device_group is not None
        logger.info("init_groupCoordinator created group")
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")
        rank_str = ",".join(map(str, self.ranks))
        device_world_size = torch.distributed.get_world_size(self.device_group)
        cpu_world_size = torch.distributed.get_world_size(self.cpu_group)
        logger.info("init_groupCoordinator,local_rank=%d,rank=%d,rank_in_group=%d,"
                    "group_ranks=%s,world_size=%d,device_world_size=%d,cpu_world_size:%d",
                    self.local_rank, self.rank, self.rank_in_group, rank_str,
                    self.world_size, device_world_size, cpu_world_size)

        self.use_pynccl = use_pynccl
        self.use_custom_allreduce = use_custom_allreduce

        # lazy import to avoid documentation build error
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce)
        from vllm.distributed.device_communicators.pynccl import (
            PyNcclCommunicator)

        self.pynccl_comm: Optional[PyNcclCommunicator]
        if use_pynccl and self.world_size > 1:
            logger.info("init groupCoordinator--pyncc_comm")
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group,
                device=self.device,
            )
        else:
            self.pynccl_comm = None
        logger.info("end init groupCoordinator--pyncc_comm")

        self.ca_comm: Optional[CustomAllreduce]
        if use_custom_allreduce and self.world_size > 1:
            # Initialize a custom fast all-reduce implementation.
            logger.info("init groupCoordinator--customAllreduce")
            self.ca_comm = CustomAllreduce(
                group=self.cpu_group,
                device=self.device,
            )
        else:
            self.ca_comm = None
        logger.info("end init groupCoordinator--customAllreduce")

        from vllm.distributed.device_communicators.shm_broadcast import (
            ShmRingBufferIO)
        self.shm_broadcaster: Optional[ShmRingBufferIO] = None

        if self.world_size > 1 and is_in_the_same_node(self.cpu_group):
            logger.info("init groupCoordinator--shm_broastcaster")
            self.shm_broadcaster = ShmRingBufferIO.create_from_process_group(
                self.cpu_group, 1 << 22, 6)
        logger.info("end init groupCoordinator--shm_broastcaster")

    @property
    def first_rank(self):
        """Return the global rank of the first process in the group"""
        return self.ranks[0]

    @property
    def last_rank(self):
        """Return the global rank of the last process in the group"""
        return self.ranks[-1]

    @property
    def is_first_rank(self):
        """Return whether the caller is the first process in the group"""
        return self.rank == self.first_rank

    @property
    def is_last_rank(self):
        """Return whether the caller is the last process in the group"""
        return self.rank == self.last_rank

    @property
    def next_rank(self):
        """Return the global rank of the process that follows the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group + 1) % world_size]

    @property
    def prev_rank(self):
        """Return the global rank of the process that precedes the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group - 1) % world_size]

    @contextmanager
    def graph_capture(
            self, graph_capture_context: Optional[GraphCaptureContext] = None):
        if graph_capture_context is None:
            stream = torch.cuda.Stream()
            graph_capture_context = GraphCaptureContext(stream)
        else:
            stream = graph_capture_context.stream

        ca_comm = self.ca_comm
        maybe_ca_context = nullcontext(
        ) if ca_comm is None else ca_comm.capture()
        with torch.cuda.stream(stream), maybe_ca_context:
            # In graph mode, we have to be very careful about the collective
            # operations. The current status is:
            #     allreduce \ Mode   |  Eager  |  Graph  |
            # --------------------------------------------
            # custom allreduce       | enabled | enabled |
            # PyNccl                 | disabled| enabled |
            # torch.distributed      | enabled | disabled|
            #
            # Note that custom allreduce will have a runtime check, if the
            #  tensor size is too large, it will fallback to the next
            #  available option.
            # In summary: When using CUDA graph, we use
            #  either custom all-reduce kernel or pynccl. When not using
            #  CUDA graph, we use either custom all-reduce kernel or
            #  PyTorch NCCL. We always prioritize using custom all-reduce
            #  kernel but fall back to PyTorch or pynccl if it is
            #  disabled or not supported.
            pynccl_comm = self.pynccl_comm
            maybe_pynccl_context: Any
            if not pynccl_comm:
                maybe_pynccl_context = nullcontext()
            else:
                maybe_pynccl_context = pynccl_comm.change_state(
                    enable=True, stream=torch.cuda.current_stream())
            with maybe_pynccl_context:
                yield graph_capture_context

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """
        NOTE: This operation will be applied in-place or out-of-place. 
        Always assume this function modifies its input, but use the return
        value as the output.
        """
        ca_comm = self.ca_comm

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        if ca_comm is not None:
            out = ca_comm.custom_all_reduce(input_)
            if out is not None:
                return out
        pynccl_comm = self.pynccl_comm
        if (pynccl_comm is not None and not pynccl_comm.disabled):
            pynccl_comm.all_reduce(input_)
        else:
            torch.distributed.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # Allocate output tensor.
        output_tensor = torch.empty((world_size, ) + input_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        # All-gather.
        torch.distributed.all_gather_into_tensor(output_tensor,
                                                 input_,
                                                 group=self.device_group)
        # Reshape
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] +
                                              (world_size *
                                               input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor

    def gather(self,
               input_: torch.Tensor,
               dst: int = 0,
               dim: int = -1) -> torch.Tensor:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        # Gather.
        torch.distributed.gather(input_,
                                 gather_list,
                                 dst=self.ranks[dst],
                                 group=self.device_group)
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def gather_extension(self,
                         input_: torch.Tensor,
                         dst: int = 0,
                         dim: int = -1) -> torch.Tensor:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        # Gather.
        torch.distributed.gather(input_,
                                 gather_list,
                                 dst=self.ranks[dst],
                                 group=self.device_group)
        if self.rank_in_group == dst:
            output_tensor = torch.stack(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        """Broadcast the input tensor.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        # Broadcast.
        torch.distributed.broadcast(input_,
                                    src=self.ranks[src],
                                    group=self.device_group)
        return input_

    def broadcast_object(self, obj: Optional[Any] = None, src: int = 0):
        """Broadcast the input object.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj
        if self.shm_broadcaster is not None:
            assert src == 0, "Shared memory broadcaster only supports src=0"
            return self.shm_broadcaster.broadcast_object(obj)
        if self.rank_in_group == src:
            torch.distributed.broadcast_object_list([obj],
                                                    src=self.ranks[src],
                                                    group=self.cpu_group)
            return obj
        else:
            recv = [None]
            torch.distributed.broadcast_object_list(recv,
                                                    src=self.ranks[src],
                                                    group=self.cpu_group)
            return recv[0]

    def broadcast_object_list(self,
                              obj_list: List[Any],
                              src: int = 0,
                              group: Optional[ProcessGroup] = None):
        """Broadcast the input object list.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj_list
        # Broadcast.
        torch.distributed.broadcast_object_list(obj_list,
                                                src=self.ranks[src],
                                                group=self.device_group)
        return obj_list

    def send_object(self, obj: Any, dst: int) -> None:
        """Send the input object list to the destination rank."""
        """NOTE: `dst` is the local rank of the destination rank."""

        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        assert dst != self.rank, (
            "Invalid destination rank. Destination rank is the same "
            "as the current rank.")

        # Serialize object to tensor and get the size as well
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)

        size_tensor = torch.tensor([object_tensor.numel()],
                                   dtype=torch.long,
                                   device="cpu")

        # Send object size

        torch.distributed.send(size_tensor,
                               dst=self.ranks[dst],
                               group=self.cpu_group)

        # Send object
        torch.distributed.send(object_tensor,
                               dst=self.ranks[dst],
                               group=self.cpu_group)

        return None

    def recv_object(self, src: int) -> Any:
        """Receive the input object list from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""

        assert src < self.world_size, f"Invalid src rank ({src})"

        assert src != self.rank, (
            "Invalid source rank. Source rank is the same as the current rank."
        )

        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")

        # Receive object size
        rank_size = torch.distributed.recv(size_tensor,
                                           src=src,
                                           group=self.cpu_group)

        # Tensor to receive serialized objects into.
        object_tensor = torch.empty(  # type: ignore[call-overload]
            size_tensor.item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device="cpu")

        rank_object = torch.distributed.recv(object_tensor,
                                             src=src,
                                             group=self.cpu_group)

        assert rank_object == rank_size, (
            "Received object sender rank does not match the size sender rank.")

        obj = pickle.loads(object_tensor.numpy().tobytes())

        return obj

    def broadcast_tensor_dict(
        self,
        tensor_dict: Optional[Dict[str, Union[torch.Tensor, Any]]] = None,
        src: int = 0,
        group: Optional[ProcessGroup] = None,
        metadata_group: Optional[ProcessGroup] = None
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """Broadcast the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if (not torch.distributed.is_initialized() or self.world_size == 1):
            return tensor_dict

        group = self.device_group
        metadata_group = self.cpu_group
        assert src < self.world_size, f"Invalid src rank ({src})"
        src = self.ranks[src]

        rank = self.rank
        if rank == src:
            metadata_list: List[Tuple[Any, Any]] = []
            assert isinstance(
                tensor_dict,
                dict), (f"Expecting a dictionary, got {type(tensor_dict)}")
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
            # `metadata_list` lives in CPU memory.
            # `broadcast_object_list` has serialization & deserialization,
            # all happening on CPU. Therefore, we can use the CPU group.
            self.broadcast_object(metadata_list, src=src)
            async_handles = []
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    handle = torch.distributed.broadcast(tensor,
                                                         src=src,
                                                         group=metadata_group,
                                                         async_op=True)
                else:
                    # use group for GPU tensors
                    handle = torch.distributed.broadcast(tensor,
                                                         src=src,
                                                         group=group,
                                                         async_op=True)
                async_handles.append(handle)
            for async_handle in async_handles:
                async_handle.wait()

        else:
            metadata_list = self.broadcast_object(None, src=src)
            tensor_dict = {}
            async_handles = []
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(value.size,
                                         dtype=value.dtype,
                                         device=value.device)
                    if tensor.numel() == 0:
                        # Skip broadcasting empty tensors.
                        _update_nested_dict(tensor_dict, key, tensor)
                        continue
                    if tensor.is_cpu:
                        # use metadata_group for CPU tensors
                        handle = torch.distributed.broadcast(
                            tensor,
                            src=src,
                            group=metadata_group,
                            async_op=True)
                    else:
                        # use group for GPU tensors
                        handle = torch.distributed.broadcast(tensor,
                                                             src=src,
                                                             group=group,
                                                             async_op=True)
                    async_handles.append(handle)
                    _update_nested_dict(tensor_dict, key, tensor)
                else:
                    _update_nested_dict(tensor_dict, key, value)
            for async_handle in async_handles:
                async_handle.wait()
        return tensor_dict

    def send_tensor_dict(
        self,
        tensor_dict: Dict[str, Union[torch.Tensor, Any]],
        dst: Optional[int] = None
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """Send the input tensor dictionary.
        NOTE: `dst` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict

        group = self.device_group
        metadata_group = self.cpu_group

        if dst is None:
            dst = self.next_rank
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        metadata_list: List[Tuple[Any, Any]] = []
        assert isinstance(
            tensor_dict,
            dict), f"Expecting a dictionary, got {type(tensor_dict)}"
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        # `metadata_list` lives in CPU memory.
        # `send_object_list` has serialization & deserialization,
        # all happening on CPU. Therefore, we can use the CPU group.
        self.send_object(metadata_list, dst=dst)
        for tensor in tensor_list:
            if tensor.numel() == 0:
                # Skip sending empty tensors.
                continue
            if tensor.is_cpu:
                # use metadata_group for CPU tensors
                torch.distributed.send(tensor, dst=dst, group=metadata_group)
            else:
                # use group for GPU tensors
                torch.distributed.send(tensor, dst=dst, group=group)
        return None

    def recv_tensor_dict(
        self,
        src: Optional[int] = None
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """Recv the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None

        group = self.device_group
        metadata_group = self.cpu_group

        if src is None:
            src = self.prev_rank
        assert src < self.world_size, f"Invalid src rank ({src})"

        recv_metadata_list = self.recv_object(src=src)
        tensor_dict: Dict[str, Any] = {}
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size,
                                     dtype=value.dtype,
                                     device=value.device)
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    _update_nested_dict(tensor_dict, key, tensor)
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    torch.distributed.recv(tensor,
                                           src=src,
                                           group=metadata_group)
                else:
                    # use group for GPU tensors
                    torch.distributed.recv(tensor, src=src, group=group)
                _update_nested_dict(tensor_dict, key, tensor)
            else:
                _update_nested_dict(tensor_dict, key, value)
        return tensor_dict

    def barrier(self):
        """Barrier synchronization among the group.
        NOTE: don't use `device_group` here! `barrier` in NCCL is
        terrible because it is internally a broadcast operation with
        secretly created GPU tensors. It is easy to mess up the current
        device. Use the CPU group instead.
        """
        torch.distributed.barrier(group=self.cpu_group)

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = self.next_rank

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(self,
             size: torch.Size,
             dtype: torch.dtype,
             src: Optional[int] = None) -> torch.Tensor:
        """Receives a tensor from the src rank."""
        """NOTE: `src` is the local rank of the destination rank."""
        if src is None:
            src = self.prev_rank

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def send_tensor(self,
                    tensor: torch.Tensor,
                    dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = self.next_rank

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv_tensor(self,
                    tensor: torch.Tensor,
                    src: Optional[int] = None) -> torch.Tensor:
        """Receives a tensor from the src rank in the specified 
        tensor buffer."""
        """NOTE: `src` is the local rank of the destination rank."""
        if src is None:
            src = self.prev_rank

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self):
        if self.device_group is not None:
            torch.distributed.destroy_process_group(self.device_group)
            self.device_group = None
        if self.cpu_group is not None:
            torch.distributed.destroy_process_group(self.cpu_group)
            self.cpu_group = None
        if self.pynccl_comm is not None:
            self.pynccl_comm = None
        if self.ca_comm is not None:
            self.ca_comm = None
        if self.shm_broadcaster is not None:
            self.shm_broadcaster = None


_WORLD: Optional[GroupCoordinator] = None


def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, ("world group is not initialized")
    return _WORLD


def init_world_group(ranks: List[int], local_rank: int,
                     backend: str) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=False,
        use_custom_allreduce=False,
    )


def init_model_parallel_group(group_ranks: List[List[int]], local_rank: int,
                              backend: str) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=True,
        use_custom_allreduce=_ENABLE_CUSTOM_ALL_REDUCE,
    )
    # use_pynccl=Flae,
    # use_custom_allreduce=_ENABLE_CUSTOM_ALL_REDUCE,


_TP: Optional[GroupCoordinator] = None


def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, ("tensor model parallel group is not initialized")
    return _TP


# kept for backward compatibility
get_tensor_model_parallel_group = get_tp_group

_PP: Optional[GroupCoordinator] = None


def get_pp_group() -> GroupCoordinator:
    assert _PP is not None, (
        "pipeline model parallel group is not initialized")
    return _PP


# kept for backward compatibility
get_pipeline_model_parallel_group = get_pp_group

_SP: Optional[List[Optional[GroupCoordinator]]] = None


def get_sp_group(rank: int) -> GroupCoordinator:
    assert _SP is not None, (
        "pipeline model parallel groups are not initialized")
    assert rank < len(_SP) and rank >= 0, ("rank is out of range of sp groups")
    assert _SP[rank] is not None, (
        f"sequence parallel group of rank {rank} is not initialized")
    sp = _SP[rank]
    assert sp is not None, (
        f"sequence parallel group of rank {rank} is not initialized")
    return sp


@contextmanager
def graph_capture():
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the CUDA graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current CUDA stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    with get_tp_group().graph_capture() as context, get_pp_group(
    ).graph_capture(context):
        yield context


logger = init_logger(__name__)

_ENABLE_CUSTOM_ALL_REDUCE = True


def set_custom_all_reduce(enable: bool):
    global _ENABLE_CUSTOM_ALL_REDUCE
    _ENABLE_CUSTOM_ALL_REDUCE = enable


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
):
    logger.debug(
        "world_size=%d rank=%d local_rank=%d "
        "distributed_init_method=%s backend=%s", world_size, rank, local_rank,
        distributed_init_method, backend)
    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment")
        # this backend is used for WORLD
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank)
    # set the local rank
    # local_rank is not available in torch ProcessGroup,
    # see https://github.com/pytorch/pytorch/issues/122816
    if local_rank == -1:
        # local rank not set, this usually happens in single-node
        # setting, where we can use rank as local rank
        if distributed_init_method == "env://":
            local_rank = envs.LOCAL_RANK
        else:
            local_rank = rank
    global _WORLD
    if _WORLD is None:
        logger.info("####################################")
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(ranks, local_rank, backend)
    else:
        logger.info("@@@@@@@@@@@@@@@@@@@")
        assert _WORLD.world_size == torch.distributed.get_world_size(), (
            "world group already initialized with a different world size")
    logger.info("world initialized:%d", _WORLD.rank)


def initialize_sequence_parallel(
    sequence_parallel_size: int = 0,
    backend: Optional[str] = None,
) -> None:
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    tp_pp_world_size: int = world_size - sequence_parallel_size

    sp_world_size: int = sequence_parallel_size
    # Build the sequence-parallel groups.
    # Each tp or sp rank should have a sequence parallel group
    num_sequence_parallel_groups: int = tp_pp_world_size
    global _SP
    if _SP is None:
        _SP = [None] * num_sequence_parallel_groups
    group_ranks = []
    ranks_str = ""
    for i in range(num_sequence_parallel_groups):
        ranks = [i] + list(
            range(tp_pp_world_size, tp_pp_world_size + sp_world_size))
        rank_str = ",".join(map(str, ranks))
        ranks_str = ranks_str+"@"+str(i)+"#"+rank_str
        group_ranks.append(ranks)
    logger.info("init_sp,local_rank=%d,rank=%d,group_ranks=%s",
                get_world_group().local_rank,
                get_world_group().rank, ranks_str)
    index = 0
    for ranks in group_ranks:
        if get_world_group().rank in ranks:
            local_rank = get_world_group().local_rank
            rank_str = ",".join(map(str, ranks))
            logger.info("init_sp_group,local_rank=%d,rank=%d,sp_index=%d,ranks=%s",
                        local_rank, get_world_group().rank, index, rank_str)
            _SP[index] = init_model_parallel_group(
                [ranks], local_rank, backend)
        index += 1


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    sequence_parallel_size: int = 0,
    backend: Optional[str] = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.
        sequence_parallel_size: number of GPUs used for extra sequence 
            model parallelism.

    Let's say we have a total of 11 GPUs denoted by g0 ... g10 and we
    use 2 GPUs to parallelize the model tensor, 4 GPUs to parallelize
    the model pipeline, and 4 GPUs to parallelizes the sequence parallel. 
    The present function will create 4 tensor model-parallel groups, 
    2 pipeline model-parallel groups and 8 sequenc-parallel groups.
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
        and 8 sequenc-parallel groups:
            [g0, g8, g9, g10], [g1, g8, g9, g10],
            [g2, g8, g9, g10], [g3, g8, g9, g10],
            [g4, g8, g9, g10], [g5, g8, g9, g10],
            [g6, g8, g9, g10], [g7, g8, g9, g10],
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    tp_pp_world_size: int = world_size - sequence_parallel_size
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)

    if (tp_pp_world_size !=
            tensor_model_parallel_size * pipeline_model_parallel_size):
        raise RuntimeError(
            f"world_size ({tp_pp_world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})")

    # Build the tensor model-parallel groups.
    num_tensor_model_parallel_groups: int = (tp_pp_world_size //
                                             tensor_model_parallel_size)

    global _TP
    assert _TP is None, ("tensor model parallel group is already initialized")
    group_ranks = []
    ranks_str = ""
    for i in range(num_tensor_model_parallel_groups):
        ranks = list(
            range(i * tensor_model_parallel_size,
                  (i + 1) * tensor_model_parallel_size))
        rank_str = ",".join(map(str, ranks))
        ranks_str = ranks_str+"@"+str(i)+"#"+rank_str
        group_ranks.append(ranks)
    ranks=list(range(tp_pp_world_size,world_size))
    rank_str = ",".join(map(str, ranks))
    ranks_str = ranks_str+"@"+str(i)+"#"+rank_str
    group_ranks.append(ranks)
    logger.info("init_tp,local_rank=%d,rank=%d,group_ranks=%s",
                get_world_group().local_rank,
                get_world_group().rank, ranks_str)
    _TP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank, backend)

    # Build the pipeline model-parallel groups.
    num_pipeline_model_parallel_groups: int = (tp_pp_world_size //
                                               pipeline_model_parallel_size)
    global _PP
    assert _PP is None, (
        "pipeline model parallel group is already initialized")
    group_ranks = []
    ranks_str = ""
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(
            range(i, tp_pp_world_size, num_pipeline_model_parallel_groups))
        rank_str = ",".join(map(str, ranks))
        ranks_str = ranks_str+"@"+str(i)+"#"+rank_str
        group_ranks.append(ranks)
    ranks=list(range(tp_pp_world_size,world_size))
    rank_str = ",".join(map(str, ranks))
    ranks_str = ranks_str+"@"+str(i)+"#"+rank_str
    group_ranks.append(ranks)
    logger.info("init_pp,local_rank=%d,rank=%d,group_ranks=%s",
                get_world_group().local_rank,
                get_world_group().rank, ranks_str)
    
    _PP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank, backend)


def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    sequence_parallel_size: int = 0,
    backend: Optional[str] = None,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    tp_pp_size = tensor_model_parallel_size * \
        pipeline_model_parallel_size
    rank = get_world_group().rank
    logger.info("init parallel,tp=%d,pp=%d,sp=%d,rank=%d",
                tensor_model_parallel_size, pipeline_model_parallel_size,
                sequence_parallel_size, rank)
    if not model_parallel_is_initialized():
        logger.info("rank:%d tp_pp start initializing", rank)
        initialize_model_parallel(tensor_model_parallel_size,
                                  pipeline_model_parallel_size,
                                  sequence_parallel_size, backend)
        logger.info("rank:%d tp_pp initialized", rank)
    if sequence_parallel_size > 0:
        logger.info("rank:%d sp start initializing", rank)
        initialize_sequence_parallel(sequence_parallel_size, backend)
        logger.info("rank:%d sp initialized", rank)
        return

    assert (
        get_tensor_model_parallel_world_size() == tensor_model_parallel_size
    ), ("tensor parallel group already initialized, but of unexpected size: "
        f"{get_tensor_model_parallel_world_size()=} vs. "
        f"{tensor_model_parallel_size=}")
    pp_world_size = get_pp_group().world_size
    assert (pp_world_size == pipeline_model_parallel_size), (
        "pipeline parallel group already initialized, but of unexpected size: "
        f"{pp_world_size=} vs. "
        f"{pipeline_model_parallel_size=}")


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return (_TP is not None and _PP is not None)


def sequence_parallel_is_initialized(sequence_parallel_size: int,
                                     is_tp_pp_node: bool, rank: int):
    """Check if tensor and pipeline parallel groups are initialized."""
    if sequence_parallel_size == 0:
        return True
    # if is_tp_pp_node:
    sp_is_initialized = True
    if _SP is None:
        sp_is_initialized = False
    #     else:
    #         sp = _SP[rank]
    #         if sp is None:
    #             sp_is_initialized = False
    # else:
    #     sp_is_initialized = True
    #     if sequence_parallel_size != 0:
    #         sp_is_initialized = True
    #         if _SP is None:
    #             sp_is_initialized = False
    #         else:
    #             for item in _SP:
    #                 if item is None:
    #                     sp_is_initialized = False
    #                     break
    return sp_is_initialized


_TP_STATE_PATCHED = False


@contextmanager
def patch_tensor_parallel_group(tp_group: GroupCoordinator):
    """Patch the tp group temporarily until this function ends.

    This method is for draft workers of speculative decoding to run draft model
    with different tp degree from that of target model workers.

    Args:
        tp_group (GroupCoordinator): the tp group coordinator
    """
    global _TP_STATE_PATCHED
    assert not _TP_STATE_PATCHED, "Should not call when it's already patched"

    _TP_STATE_PATCHED = True
    old_tp_group = get_tp_group()
    global _TP
    _TP = tp_group
    try:
        yield
    finally:
        # restore the original state
        _TP_STATE_PATCHED = False
        _TP = old_tp_group


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_tp_group().world_size


def get_tensor_model_parallel_rank():
    global_rank = get_world_group().rank_in_group
    if global_rank < get_tp_group().world_size:
        """Return my rank for the tensor model parallel group."""
        return get_tp_group().rank_in_group
    else:
        return -1


def get_sequence_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global_rank = get_world_group().rank_in_group
    if global_rank >= get_tp_group().world_size:
        return get_sp_group(0).rank_in_group - 1
    else:
        return -1


def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _TP
    if _TP:
        _TP.destroy()
    _TP = None

    global _PP
    if _PP:
        _PP.destroy()
    _PP = None

    # Destroy _SP
    global _SP
    if _SP:
        for sp in _SP:
            if sp:
                sp.destroy()
    _SP = None


def destroy_distributed_environment():
    global _WORLD
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_in_the_same_node(pg: ProcessGroup):
    """
    This is a collective operation that checks if all processes in the group
    are in the same node. It tests if all processes are attached to the same
    memory system (shared access to shared memory).
    """
    assert torch.distributed.get_backend(
        pg) != torch.distributed.Backend.NCCL, (
            "is_in_the_same_node should be tested with a non-NCCL group.")
    # local rank inside the group
    rank = torch.distributed.get_rank(group=pg)
    world_size = torch.distributed.get_world_size(group=pg)

    # local tensor in each process to store the result
    is_in_the_same_node = torch.tensor([0] * world_size, dtype=torch.int32)

    # global ranks of the processes in the group
    ranks = torch.distributed.get_process_group_ranks(pg)

    magic_message = b"magic_message"
    shm = None

    try:
        with contextlib.suppress(OSError):
            if rank == 0:
                # create a shared memory segment
                shm = shared_memory.SharedMemory(create=True, size=128)
                shm.buf[:len(magic_message)] = magic_message
                torch.distributed.broadcast_object_list([shm.name],
                                                        src=ranks[0],
                                                        group=pg)
                is_in_the_same_node[0] = 1
            else:
                # try to open the shared memory segment
                recv = [None]
                torch.distributed.broadcast_object_list(recv,
                                                        src=ranks[0],
                                                        group=pg)
                name = recv[0]
                # fix to https://stackoverflow.com/q/62748654/9191338
                # Python incorrectly tracks shared memory even if it is not
                # created by the process. The following patch is a workaround.
                with patch("multiprocessing.resource_tracker.register",
                           lambda *args, **kwargs: None):
                    shm = shared_memory.SharedMemory(name=name)
                if shm.buf[:len(magic_message)] == magic_message:
                    is_in_the_same_node[rank] = 1
    except Exception as e:
        logger.error("Error ignored in is_in_the_same_node: %s", e)
    finally:
        if shm:
            shm.close()

    torch.distributed.barrier(group=pg)

    # clean up the shared memory segment
    with contextlib.suppress(OSError):
        if rank == 0 and shm:
            shm.unlink()
    torch.distributed.all_reduce(is_in_the_same_node, group=pg)
    logger.info("is_in_the_same_node:%d==%d",
                is_in_the_same_node.sum().item(), world_size)
    return is_in_the_same_node.sum().item() == world_size
