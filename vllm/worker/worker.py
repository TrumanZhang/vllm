"""A GPU worker class."""
import gc
import os
from typing import List, Optional, Set, Tuple, Type

import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         SpeculativeConfig, VisionLanguageConfig)
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment, recv_sp_tensor,
                              send_sp_tensor, set_custom_all_reduce)
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.sequence import ExecuteModelRequest
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.embedding_model_runner import EmbeddingModelRunner
from vllm.worker.model_runner import GPUModelRunnerBase, ModelRunner
from vllm.worker.worker_base import LocalOrDistributedWorkerBase, WorkerInput

_ALAILABLE_GRAPH = False


class Worker(LocalOrDistributedWorkerBase):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        vision_language_config: Optional[VisionLanguageConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        is_driver_worker: bool = False,
        is_sp_worker: bool = False,
        model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker
        self.is_sp_worker = is_sp_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()
        self.vision_language_config = vision_language_config
        if self.vision_language_config:
            assert not self.lora_config, (
                "To be tested: vision language model with LoRA settings.")

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        speculative_args = {} if speculative_config is None \
            or (speculative_config.draft_model_config.model ==
                model_config.model) \
            or (speculative_config.draft_model_config.hf_config.model_type !=
                "mlp_speculator") else {"return_hidden_states": True}

        ModelRunnerClass: Type[GPUModelRunnerBase] = ModelRunner
        if model_runner_cls is not None:
            ModelRunnerClass = model_runner_cls
        elif self.model_config.embedding_mode:
            ModelRunnerClass = EmbeddingModelRunner
        self.model_runner: GPUModelRunnerBase = ModelRunnerClass(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config=load_config,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            vision_language_config=vision_language_config,
            is_sp_worker=is_sp_worker,
            **speculative_args,
        )
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: CacheEngine
        # Initialize gpu_cache as embedding models don't initialize kv_caches
        self.gpu_cache: Optional[List[torch.tensor]] = None
        # initialize CUDA stream for KV migration
        # SP worker only rx KV cache and master workers just tx KV cache
        # Note the scheduler should call kv_send_stream.synchronize() before
        # the next round, but it is not safe [TODO].
        if is_sp_worker:
            tp_pp_world = self.parallel_config.pipeline_parallel_size \
                * self.parallel_config.tensor_parallel_size
            self.kv_recv_streams = [
                torch.cuda.Stream() for _ in range(tp_pp_world)
            ]
        else:
            self.kv_send_stream = torch.cuda.Stream()

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.model_runner.save_sharded_state(
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config, )

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        peak_memory = self.init_gpu_memory - free_gpu_memory
        assert peak_memory > 0, (
            "Error in memory profiling. This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

        cache_block_size = self.get_cache_block_size_bytes()
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization -
             peak_memory) // cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.is_sp_worker:
            num_cpu_blocks = -1 if num_cpu_blocks == 0 else -num_cpu_blocks
            num_gpu_blocks = int(num_gpu_blocks //
                                 self.parallel_config.tensor_parallel_size)

        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int,
                         num_remote_gpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        raise_if_cache_size_invalid(num_gpu_blocks,
                                    self.cache_config.block_size,
                                    self.model_config.max_model_len)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        self.cache_config.num_remote_gpu_blocks = num_remote_gpu_blocks
        if self.is_sp_worker:
            self.cache_config.num_gpu_blocks = num_remote_gpu_blocks
            self.cache_config.num_cpu_blocks = 0

        self._init_cache_engine()
        self._warm_up_model()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config,
                                        self.device_config, self.is_sp_worker)
        self.gpu_cache = self.cache_engine.gpu_cache

    def _warm_up_model(self) -> None:
        if _ALAILABLE_GRAPH and not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    @property
    def do_metadata_sp_broadcast(self) -> bool:
        return self.parallel_config.sequence_parallel_size > 1

    @property
    def kv_cache(self) -> Optional[List[torch.Tensor]]:
        return self.gpu_cache

    @torch.inference_mode()
    def send_chunk(self, sp_group: int = 0, dst: int = 0) -> None:
        # dst is the local sp rank
        chunk_size = self.cache_config.chunk_size
        block_idx = self.cache_config.num_gpu_blocks - chunk_size

        # Use a send stream to do KV migration
        with torch.cuda.stream(self.kv_send_stream[sp_group]):
            for i in self.model_config.get_num_layers(self.parallel_config):
                chunk = self.cache_engine.get_blocks(layer=i,
                                                     start=block_idx,
                                                     step=chunk_size)
                send_sp_tensor(chunk, sp_group, dst)

    @torch.inference_mode()
    def recv_chunk(self, sp_group: int, chunk_idx: int) -> None:
        chunk_size = self.cache_config.chunk_size
        pp_size = self.parallel_config.pipeline_parallel_size
        tp_size = self.parallel_config.tensor_parallel_size
        paralled_blocks = int(self.cache_config.num_gpu_blocks /
                              (pp_size * tp_size))
        block_idx = sp_group * paralled_blocks + chunk_size * chunk_idx

        # Use a recv stream to do KV migration
        with torch.cuda.stream(self.kv_recv_streams[sp_group]):
            for i in self.model_config.get_num_layers(self.parallel_config):
                chunk = self.cache_engine.get_blocks(layer=i,
                                                     start=block_idx,
                                                     step=chunk_size)
                recv_sp_tensor(chunk, sp_group, src=0)

    @torch.inference_mode()
    def migrate_chunk(self, dst_chunk: int, dst_rank: int) -> None:
        """
        Migrate a KV chunk. Note that dst_rank is a local rank in the SP group
        """
        # Temporary method to get the rank in sp group.
        pp_size = self.parallel_config.pipeline_parallel_size
        tp_size = self.parallel_config.tensor_parallel_size
        dst_global_rank = dst_rank + pp_size * tp_size - 1

        # Only the master workers or the target sp workers
        # involves in the KV cache migration.
        for src_global_rank in range(pp_size * tp_size):
            if self.rank == src_global_rank:
                self.send_chunk(sp_group=src_global_rank, dst=dst_rank)
            elif self.rank == dst_global_rank:
                self.recv_chunk(sp_group=src_global_rank, chunk_idx=dst_chunk)

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
        # they contain parameters to launch cudamemcpyasync.
        blocks_to_swap_in = torch.tensor(execute_model_req.blocks_to_swap_in,
                                         device="cpu",
                                         dtype=torch.int64).view(-1, 2)
        blocks_to_swap_out = torch.tensor(execute_model_req.blocks_to_swap_out,
                                          device="cpu",
                                          dtype=torch.int64).view(-1, 2)
        # `blocks_to_copy` is a gpu tensor. The src and tgt of
        # blocks to copy are in the same device, and `blocks_to_copy`
        # can be used directly within cuda kernels.
        blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
                                      device=self.device,
                                      dtype=torch.int64).view(-1, 2)

        # # `blocks_to_migrate` is a gpu tensor. The src blocks are in the local
        # # GPU cache, while the tgt blocks are in the GPU chunk.
        # blocks_to_migrate = torch.tensor(execute_model_req.blocks_to_migrate,
        #                               device=self.device,
        #                               dtype=torch.int64).view(-1, 2)
        # `superblock_to_migrate` is a gpu tensor which records the dest chunk
        # in a remote SP GPU worker
        superblock_to_migrate = torch.tensor(
            execute_model_req.superblock_to_migrate,
            device=self.device,
            dtype=torch.int64).view(-1)

        return WorkerInput(num_seq_groups=num_seq_groups,
                           blocks_to_swap_in=blocks_to_swap_in,
                           blocks_to_swap_out=blocks_to_swap_out,
                           blocks_to_copy=blocks_to_copy,
                           superblock_to_migrate=superblock_to_migrate)

    @torch.inference_mode()
    def execute_worker(self, worker_input: WorkerInput) -> None:
        # Issue cache operations.
        if (worker_input.blocks_to_swap_in is not None
                and worker_input.blocks_to_swap_in.numel() > 0):
            self.cache_engine.swap_in(worker_input.blocks_to_swap_in)
        if (worker_input.blocks_to_swap_out is not None
                and worker_input.blocks_to_swap_out.numel() > 0):
            self.cache_engine.swap_out(worker_input.blocks_to_swap_out)
        if (worker_input.blocks_to_copy is not None
                and worker_input.blocks_to_copy.numel() > 0):
            self.cache_engine.copy(worker_input.blocks_to_copy)
        # Note sequence parallel requires master workers (in sp and tp)
        # to first copy blocks their chunk memories, and then migrate
        # the chunk to the tgt sp worker
        if (worker_input.superblock_to_migrate is not None
                and worker_input.superblock_to_migrate.numel() > 0):
            chunk_to_migrate = worker_input.superblock_to_migrate.tolist()
            self.migrate_chunk(dst_chunk=chunk_to_migrate[0],
                               dst_rank=chunk_to_migrate[1])

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return CacheEngine.get_cache_block_size(self.cache_config,
                                                self.model_config,
                                                self.parallel_config)


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size,
                                      parallel_config.sequence_parallel_size)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")


def raise_if_cache_size_invalid(num_gpu_blocks, block_size,
                                max_model_len) -> None:
    if num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * num_gpu_blocks
    if max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")
