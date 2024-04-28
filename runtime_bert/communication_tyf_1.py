# pylint: disable=C0114,C0115,C0116,E1101

from concurrent.futures import ThreadPoolExecutor
import os
import queue
import threading
from typing import List

import torch
import torch.distributed as dist


class LoadBalancer(object):

    def __init__(self) -> None:
        self._previous_tables = {}
        self._next_tables = {}
        self._local_targets = None

    def get_splits_table(self, tensor_name: str, previous: bool):
        tables = self._previous_tables if previous else self._next_tables
        return tables[tensor_name]

    @staticmethod
    def _get_target_batch(profiles, ranks, batch_size) -> torch.Tensor:
        profiles = profiles[ranks]
        profiles = profiles / profiles.sum()
        targets = (profiles * batch_size).round().to(torch.int32)
        diff = batch_size - targets.sum()
        num_elements = targets.numel()
        diff_per_element = diff // num_elements
        remaining_diff = diff % num_elements
        targets += diff_per_element
        targets[:remaining_diff] += 1
        return targets

    def _make_table(self, send_targets: torch.Tensor, recv_targets: torch.Tensor) -> torch.Tensor:
        send_targets = send_targets.clone()
        recv_targets = recv_targets.clone()
        num_senders = len(send_targets)
        num_receivers = len(recv_targets)
        table = torch.zeros(num_senders, num_receivers, dtype=torch.int32)

        for i in range(num_senders):
            for j in range(num_receivers):
                table[i, j] = min(send_targets[i], recv_targets[j])
                send_targets[i] -= table[i, j]
                recv_targets[j] -= table[i, j]

        return table

    def initialize(self, receive_ranks, send_ranks, dp_ranks, profiles, batch_size):
        local_targets = self._get_target_batch(
            profiles, dp_ranks, batch_size)
        self._local_targets = local_targets
        for tensor_name, send_ranks_list in send_ranks.items():
            if len(send_ranks_list):
                send_targets = self._get_target_batch(
                    profiles, send_ranks_list, batch_size
                )
                self._next_tables[tensor_name] = self._make_table(
                    local_targets, send_targets)
        for tensor_name, receive_ranks_list in receive_ranks.items():
            if len(receive_ranks_list):
                receive_targets = self._get_target_batch(
                    profiles, receive_ranks_list, batch_size
                )
                self._previous_tables[tensor_name] = self._make_table(
                    receive_targets, local_targets).T


class CommunicationHandler(object):

    def __init__(
        self,
        master_addr, master_port,
        rank, local_rank, num_ranks_in_server,
        world_size, fp16, backend,
    ) -> None:

        # attrs used for initialize the distributed environment.
        self.rank = rank
        self.local_rank = local_rank
        self.backend = backend
        self.num_ranks_in_server = num_ranks_in_server
        self.world_size = world_size
        self.fp16 = fp16

        # attrs later initialized in method initialize
        self.receive_ranks = None
        self.send_ranks = None
        self.tensor_tags = None
        self.dp_ranks = None
        self._training_tensor_dtypes = None
        self._profiles = None
        self._target_tensor_names = None

        # tensor shapes
        self.tensor_shapes = {}

        # private attrs
        self._fut_queue = queue.Queue()
        self._waiter_thread = threading.Thread(
            target=self._waiter, daemon=True)
        self._local_dp_rank = -1
        self._load_balancer = LoadBalancer()
        self._load_dp_rank = -1
        self._wait_cond = threading.Condition()

        self._recv_buffers = {}
        self._thread_pool = None
        self._thread_pool_recv = ThreadPoolExecutor()
        self._recv_futures = {}

        assert num_ranks_in_server > 0
        assert not self.fp16
        assert backend is None or backend == "nccl" or backend == "gloo"

        # Initialize the distributed environment.
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        assert dist.get_world_size() == self.world_size
        print(f"Finished initializing process group; backend: {backend}, rank: {rank}"
              f"world_size: {world_size}")

    def get_local_batches(self):
        # pylint: disable=W0212
        assert self._load_balancer._local_targets is not None
        return self._load_balancer._local_targets
        # pylint: enable=W0212

    def initialize(self,
                   profiles: torch.Tensor,
                   receive_ranks, send_ranks, dp_ranks: List,
                   tensor_tags, target_tensor_names, training_tensor_dtypes,
                   rank_in_stage,
                   num_ranks_in_stage,
                   ranks_in_previous_stage,
                   ranks_in_next_stage,
                   batch_size
                   ):
        self.receive_ranks = receive_ranks
        self.send_ranks = send_ranks
        self.tensor_tags = tensor_tags
        self.dp_ranks = dp_ranks
        self._local_dp_rank = dp_ranks.index(self.rank)
        self._training_tensor_dtypes = training_tensor_dtypes
        self._profiles = profiles
        self._target_tensor_names = target_tensor_names

        self._recv_buffers = {}
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=False)
        self._thread_pool = ThreadPoolExecutor()

        self._thread_pool_recv.shutdown(True)
        self._recv_futures.clear()
        self._thread_pool_recv = ThreadPoolExecutor()

        for target_tensor_name in self._target_tensor_names:
            # Queues for target in forward pass.
            if len(ranks_in_next_stage) > 0:
                self.send_ranks[target_tensor_name] = ranks_in_next_stage
            if len(ranks_in_previous_stage) > 0:
                self.receive_ranks[target_tensor_name] = ranks_in_previous_stage

        self._load_balancer.initialize(
            self.receive_ranks,
            self.send_ranks,
            self.dp_ranks,
            self._profiles,
            batch_size
        )

    def _waiter(self):
        while True:
            future = self._fut_queue.get()

            # print("waiter",id(future))
            if future is None:
                return
            else:
                future.wait()
            if self._fut_queue.empty():
                with self._wait_cond:
                    self._wait_cond.notify_all()

    def set_tensor_shapes(self, tensor_shapes):
        self.tensor_shapes = tensor_shapes

    def start_helper_threads(self, *_, **__):
        self._waiter_thread.start()
    # from memory_profiler import profile
    # @profile
    def send(self,
             tensor_name, tensor: torch.Tensor,
             forward_minibatch_id,
             backward_minibatch_id,
             backward=False):
        def send_local_task(tensor_name, tensor):

            if self.backend == "gloo":
                tensor = tensor.cpu()
            target_ = self.receive_ranks if backward else self.send_ranks
            target_ranks = target_[tensor_name]
            split_table = self._load_balancer.get_splits_table(
                tensor_name, backward)
            splits = split_table[self._local_dp_rank]

            current_idx = 0
            for i, target_rank in enumerate(target_ranks):
                future = dist.isend(
                    tensor=tensor[current_idx:current_idx + splits[i]],
                    dst=target_rank,
                    tag=self.tensor_tags[tensor_name]
                )
                # print("send",id(future),tensor_name,backward)
                # if tensor_name=='target':
                #     print("in send",self.rank,tensor_name,target_rank,tensor[current_idx:current_idx + splits[i]],tensor)
                self._fut_queue.put(future)
                current_idx += splits[i]

        self._thread_pool.submit(send_local_task, tensor_name, tensor)

    def _recv_local_task(self, tensor_name, backward=False):
        target_ = self.send_ranks if backward else self.receive_ranks
        target_ranks = target_[tensor_name]
        split_table = self._load_balancer.get_splits_table(
            tensor_name, not backward)
        shape = self.tensor_shapes[tensor_name]

        splits = split_table[self._local_dp_rank]

        if tensor_name in self._recv_buffers:

            tensor = self._recv_buffers[tensor_name]
        elif self.backend == "gloo":

            tensor = torch.zeros(
                splits.sum(), *shape[1:], dtype=self._training_tensor_dtypes[tensor_name])
            self._recv_buffers[tensor_name] = tensor
        else:
            tensor = torch.zeros(
                splits.sum(), *shape[1:], dtype=self._training_tensor_dtypes[tensor_name]).cuda(self.local_rank)
            self._recv_buffers[tensor_name] = tensor

        current_idx = 0

        futures = []

        for i, target_rank in enumerate(target_ranks):
            futures.append(dist.irecv(
                tensor=tensor[current_idx:current_idx+splits[i]],
                src=target_rank,
                tag=self.tensor_tags[tensor_name]
            ))

            current_idx += splits[i]

        for future in futures:
            future.wait()
        if self.backend == "gloo":
            # tensor_ = tensor.clone().detach()
            # if tensor_name == 'target':
            #     print("in recv", self.rank, tensor_name, target_rank, tensor)
            tensor = tensor.cuda(self.local_rank)
            assert tensor.is_cuda
            if tensor.dtype == torch.float32:
                tensor = tensor.requires_grad_()
        return tensor

    # from memory_profiler import profile
    # @profile
    def recv(self, tensor_name,
             forward_minibatch_id,
             backward_minibatch_id,
             backward=False, stopped=False):
        #print("recv", tensor_name, "backward",backward,"stoppped",stopped)
        if tensor_name in self._recv_futures:
            future = self._recv_futures[tensor_name]
            # print("[R] Async: ", tensor_name, backward, id(future))
            result = future.result()
        else:
            # print("[R] Sync: ", tensor_name, backward)
            result = self._recv_local_task(tensor_name, backward)

        if not stopped:

            future = self._thread_pool_recv.submit(
                self._recv_local_task, tensor_name, backward
            )
            # print("[R] Launch Async: ", tensor_name, backward, id(future))
            self._recv_futures[tensor_name] = future
        # print(result.requires_grad)
        return result

    def wait(self):
        with self._wait_cond:
            self._wait_cond.wait_for(self._fut_queue.empty)
        self._recv_buffers.clear()
        self._recv_futures.clear()

