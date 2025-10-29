import torch
import torch.distributed as dist
import time
from . import task

@task("broadcast")
def broadcast(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group(list(range(world_size)))

    tensor_size = 4
    if rank == 0:
        tensor = torch.arange(tensor_size, dtype=torch.long)
    else:
        tensor = torch.zeros(tensor_size, dtype=torch.long)
    print(f"Rank {rank}: tensor before broadcast = {tensor}")

    dist.broadcast(tensor, src=0, group=group)
    print(f"Rank {rank}: tensor after broadcast = {tensor}")

    dist.destroy_process_group()

@task("reduce")
def reduce(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group(list(range(world_size)))

    tensor = torch.tensor([rank], dtype=torch.long)
    print(f"Rank {rank}: tensor before reduce = {tensor}")

    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=group)
    print(f"Rank {rank}: tensor after reduce = {tensor}")

    dist.destroy_process_group()

@task("all_reduce")
def all_reduce(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group(list(range(world_size)))

    tensor = torch.tensor([rank], dtype=torch.long)
    print(f"Rank {rank}: tensor before all_reduce = {tensor}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print(f"Rank {rank}: tensor after all_reduce = {tensor}")

    dist.destroy_process_group()
 
@task("all_gather")
def all_gather(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group(list(range(world_size)))

    send_tensor = torch.tensor([rank], dtype=torch.long)
    recv_list = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
    print(f"Rank {rank}: tensor before all_gather = {send_tensor}")

    dist.all_gather(recv_list, send_tensor, group=group)
    print(f"Rank {rank}: all_gather result = {recv_list}")

    dist.destroy_process_group()
