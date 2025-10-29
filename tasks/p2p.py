import torch
import torch.distributed as dist
from . import task

@task("send")
def send(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    tensor = torch.zeros(1, dtype=torch.long)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank', rank, 'has data', tensor[0])
