from __future__ import annotations
import pickle
from typing import Any
import torch
import torch.distributed as dist


def send_bytes(dst: int, b: bytes) -> None:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
    length = torch.tensor([len(b)], dtype=torch.long)
    dist.send(length, dst=dst)
    if len(b) > 0:
        arr = torch.tensor(list(b), dtype=torch.uint8)
        dist.send(arr, dst=dst)


def recv_bytes(src: int) -> bytes:
    length = torch.zeros(1, dtype=torch.long)
    dist.recv(length, src=src)
    n = int(length.item())
    if n == 0:
        return b""
    buf = torch.empty(n, dtype=torch.uint8)
    dist.recv(buf, src=src)
    return bytes(buf.tolist())


def send_obj(dst: int, obj: Any) -> None:
    b = pickle.dumps(obj)
    send_bytes(dst, b)


def recv_obj(src: int) -> Any:
    b = recv_bytes(src)
    if not b:
        return None
    return pickle.loads(b)