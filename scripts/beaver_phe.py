import os
import csv
import random
import sys
from typing import List

import torch.distributed as dist

from scripts.communicator import send_obj, recv_obj
from scripts.paillier_crypto import *
from config import MPC_MODULO, PAILLIER_KEY_SIZE

def modq(x: int) -> int:
    return int(x % MPC_MODULO)


def main(rank: int, world_size: int):
    T_env = os.environ.get("BEAVER_TRICKS_COUNT", os.environ.get("BEAVER_TRIPLES", "10"))
    try:
        T = int(T_env)
    except Exception:
        T = 10

    outdir = os.environ.get("OUT_DIR", "/app/output") or "./output"

    if not dist.is_initialized():
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", init_method=f"tcp://{addr}:{port}", rank=rank, world_size=world_size)

    peer = 1 - rank

    if rank == 0:
        pubkey, privkey = generate_keypair(n_length=int(PAILLIER_KEY_SIZE))
        send_obj(peer, ("PUB_N", int(pubkey.n)))
    else:
        tag, n = recv_obj(0)
        assert tag == "PUB_N"
        pubkey = pubkey_from_n(n)
        privkey = None

    a_local: List[int] = [random.randrange(0, MPC_MODULO) for _ in range(T)]
    b_local: List[int] = [random.randrange(0, MPC_MODULO) for _ in range(T)]
    out_triples: List[Tuple[int, int, int]] = []

    if rank == 0:
        for i in range(T):
            enc_a = pubkey.encrypt(int(a_local[i]))
            enc_b = pubkey.encrypt(int(b_local[i]))
            send_obj(peer, ("ENC_AB", enc_to_primitive(enc_a), enc_to_primitive(enc_b)))

            tag, enc_masked_prim = recv_obj(peer)
            assert tag == "ENC_MASKED"
            enc_masked = primitive_to_enc(pubkey, enc_masked_prim)
            decrypted = privkey.decrypt(enc_masked) % MPC_MODULO
            c0 = modq((a_local[i] * b_local[i]) + decrypted)
            out_triples.append((a_local[i], b_local[i], c0))

        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, "p1.csv")
        with open(outpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b", "c"])
            for a, b, c in out_triples:
                writer.writerow([a, b, c])
        print(f"[rank 0] wrote {outpath}", file=sys.stderr)

    else:
        for i in range(T):
            tag, enc_a_prim, enc_b_prim = recv_obj(0)
            assert tag == "ENC_AB"
            enc_a = primitive_to_enc(pubkey, enc_a_prim)
            enc_b = primitive_to_enc(pubkey, enc_b_prim)

            a2 = int(a_local[i])
            b2 = int(b_local[i])
            r2 = random.randrange(0, MPC_MODULO)

            enc_sum = (enc_a * b2) + (enc_b * a2) + pubkey.encrypt((a2 * b2 + r2) % MPC_MODULO)
            send_obj(0, ("ENC_MASKED", enc_to_primitive(enc_sum)))

            c2 = modq(-r2)
            out_triples.append((a2, b2, c2))

        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, "p2.csv")
        with open(outpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b", "c"])
            for a, b, c in out_triples:
                writer.writerow([a, b, c])
        print(f"[rank 1] wrote {outpath}", file=sys.stderr)
    dist.barrier()