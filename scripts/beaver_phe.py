import argparse
import csv
import os
import random
import torch
import torch.distributed as dist
from phe import paillier

from config import PAILLIER_KEY_SIZE, MPC_MODULO


def send_bytes(dst: int, b: bytes):
    length = torch.tensor([len(b)], dtype=torch.long)
    dist.send(length, dst=dst)
    if len(b) > 0:
        arr = torch.frombuffer(b, dtype=torch.uint8).clone()
        dist.send(arr, dst=dst)


def recv_bytes(src: int) -> bytes:
    length = torch.tensor([0], dtype=torch.long)
    dist.recv(length, src=src)
    n = int(length.item())
    if n == 0:
        return b""
    arr = torch.empty(n, dtype=torch.uint8)
    dist.recv(arr, src=src)
    return bytes(arr.tolist())


def send_obj(dst: int, obj):
    import pickle
    b = pickle.dumps(obj)
    send_bytes(dst, b)


def recv_obj(src: int):
    import pickle
    b = recv_bytes(src)
    return pickle.loads(b)


def init_dist_from_env(rank: int, world_size: int):
    if not dist.is_initialized():
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{addr}:{port}",
            rank=rank,
            world_size=world_size,
        )


def modq(x: int) -> int:
    return int(x % MPC_MODULO)


def enc_to_primitive(enc) -> tuple[int, int]:
    return (int(enc.ciphertext()), int(getattr(enc, "exponent", 0)))


def primitive_to_enc(pubkey: paillier.PaillierPublicKey, prim: tuple[int, int]):
    c, exp = prim
    return paillier.EncryptedNumber(public_key=pubkey, ciphertext=int(c), exponent=int(exp))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--triples", type=int, default=100, help="Number of Beaver triples")
    parser.add_argument("--outdir", type=str, default="/output", help="Directory to save p1.csv / p2.csv (host-mounted)")
    args = parser.parse_args()

    rank = args.rank
    world_size = args.world_size
    T = args.triples
    outdir = args.outdir

    assert world_size == 2, "This script implements a 2-party protocol (world_size must be 2)"

    init_dist_from_env(rank, world_size)
    peer = 1 - rank

    if rank == 0:
        pubkey, privkey = paillier.generate_paillier_keypair(n_length=PAILLIER_KEY_SIZE)
        send_obj(peer, ("PUB_N", int(pubkey.n)))
    else:
        tag, n = recv_obj(0)
        assert tag == "PUB_N"
        pubkey = paillier.PaillierPublicKey(n)
        privkey = None

    a_local = [random.randrange(0, MPC_MODULO) for _ in range(T)]
    b_local = [random.randrange(0, MPC_MODULO) for _ in range(T)]

    if rank == 0:
        enc_a_list = [enc_to_primitive(pubkey.encrypt(int(x))) for x in a_local]
        enc_b_list = [enc_to_primitive(pubkey.encrypt(int(x))) for x in b_local]
        send_obj(peer, ("ENC_AB_LISTS", enc_a_list, enc_b_list))

        tag, enc_cross_minus_r2_prims = recv_obj(peer)
        assert tag == "ENC_CROSS_MINUS_R2"
        enc_cross_minus_r2 = [primitive_to_enc(pubkey, prim) for prim in enc_cross_minus_r2_prims]
        cross_minus_r2 = [privkey.decrypt(enc) % MPC_MODULO for enc in enc_cross_minus_r2]
        cross_share_0 = [modq(x) for x in cross_minus_r2]

        c_shares = []
        for i in range(T):
            local_term = (a_local[i] * b_local[i]) % MPC_MODULO
            c0 = modq(local_term + cross_share_0[i])
            c_shares.append(c0)

        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, "p1.csv")
        with open(outpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b", "c"])
            for i in range(T):
                writer.writerow([a_local[i], b_local[i], c_shares[i]])
        print(f"[rank 0] saved {outpath}")

    else:
        tag, enc_a_prims, enc_b_prims = recv_obj(0)
        assert tag == "ENC_AB_LISTS"
        enc_a_list = [primitive_to_enc(pubkey, prim) for prim in enc_a_prims]
        enc_b_list = [primitive_to_enc(pubkey, prim) for prim in enc_b_prims]

        enc_cross_minus_r2_prims = []
        r2_list = []
        for i in range(T):
            enc_a1 = enc_a_list[i]
            enc_b1 = enc_b_list[i]
            a2 = int(a_local[i])
            b2 = int(b_local[i])
            enc_a1_b2 = enc_a1 * b2
            enc_b1_a2 = enc_b1 * a2
            enc_sum = enc_a1_b2 + enc_b1_a2
            r2 = random.randrange(0, MPC_MODULO)
            enc_r2 = pubkey.encrypt(-int(r2))
            enc_masked = enc_sum + enc_r2
            enc_cross_minus_r2_prims.append(enc_to_primitive(enc_masked))
            r2_list.append(r2)

        send_obj(0, ("ENC_CROSS_MINUS_R2", enc_cross_minus_r2_prims))

        c_shares = []
        for i in range(T):
            local_term = (a_local[i] * b_local[i]) % MPC_MODULO
            c2 = modq(local_term + r2_list[i])
            c_shares.append(c2)

        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, "p2.csv")
        with open(outpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b", "c"])
            for i in range(T):
                writer.writerow([a_local[i], b_local[i], c_shares[i]])
        print(f"[rank 1] saved {outpath}")

    dist.barrier()


if __name__ == "__main__":
    main()