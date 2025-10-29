from typing import Tuple
from phe import paillier


def generate_keypair(n_length: int = 2048) -> tuple[paillier.PaillierPublicKey, paillier.PaillierPrivateKey]:
    return paillier.generate_paillier_keypair(n_length=n_length)


def pubkey_from_n(n: int) -> paillier.PaillierPublicKey:
    return paillier.PaillierPublicKey(n=n)


def enc_to_primitive(enc: paillier.EncryptedNumber) -> Tuple[int, int]:
    return (int(enc.ciphertext()), int(getattr(enc, "exponent", 0)))


def primitive_to_enc(pubkey: paillier.PaillierPublicKey, prim: Tuple[int, int]) -> paillier.EncryptedNumber:
    c, exp = prim
    return paillier.EncryptedNumber(public_key=pubkey, ciphertext=int(c), exponent=int(exp))