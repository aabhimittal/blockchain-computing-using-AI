"""Pure-Python cryptographic primitives for NeuroChain.

Implements the secp256k1 elliptic curve with ECDSA sign/verify and RFC-6979
deterministic nonces, plus SHA-256 based hashing helpers and address
derivation. Written without third-party crypto dependencies so the whole
chain can be exercised in tests and CI without native builds.

This is educational-grade crypto: correct on the curve math, but not
constant-time. Do not use it to secure real funds.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Tuple

# ---------------------------------------------------------------------------
# secp256k1 domain parameters
# ---------------------------------------------------------------------------
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
A = 0
B = 7
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
G = (GX, GY)

Point = Tuple[int, int]  # affine point; None-as-(0,0) is treated as infinity
INF: Point = (0, 0)


def _inv(x: int, m: int) -> int:
    """Modular inverse via Fermat's little theorem (m is prime here)."""
    return pow(x, m - 2, m)


def point_add(p: Point, q: Point) -> Point:
    if p == INF:
        return q
    if q == INF:
        return p
    if p[0] == q[0] and (p[1] + q[1]) % P == 0:
        return INF
    if p == q:
        lam = (3 * p[0] * p[0] + A) * _inv(2 * p[1], P) % P
    else:
        lam = (q[1] - p[1]) * _inv((q[0] - p[0]) % P, P) % P
    x = (lam * lam - p[0] - q[0]) % P
    y = (lam * (p[0] - x) - p[1]) % P
    return (x, y)


def point_mul(k: int, p: Point = G) -> Point:
    """Double-and-add scalar multiplication."""
    result = INF
    addend = p
    k %= N
    while k:
        if k & 1:
            result = point_add(result, addend)
        addend = point_add(addend, addend)
        k >>= 1
    return result


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------
def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def double_sha256(data: bytes) -> bytes:
    return sha256(sha256(data))


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ripemd160(data: bytes) -> bytes:
    try:
        h = hashlib.new("ripemd160")
        h.update(data)
        return h.digest()
    except ValueError:
        # Some builds omit ripemd160; fall back to a truncated SHA-256.
        return sha256(data)[:20]


# ---------------------------------------------------------------------------
# Keys, addresses, signatures
# ---------------------------------------------------------------------------
def _bits2int(b: bytes) -> int:
    return int.from_bytes(b, "big")


def _rfc6979_nonce(priv: int, msg_hash: bytes) -> int:
    """Deterministic nonce (RFC 6979) so signatures are reproducible in tests."""
    x = priv.to_bytes(32, "big")
    h1 = msg_hash
    v = b"\x01" * 32
    k = b"\x00" * 32
    k = hmac.new(k, v + b"\x00" + x + h1, hashlib.sha256).digest()
    v = hmac.new(k, v, hashlib.sha256).digest()
    k = hmac.new(k, v + b"\x01" + x + h1, hashlib.sha256).digest()
    v = hmac.new(k, v, hashlib.sha256).digest()
    while True:
        v = hmac.new(k, v, hashlib.sha256).digest()
        cand = _bits2int(v)
        if 1 <= cand < N:
            return cand
        k = hmac.new(k, v + b"\x00", hashlib.sha256).digest()
        v = hmac.new(k, v, hashlib.sha256).digest()


def generate_private_key() -> int:
    while True:
        candidate = _bits2int(os.urandom(32))
        if 1 <= candidate < N:
            return candidate


def private_to_public(priv: int) -> Point:
    return point_mul(priv, G)


def public_to_bytes(pub: Point) -> bytes:
    """Compressed SEC1 encoding (33 bytes)."""
    prefix = b"\x02" if pub[1] % 2 == 0 else b"\x03"
    return prefix + pub[0].to_bytes(32, "big")


def public_to_address(pub: Point) -> str:
    """Bitcoin-style address body: RIPEMD160(SHA256(pubkey)), hex, 0x-prefixed."""
    digest = _ripemd160(sha256(public_to_bytes(pub)))
    return "0x" + digest.hex()


def sign(priv: int, message: bytes) -> Tuple[int, int]:
    z = _bits2int(sha256(message))
    while True:
        k = _rfc6979_nonce(priv, sha256(message))
        x1 = point_mul(k, G)[0]
        r = x1 % N
        if r == 0:
            continue
        s = (_inv(k, N) * (z + r * priv)) % N
        if s == 0:
            continue
        # Low-s normalization (BIP-62) to avoid malleability.
        if s > N // 2:
            s = N - s
        return (r, s)


def verify(pub: Point, message: bytes, signature: Tuple[int, int]) -> bool:
    r, s = signature
    if not (1 <= r < N and 1 <= s < N):
        return False
    z = _bits2int(sha256(message))
    w = _inv(s, N)
    u1 = (z * w) % N
    u2 = (r * w) % N
    point = point_add(point_mul(u1, G), point_mul(u2, pub))
    if point == INF:
        return False
    return (point[0] % N) == r


@dataclass(frozen=True)
class KeyPair:
    """A wallet keypair with convenience accessors."""

    private_key: int

    @classmethod
    def generate(cls) -> "KeyPair":
        return cls(generate_private_key())

    @property
    def public_key(self) -> Point:
        return private_to_public(self.private_key)

    @property
    def address(self) -> str:
        return public_to_address(self.public_key)

    @property
    def public_key_hex(self) -> str:
        return public_to_bytes(self.public_key).hex()

    def sign(self, message: bytes) -> Tuple[int, int]:
        return sign(self.private_key, message)


def decompress_public_hex(public_key_hex: str) -> Point:
    """Recover the full curve point from a compressed-hex public key."""
    raw = bytes.fromhex(public_key_hex)
    prefix, x = raw[0], _bits2int(raw[1:33])
    y_sq = (pow(x, 3, P) + B) % P
    y = pow(y_sq, (P + 1) // 4, P)
    if y % 2 != (0 if prefix == 2 else 1):
        y = P - y
    return (x, y)


def address_from_public_hex(public_key_hex: str) -> str:
    return public_to_address(decompress_public_hex(public_key_hex))


def verify_hex(public_key_hex: str, message: bytes, signature: Tuple[int, int]) -> bool:
    """Verify given a compressed-hex public key (decompress on the curve)."""
    return verify(decompress_public_hex(public_key_hex), message, signature)
