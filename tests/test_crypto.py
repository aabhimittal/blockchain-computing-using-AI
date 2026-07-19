from neurochain.core.crypto import (
    KeyPair,
    address_from_public_hex,
    point_mul,
    G,
    N,
    sign,
    verify,
    verify_hex,
)


def test_keypair_address_deterministic_from_pubkey():
    kp = KeyPair.generate()
    assert kp.address == address_from_public_hex(kp.public_key_hex)
    assert kp.address.startswith("0x")


def test_sign_and_verify_roundtrip():
    kp = KeyPair.generate()
    msg = b"transfer 10 coins"
    sig = kp.sign(msg)
    assert verify(kp.public_key, msg, sig)
    assert verify_hex(kp.public_key_hex, msg, sig)


def test_signature_rejects_tampered_message():
    kp = KeyPair.generate()
    sig = kp.sign(b"pay alice")
    assert not verify(kp.public_key, b"pay bob", sig)


def test_signature_is_deterministic_rfc6979():
    kp = KeyPair(private_key=123456789)
    assert kp.sign(b"same") == kp.sign(b"same")


def test_low_s_normalization():
    _, s = sign(987654321, b"hello")
    assert s <= N // 2


def test_generator_order():
    # k*G for k=N should be the point at infinity representation
    assert point_mul(N, G) == (0, 0)
