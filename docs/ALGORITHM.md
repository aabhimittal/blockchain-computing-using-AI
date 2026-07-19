# Proof-of-Useful-Intelligence (PoUI)

PoUI is a consensus mechanism that replaces wasteful hash-grinding with
*useful* machine-learning computation, while preserving the security property
that makes Proof-of-Work sound: **asymmetry** — a proof is expensive to
produce but cheap to verify.

## Motivation

Bitcoin-style PoW burns energy solving SHA-256 pre-image puzzles whose only
value is being hard. PoUI keeps the puzzle hard, but makes the work produce a
by-product with external value: trained neural-network models bound to
on-chain entropy. The scheme is a concrete instance of *Proof-of-Useful-Work*
with a verification path that does **not** require re-training.

## The per-block challenge

For a candidate block with predecessor hash `H_prev` and transaction Merkle
root `M`, the challenge seed is:

```
seed = SHA256(H_prev || M)   # first 8 bytes as an integer
```

The seed deterministically generates a synthetic regression task
`D = {(x_i, y_i)}` with `x_i ∈ R^4` and a non-linear target
`y_i = sin(x_i · w*)` for a seed-derived `w*`. Because the seed depends on
`H_prev` and `M`, the task cannot be precomputed before the block's contents
and parent are fixed — this binds the work to the chain and to the specific
transaction set.

## Sealing (mining)

The miner trains a small multilayer perceptron (`4 → 16 → 16 → 1`) with Adam
to minimize MSE on `D`. It must drive the loss below a difficulty-scaled
threshold:

```
threshold(d, r) = BASE · DECAY^d · (1 + 0.15 · r)
BASE = 0.15,  DECAY = 0.7
```

where `d` is the block difficulty and `r ∈ [0,1]` is the miner's reputation
(higher reputation earns a small leniency, coupling reputation to consensus).
Because the target is smooth and the network can fit it arbitrarily well given
enough gradient steps, a lower threshold simply demands **more training
compute** — that is the tunable "work".

The miner then finds a small binding nonce so that

```
SHA256(weights_digest : seed : nonce)
```

has the required leading-zero hex prefix. This anchors the useful-work proof
with a light proof-of-work (anti-grinding on model randomness and a
tie-break for otherwise equal work).

The block's `proof` records: the challenge seed, the achieved loss, the
threshold, the (rounded) trained weights, their digest, the binding hash, and
a normalized `work_quality ∈ [0,1]`.

## Verification

A verifier re-derives everything from public block data and performs **one
forward pass**:

1. Recompute `seed` from `H_prev || M`; reject if it disagrees with the proof.
2. Recompute `SHA256` of the submitted weights; reject on digest mismatch.
3. Recompute the binding hash from `(digest, seed, nonce)`; reject if it
   doesn't match or lacks the required prefix.
4. Load the submitted weights, evaluate MSE on the deterministic dataset, and
   reject unless `loss ≤ threshold`.

No training happens in verification — the cost is a single matrix multiply,
giving PoW-like asymmetry.

## Fork choice

Instead of "longest chain" or raw cumulative hash, PoUI selects the chain with
the greatest **cumulative useful work**:

```
score(chain) = Σ_blocks  difficulty · (1 + work_quality)
```

This rewards chains that are both harder and produced higher-quality trained
models. The `difficulty` term keeps it monotone in chain length (heaviest
chain), and `work_quality` breaks ties toward more useful intelligence.

## Security notes & limitations

- **Not production crypto.** The bundled secp256k1/ECDSA is correct but not
  constant-time; the ML challenge is a research prototype.
- **Grindability.** A miner could grind transaction ordering to shift `M` and
  thus `seed`; the binding nonce and reputation weighting raise the cost, but
  a deployed system would add stake and VDF-style delay.
- **Outsourcing.** Because verification is a forward pass, the useful task
  should be chosen so that its solution has genuine external value (the
  synthetic task here is a stand-in for domain datasets — surrogate models,
  hyper-parameter search, etc.).
