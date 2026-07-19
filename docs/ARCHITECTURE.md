# Architecture

NeuroChain is organized into four layers plus thin interface layers (CLI/API).

```
                    ┌──────────────────────────────┐
   CLI / REST API   │  neurochain.cli / api.server │
                    └───────────────┬──────────────┘
                                    │
                    ┌───────────────▼──────────────┐
   Network layer    │  Node  ·  Mempool            │  in-process peer,
                    │  (fork choice, propagation)  │  AI-gated tx pool
                    └───────────────┬──────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────▼────────┐        ┌─────────▼──────────┐      ┌──────────▼─────────┐
│ Core ledger    │        │ Consensus (PoUI)   │      │ AI subsystems      │
│ blockchain     │◄──────►│ seal / validate    │◄────►│ neural_net (MLP)   │
│ block, tx      │        │ challenge + proof  │      │ adaptive_difficulty│
│ merkle, crypto │        │                    │      │ fraud_detection    │
│ wallet         │        │                    │      │ validator_reputation│
└────────────────┘        └────────────────────┘      └────────────────────┘
```

## Modules

### `neurochain.core`
- **crypto.py** — pure-Python secp256k1 curve math, ECDSA sign/verify with
  RFC-6979 deterministic nonces, address derivation, hashing helpers.
- **transaction.py** — signed value transfers; nonce-based replay protection;
  coinbase transactions for block rewards.
- **merkle.py** — Merkle root + inclusion proofs over transaction ids.
- **block.py** — block header, deterministic hashing, (de)serialization.
- **blockchain.py** — the ledger: account balances/nonces, full block
  validation (structure, consensus proof, signatures, balances, nonces),
  atomic application, chain-integrity checks.
- **wallet.py** — keypair management + signed-transaction construction.

### `neurochain.consensus`
- **base.py** — `ConsensusEngine` interface (`seal`, `validate`).
- **proof_of_useful_intelligence.py** — the PoUI engine (see
  [ALGORITHM.md](ALGORITHM.md)). The ledger is consensus-agnostic; any engine
  implementing the interface can be swapped in.

### `neurochain.ai`
- **neural_net.py** — dependency-light NumPy MLP (He init, ReLU, Adam,
  mini-batch SGD). Reused by consensus, fraud detection and reputation.
- **adaptive_difficulty.py** — tabular Q-learning controller that *learns* a
  difficulty-retargeting policy online instead of a fixed arithmetic rule.
- **fraud_detection.py** — unsupervised autoencoder anomaly detector; scores
  transactions by reconstruction error against the learned "normal" manifold.
- **validator_reputation.py** — EWMA reputation ledger; rewards consistent
  useful work, slashes invalid submissions, weights leader selection.

### `neurochain.network`
- **mempool.py** — pending-transaction pool with optional AI fraud gating and
  fee-priority selection.
- **node.py** — wires ledger + mempool + consensus + AI together; exposes
  submit / mine / import / reconcile, including the useful-work fork-choice
  rule.

## Data flow: mining a block

1. `Node.mine_block` selects fee-priority transactions from the mempool.
2. `Blockchain.create_candidate` prepends a coinbase reward transaction.
3. `ProofOfUsefulIntelligence.seal` derives the challenge, trains an MLP until
   the loss meets the difficulty threshold, and attaches the proof + binding
   nonce.
4. `Blockchain.add_block` re-validates everything and applies balances/nonces.
5. The RL difficulty controller retunes difficulty from the observed block
   time; the reputation ledger updates the miner's score.

## Design principles

- **Pluggable consensus** behind a narrow interface.
- **No heavy ML frameworks** — everything runs on NumPy so CI is fast and the
  math is transparent.
- **Deterministic where it matters** — RFC-6979 signatures, seeded RNGs, and a
  deterministic difficulty RNG keep consensus reproducible across nodes.
