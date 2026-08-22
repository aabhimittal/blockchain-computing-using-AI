# NeuroChain — Blockchain Computing with AI-Based Consensus

> A from-scratch blockchain whose consensus, difficulty control, transaction
> screening and fork choice are all driven by machine learning — implemented
> in pure Python + NumPy, with no heavyweight dependencies.

NeuroChain explores a simple question: **what if the energy a blockchain
spends securing itself produced something useful?** Instead of SHA-256
hash-grinding, miners train small neural networks on puzzles bound to the
chain's own state. The result is a novel consensus algorithm —
**Proof-of-Useful-Intelligence (PoUI)** — wrapped in a complete, tested
ledger with four cooperating AI subsystems.

## Highlights

| Component | What's novel | Module |
|---|---|---|
| **Proof-of-Useful-Intelligence** | Consensus by *training a model* to a difficulty-scaled loss; PoW-style asymmetry (train-hard / verify-cheap) | `consensus/proof_of_useful_intelligence.py` |
| **RL adaptive difficulty** | Difficulty retargeting *learned online* via tabular Q-learning, not a fixed arithmetic rule | `ai/adaptive_difficulty.py` |
| **Autoencoder fraud detection** | Unsupervised anomaly scoring gates the mempool — no labelled fraud data needed | `ai/fraud_detection.py` |
| **Validator reputation** | EWMA reputation rewards useful work, slashes bad blocks, and weights leader selection | `ai/validator_reputation.py` |
| **Useful-work fork choice** | Chains win by cumulative *useful work + quality*, not just length | `network/node.py` |
| **Pure-Python crypto** | secp256k1 ECDSA (RFC-6979) with zero third-party crypto deps | `core/crypto.py` |

## Install

```bash
git clone https://github.com/aabhimittal/blockchain-computing-using-ai.git
cd blockchain-computing-using-ai
pip install -e .            # core (numpy only)
pip install -e '.[api]'     # + FastAPI REST server (optional)
pip install -e '.[dev]'     # + pytest
```

Requires Python ≥ 3.9.

## Quickstart

```python
from neurochain import Node, Wallet

miner, alice = Wallet(), Wallet()
node = Node(address=miner.address, initial_difficulty=2)

node.mine_block()                       # mine an (empty) block -> block reward
tx = miner.create_transaction(alice.address, 30,
                              nonce=node.blockchain.nonce_of(miner.address), fee=1)
node.submit(tx)                         # goes through AI fraud screening
node.mine_block()                       # confirm it

print(node.blockchain.balance_of(alice.address))   # 30.0
print(node.blockchain.is_valid_chain())            # True
```

### Command line

```bash
neurochain wallet                  # generate a keypair
neurochain simulate --blocks 8     # mine locally and print chain/AI stats
neurochain demo                    # full end-to-end walkthrough
neurochain serve --port 8000       # REST API (requires [api] extras)
```

### Examples

```bash
python -m examples.demo              # wallets, transfers, fraud, reputation, integrity
python -m examples.simulate_network  # two nodes + useful-work fork choice
python -m examples.advanced_features # NeuroVM, fee market, federated learning, SPV
```

## Advanced subsystems

Beyond the core ledger and consensus, NeuroChain ships four self-contained modules
that extend it to more real-world use cases, each hardened against industrial edge
cases (see `tests/`):

| Module | What it adds | Use case |
| --- | --- | --- |
| `neurochain.vm` — **NeuroVM** | Deterministic, gas-metered stack machine with reverts, 256-bit wrapping arithmetic, jump-dest validation, and a standard-contract library (timelock, multisig, escrow, HTLC). | Programmable transactions / smart contracts with bounded, replayable execution. |
| `neurochain.economics` — **FeeMarket** | EIP-1559-style base-fee controller: fees rise/fall toward a target block occupancy, with a floor and per-block change cap; priority tips bid for inclusion. | Congestion pricing and predictable fees. |
| `neurochain.ai.federated` | Byzantine-robust aggregation of on-chain model updates — coordinate-wise trimmed mean, Krum, reputation-weighted mean — with NaN/Inf dropping and L2 norm clipping. | Decentralised (federated) model training as useful work, poison-resistant. |
| `neurochain.core.light_client` | SPV Merkle inclusion proofs built from a block and verified against a trusted header root. | Light clients that confirm payments without downloading full blocks. |

Edge cases exercised include VM out-of-gas / division-by-zero / stack under-and-overflow / invalid jumps / storage rollback on revert, fee-market clamping of malformed block sizes and floor enforcement, federated dimension mismatches and all-Byzantine batches, and SPV tampered-root and odd-leaf-count trees.

## How Proof-of-Useful-Intelligence works

For each block, a learning challenge is derived deterministically from the
parent hash and the transaction Merkle root, so it can't be precomputed:

```
seed      = SHA256(previous_hash || merkle_root)
challenge = synthetic regression task D(seed)   # y = sin(x · w*)
```

To **seal** a block, the miner trains an MLP until its loss on `D` falls below
a difficulty-scaled threshold, then finds a small binding nonce
(`SHA256(weights_digest : seed : nonce)` with a leading-zero prefix). To
**verify**, any node re-derives `D`, loads the submitted weights, and checks
the loss with a *single forward pass* — no re-training. Harder difficulty ⇒
lower threshold ⇒ more gradient-descent compute, so "difficulty" maps to real,
tunable, *useful* work.

Full details and security discussion: **[docs/ALGORITHM.md](docs/ALGORITHM.md)**.
Architecture overview: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## REST API

```bash
neurochain serve --port 8000
```

| Method | Path | Description |
|---|---|---|
| GET  | `/` | node summary |
| GET  | `/chain` | full chain JSON |
| GET  | `/stats` | height, difficulty, useful-work score |
| GET  | `/balance/{address}` | balance + nonce |
| POST | `/wallet` | new demo wallet |
| POST | `/tx` | submit a signed transaction |
| POST | `/mine` | mine the mempool into a block |

## Testing

```bash
python -m pytest -q          # 31 tests: crypto, ledger, consensus, AI, node
```

CI runs the suite on Python 3.9 / 3.11 / 3.12 (see `.github/workflows/ci.yml`).

## Project layout

```
neurochain/
├── core/        crypto · transaction · merkle · block · blockchain · wallet · light_client
├── consensus/   base interface · proof_of_useful_intelligence
├── ai/          neural_net · adaptive_difficulty · fraud_detection · validator_reputation · federated
├── vm/          NeuroVM stack machine · standard contract templates
├── economics/   EIP-1559 fee market
├── network/     mempool · node
├── api/         FastAPI server (optional)
└── cli.py
examples/        demo · simulate_network · advanced_features
tests/           pytest suite
docs/            ALGORITHM.md · ARCHITECTURE.md
```

## Disclaimer

This is a research/education project. The cryptography and consensus are
implemented for clarity, not for securing real assets — do not use it in
production or to protect anything of value.

## License

MIT — see [LICENSE](LICENSE).
