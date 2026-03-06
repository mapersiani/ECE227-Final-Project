# Government Environmental Regulation Opinion Dynamics

This repository runs opinion-dynamics simulations on fixed persona nodes using two models:
- `semantic` (text updates from Ollama)
- `degroot` (numeric consensus baseline)

The canonical persona dataset is `data/nodes.json` and must contain 36 nodes.

## Scope

- Topic: `Government Environmental Regulations`
- Graphs: `er`, `rgglr`
- Node count: 36 (enforced at load/build time)
- Bot policy: if `--bot on`, bot is injected at `t=0`

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Ollama must be running for `semantic` runs.

## CLI

Single command mode:

```bash
python main.py run --graph {er|rgglr} --bot {off|on} --model {semantic|degroot|both} --seed 42
```

Examples:

```bash
# ER baseline comparison
python main.py run --graph er --bot off --model both --seed 42

# RGG+long-range baseline comparison
python main.py run --graph rgglr --bot off --model both --seed 42

# Intervention (semantic only)
python main.py run --graph er --bot on --model semantic --seed 42
python main.py run --graph rgglr --bot on --model semantic --seed 42
```

Notes:
- `--model degroot` and `--model both` currently support `--bot off` only.
- `--plot` saves variance and side-count figures in `outputs/`.
- `--no-log` disables JSONL interaction logs.

## Active Runtime Path

Primary execution path:
- `main.py`
- `src/config.py`
- `src/network.py`
- `src/graphs/er.py`
- `src/graphs/rgg_long_range.py`
- `src/simulation.py`
- `src/intervention.py`
- `src/measurement.py`
- `src/llm_client.py`
- `src/agent.py`

## Repository Layout

```text
ECE227-Final-Project/
├── main.py
├── data/
│   ├── nodes.json
│   └── README.md
├── outputs/
├── requirements.txt
├── .env.example
└── src/
    ├── agent.py
    ├── config.py
    ├── intervention.py
    ├── llm_client.py
    ├── measurement.py
    ├── network.py
    ├── simulation.py
    ├── graphs/
    │   ├── er.py
    │   └── rgg_long_range.py
```

## Outputs

Typical files in `outputs/`:
- `*_semantic_variance.png`
- `*_degroot_variance.png`
- `*_semantic_vs_degroot.png`
- `*_side_counts.png`
- `logs/run_*.jsonl`
