# Semantic Opinion Dynamics: Final Integration Branch

ECE 227 Final Project on government environmental regulation debates using LLM agents on social graphs.

## Scope

- Topic is fixed: **Government Environmental Regulations**
- Persona source is fixed: **`data/nodes.json`** (36 nodes)
- Supported graph types: `er`, `rgglr`
- Bot policy: if enabled, injected at **t=0**

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Single CLI

Only one command mode is supported:

```bash
python main.py run --graph {er|rgglr} --bot {off|on} --model {semantic|degroot|both} --seed 42
```

Canonical four runs:

```bash
python main.py run --graph er --bot off --model both --seed 42
python main.py run --graph er --bot on --model semantic --seed 42
python main.py run --graph rgglr --bot off --model both --seed 42
python main.py run --graph rgglr --bot on --model semantic --seed 42
```

Useful flags:

- `--steps`
- `--model`
- `--edge-prob` (ER only)
- `--radius --long-range-fraction --long-range-k` (RGGLR only)
- `--bot-prob`
- `--plot`
- `--no-log`

Note: DeGroot runs currently support `--bot off` only.

## Project Structure

```text
ECE227-Final-Project/
├── main.py
├── data/
│   ├── nodes.json
│   └── README.md
├── docs/
│   └── INTEGRATION_PLAN.md
├── outputs/
├── requirements.txt
├── .env.example
└── src/
    ├── config.py
    ├── network.py
    ├── graphs/
    │   ├── er.py
    │   └── rgg_long_range.py
    ├── simulation.py
    ├── intervention.py
    ├── measurement.py
    └── experiments/
        └── matrix.py
```
