# Semantic Opinion Dynamics: LLM Agents on Complex Networks

ECE 227 Final Project — Replacing classical "weighted average" opinion dynamics with **in-context learning** as the update rule.

## Overview

Classical opinion dynamics (e.g., DeGroot) model opinions as scalars in [0, 1] and update via weighted averaging. This fails to capture semantic nuance (framing, rhetoric, logical fallacies). This project models network nodes as **generative agents** that hold text-based beliefs and update them through **conversation** with neighbors, using an LLM (Gemini).

### Novelty

- **50 years of network science** assume weighted averaging for opinion updates
- **This project** replaces that with **in-context learning**: agents read neighbor opinions and produce updated opinions via LLM inference

### Example

In a polarized political network, a centrist might be persuaded by the neighbor with better rhetorical arguments—not by numerical averaging.

## Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# API key for Gemini
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

## Project Structure

```
ECE227-Final-Project/
├── main.py              # CLI entry point
├── requirements.txt
├── .env.example
├── README.md
└── src/
    ├── config.py        # Personas (left, center_left, center_right, right), topic, API keys
    ├── network.py       # SBM graph: 20 nodes, 4 blocks
    ├── agent.py         # Agent with persona and opinion text
    ├── llm_client.py    # Gemini API for opinion updates
    ├── simulation.py    # Discrete-time semantic simulation
    ├── measurement.py   # SBERT embeddings, semantic variance
    └── intervention.py  # Disinformation bot study
```

## Usage

### 1. Semantic simulation (LLM-based)

```bash
python main.py semantic --topic "AI Regulation" --steps 5 --plot
```

### 2. DeGroot baseline

```bash
python main.py degroot --steps 5 --plot
```

### 3. Intervention study (disinformation bot)

```bash
python main.py intervention --steps 5 --plot
```

## Components

| Component | Description |
|-----------|-------------|
| **Network** | SBM: 20 nodes in 4 blocks (left, center left, center right, right) |
| **Agents** | Personas (tech optimist, skeptic, centrist, libertarian, pro-regulation) + initial opinion text |
| **Simulation** | Each step: agents read neighbors' opinions → LLM generates updated opinion |
| **Measurement** | SBERT embeddings → semantic variance (distance from centroid) |
| **Intervention** | Bot node with high posting frequency; compare topology resilience |

## Requirements

- Python 3.9+
- `GOOGLE_API_KEY` (Gemini API) for semantic simulation
- DeGroot and intervention plots work without API key only if using pre-computed embeddings; the semantic run requires API calls

## License

MIT
