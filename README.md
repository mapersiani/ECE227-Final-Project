# Semantic Opinion Dynamics: LLM Agents on Complex Networks

ECE 227 Final Project — Replacing classical "weighted average" opinion dynamics with **in-context learning** as the update rule.

## Overview

Classical opinion dynamics (e.g., DeGroot) model opinions as scalars in [0, 1] and update via weighted averaging. This fails to capture semantic nuance (framing, rhetoric, logical fallacies). This project models network nodes as **generative agents** that hold text-based beliefs and update them through **conversation** with neighbors, using an LLM (Ollama, run locally).

### Novelty

- **50 years of network science** assume weighted averaging for opinion updates
- **This project** replaces that with **in-context learning**: agents read neighbor opinions and produce updated opinions via LLM inference

### Example

In a polarized political network, a centrist might be persuaded by the neighbor with better rhetorical arguments—not by numerical averaging.

---

## Quick Start

```bash
# 1. Install Ollama and pull a model (see Setup below)
brew install ollama
brew services start ollama
ollama pull llama3.2:3b

# 2. Clone, create venv, install deps
git clone https://github.com/mapersiani/ECE227-Final-Project.git
cd ECE227-Final-Project
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Set up .env (copy .env.example to .env; defaults work if you skip)
cp .env.example .env   # Windows: copy .env.example .env

# 4. Run the comparison (semantic + DeGroot)
python main.py compare --steps 5 --plot
```

---

## Setup

### 1. Install Ollama (required for semantic simulation)

Ollama runs LLMs locally—no API keys, no rate limits.

**macOS**
```bash
brew install ollama
brew services start ollama   # starts Ollama and keeps it running
```

**Windows**  
Download from [ollama.com](https://ollama.com) and run the installer. Ollama runs in the system tray.

**Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve   # or configure as a systemd service
```

### 2. Pull a model

```bash
ollama pull llama3.2:3b
```

Alternative models: `phi3:mini`, `mistral:7b` (larger = slower but higher quality). Set in `.env` via `OLLAMA_MODEL=phi3:mini` if desired.

### 3. Python environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Set up .env

```bash
cp .env.example .env   # macOS/Linux
copy .env.example .env # Windows (Command Prompt)
```

Defaults work without editing. Optionally:
- `OLLAMA_MODEL` – change model (e.g. `phi3:mini`, `mistral:7b`)
- `HF_TOKEN` – add if you have a Hugging Face account (faster model downloads)

**Requirements:** Python 3.9+

---

## Usage

### Compare semantic vs DeGroot (recommended first run)

Runs both and plots them together:

```bash
python main.py compare --topic "AI Regulation" --steps 5 --plot
```

- **DeGroot** (orange): scalar consensus—variance decreases.
- **Semantic** (blue): LLM-based—may polarize or diverge.

Output: `outputs/semantic_vs_degroot.png`

### Semantic only (LLM-based)

```bash
python main.py semantic --topic "AI Regulation" --steps 5 --plot
```

Output: variance printed to terminal, `outputs/semantic_variance.png` if `--plot`

**Runtime:** ~5–10 min for 5 steps (20 agents × 5 steps = 100 LLM calls).

### DeGroot only (fast, no LLM)

```bash
python main.py degroot --steps 5 --plot
```

Runs in seconds. Output: `outputs/degroot_variance.png`

### Intervention study (disinformation bot)

```bash
python main.py intervention --topic "AI Regulation" --steps 5 --plot
```

Adds a bot that posts frequently; measures resilience. Output: `outputs/intervention_comparison.png`

---

## What You'll See

When running semantic or compare:

1. **Startup:** `Creating network...` → `Graph: nodes=20, edges=49` → `Creating agents...` → `Running semantic simulation...`
2. **SBERT loading** (first run): `Loading weights: 100%` and a BertModel report
3. **Long pause** (5–10 min): no progress bar—it's working
4. **Results:** variance per timestep printed, then `Saved outputs/...png`

---

## Project Structure

```
ECE227-Final-Project/
├── main.py              # CLI: semantic, degroot, compare, intervention
├── outputs/             # Plot output directory (created on first --plot)
├── requirements.txt
├── .env.example         # Copy to .env (cp .env.example .env)
├── README.md
└── src/
    ├── config.py        # Personas (left, center_left, center_right, right), topic
    ├── network.py       # SBM graph: 20 nodes, 4 blocks
    ├── agent.py         # Agent with persona and opinion text
    ├── llm_client.py    # Ollama API for opinion updates
    ├── simulation.py    # Discrete-time semantic simulation
    ├── measurement.py   # SBERT embeddings, semantic variance
    └── intervention.py  # Disinformation bot study
```

## Components

| Component | Description |
|-----------|-------------|
| **Network** | SBM: 20 nodes in 4 blocks (left, center left, center right, right) |
| **Agents** | Personas aligned to blocks (left, center_left, center_right, right) |
| **Simulation** | Each step: agents read neighbors' opinions → LLM generates updated opinion |
| **Measurement** | SBERT embeddings → semantic variance (distance from centroid) |
| **Intervention** | Bot node with high posting frequency |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `could not connect to ollama server` | Start Ollama: `ollama serve` or `brew services start ollama` |
| `model not found` | Pull a model: `ollama pull llama3.2:3b` |
| Different model | Copy `.env.example` to `.env`, set `OLLAMA_MODEL=phi3:mini` |
| Import errors | Ensure venv is activated and run `pip install -r requirements.txt` |
| Slow semantic run | Expected. Try `--steps 2` or `--steps 3` for faster runs |
| HF_TOKEN warning | Optional. Set `HF_TOKEN` in `.env` if you have a Hugging Face account for faster model downloads; omit for anonymous use |

---

## License

MIT
