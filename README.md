# Semantic Opinion Dynamics: LLM Agents on ER Graph + DeGroot Baseline

ECE 227 Final Project — Comparing classical DeGroot opinion dynamics with **semantic,
LLM-based** dynamics on the same Erdős–Rényi (ER) graph of personas.

## Overview

- **Personas**: environmental regulation personas in `nodes.json` (Left, Center Left, Center Right, Right)
- **Graph**: ER graph `G(n, p)` over these personas
- **Baselines**:
  - **DeGroot**: scalar opinions in \([0, 1]\), averaging over neighbors
  - **Semantic**: text opinions updated via a local LLM (Ollama) + SBERT embeddings

We measure **variance over time** for both models and compare convergence vs. polarization.

---

## Quick Start

```bash
# 1. Install Ollama and pull a model (for semantic runs)
brew install ollama
brew services start ollama
ollama pull llama3.2:3b

# 2. Clone, create venv, install deps
git clone https://github.com/mapersiani/ECE227-Final-Project.git
cd ECE227-Final-Project
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Set up .env (optional; defaults usually fine)
cp .env.example .env   # macOS/Linux
copy .env.example .env # Windows (Command Prompt)

# 4. Run comparison (semantic + DeGroot)
python main.py compare --steps 5 --edge-prob 0.15 --plot
```

---

## Setup Details

### Python environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Requirements:** Python 3.9+

### nodes.json

`nodes.json` contains persona objects with fields:

- `name`: e.g. `left_ej_clinic_attorney`
- `prompt`: persona description (used as LLM persona prompt)
- `style`: rhetorical style
- `initial`: initial text opinion

These personas define the **node set**. DeGroot uses scalar opinions; the semantic model
uses the `prompt` and `initial` text.

### .env

```bash
cp .env.example .env   # macOS/Linux
copy .env.example .env # Windows (Command Prompt)
```

Defaults work without editing. Optionally:

- `OLLAMA_MODEL` – change Ollama model (e.g. `phi3:mini`, `mistral:7b`)
- `HF_TOKEN` – add if you have a Hugging Face account (faster SBERT downloads)

---

## Usage

### Semantic simulation (LLM-based)

```bash
python main.py semantic --steps 5 --edge-prob 0.15 --plot
```

Output:

- Semantic variance printed per timestep
- Plot saved to `outputs/semantic_variance.png` if `--plot`
- Classification plot saved to `outputs/semantic_side_counts.png` if `--plot`
- Interaction log written to `outputs/logs/semantic_<timestamp>_p<...>_seed<...>.jsonl` (disable with `--no-log`)

### DeGroot baseline (scalar, no LLM)

```bash
python main.py degroot --steps 5 --edge-prob 0.15 --plot
```

Output:

- Scalar variance printed per timestep
- `outputs/degroot_variance.png`

### Export the graph to Gephi

Run any mode with `--export-gephi` to write:

- `outputs/gephi/er_p<...>_seed<...>.gexf`
- `outputs/gephi/er_p<...>_seed<...>.graphml`

Open the `.gexf` in Gephi (recommended).

### Compare semantic vs DeGroot

```bash
python main.py compare --steps 5 --edge-prob 0.15 --plot
```

Output:

- Both variance series printed
- `outputs/semantic_vs_degroot.png` (DeGroot = orange, Semantic = blue)

### Intervention study (disinformation bot)

```bash
python main.py intervention --steps 5 --edge-prob 0.15 --plot
```

Output:

- Semantic variance with bot over time
- `outputs/intervention_comparison.png`
- `outputs/intervention_side_counts.png`
- Interaction log written to `outputs/logs/intervention_<timestamp>_p<...>_seed<...>.jsonl` (disable with `--no-log`)

---

## Project Structure

```text
ECE227-Final-Project/
├── main.py                  # CLI: semantic, degroot, compare, intervention
├── outputs/                 # All experiment outputs
│   ├── semantic_vs_degroot.png          # Comparison plot
│   ├── semantic_side_counts.png         # Semantic side-counts over time
│   ├── logs/                            # JSONL interaction logs
│   │   └── semantic_*.jsonl, compare_*.jsonl, intervention_*.jsonl
│   └── gephi/                           # Graph exports for Gephi
│       ├── er_p*_seed*.gexf
│       └── er_p*_seed*.graphml
├── requirements.txt
├── .env.example         # Copy to .env
├── nodes.json           # Personas (Left, Center Left, Center Right, Right)
├── README.md
└── src/
    ├── config.py        # Global defaults (fixed topic, steps, Ollama settings)
    ├── network.py       # ER graph over personas + DeGroot
    ├── agent.py         # Persona agent dataclass
    ├── llm_client.py    # Ollama API for opinion updates
    ├── simulation.py    # Semantic (LLM) simulation on ER graph
    ├── measurement.py   # SBERT embeddings, semantic variance
    └── intervention.py  # Disinformation bot study
```

---

## License

MIT
