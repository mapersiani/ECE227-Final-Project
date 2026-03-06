# Government Environmental Regulation Opinion Dynamics

This project simulates opinion dynamics on a fixed 36-persona population using:
- `degroot` baseline (numeric consensus)
- `semantic` model (LLM-updated text opinions)

Personas and initial opinions come from [`data/nodes.json`](data/nodes.json).

## Canonical Experiment Design

To avoid drift/inconsistency, core settings are hard-coded in [`src/config.py`](src/config.py):
- Topic: `Government Environmental Regulations`
- Graphs: `er`, `rgglr`
- Node count: `36`
- Canonical seeds: `11, 23, 42`
- Steps: `10`
- ER edge probability: `0.15`
- RGGLR params: radius `0.30`, long-range fraction `0.30`, long-range k `2`
- Bot injection step: `t=0`
- Log mode: compact per-step summary (`summary`)
- Prompt budget: max `6` neighbor opinions, max `320` chars per neighbor (for faster run. Can be changed)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For semantic runs, start Ollama and ensure model `llama3.2:3b` is available:

```bash
ollama serve
ollama pull llama3.2:3b
```

## CLI

```bash
python main.py --help
```

Two modes are supported:
- `run`: run one condition
- `matrix`: run the full canonical matrix

### Run One Condition

```bash
python main.py run --graph {er|rgglr} --bot {off|on} --model {semantic|degroot|both} --seed {11|23|42}
```

Examples:

```bash
# ER: DeGroot + semantic (no bot)
python main.py run --graph er --bot off --model both --seed 42

# RGGLR: semantic with bot intervention
python main.py run --graph rgglr --bot on --model semantic --seed 42
```

Notes:
- `--model degroot` and `--model both` support `--bot off` only.
- A dedicated output folder is created automatically for every run.
- `--no-log` disables per-run compact step summaries.

### Run Full Canonical Matrix

```bash
python main.py matrix
```

This executes, for each seed in `11,23,42` and each graph in `er,rgglr`:
- `degroot` with bot off
- `semantic` with bot off
- `semantic` with bot on

Optional flags:

```bash
python main.py matrix --show-progress --log-runs
python main.py matrix --out outputs/my_matrix_copy.csv
```

## Outputs

### `run` mode
Each run creates a folder:

- `outputs/run_<graph>_<model>_bot-<off|on>_seed-<seed>_<timestamp>/`

Typical contents:
- `network_topology.png` (graph structure view)
- `opinion_drift_network.png` (semantic runs; node color = opinion drift from initial)
- `semantic_variance.png` (semantic runs)
- `degroot_variance.png` (degroot runs)
- `semantic_vs_degroot.png` (model=`both`)
- `side_counts.png` (semantic runs)
- `timeseries.csv` (per-timestep metrics)
- `run_summary.json` (config + graph metrics + final variance stats)
- `logs/step_summary.jsonl` (semantic runs, unless `--no-log`)
  - Contains: one metadata row, one summary row per step, one final run summary row

### `matrix` mode
Each matrix invocation creates a folder:

- `outputs/matrix_er-rgglr_seeds-3_steps-10_<timestamp>/`

Typical contents:
- `matrix_results.csv` (all per-timestep rows)
- `condition_variance_trajectories.png` (mean variance trajectories by condition)
- `final_step_variance_bars.png` (final-step means with std error bars)
- `variance_heatmap.png` (condition x timestep variance heatmap)
- `bot_effect_over_time.png` (semantic bot-on minus bot-off)
- `semantic_degroot_gap.png` (semantic-off minus degroot-off)
- `semantic_side_entropy.png` (semantic side-mix entropy over time)
- `matrix_summary.json` (run metadata + final-step summary)
- `logs/*_summary.jsonl` (optional, with `--log-runs`)

If `--out` is provided, an extra copy of `matrix_results.csv` is written there.

The matrix CSV includes per-timestep rows with:
- condition fields: `graph`, `model`, `bot`, `seed`, `t`
- dynamics fields: `variance`, `delta_from_t0`, `delta_from_prev`
- semantic side counts (semantic rows): `left_count`, `center_left_count`, `center_right_count`, `right_count`
- graph structure fields: node/edge counts, degree stats, density, components, isolates, local vs long-range edges
- `bot_degree` for bot-on rows

## Active Runtime Files

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
