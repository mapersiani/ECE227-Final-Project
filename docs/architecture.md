# Project Architecture & Run Flow

The following Mermaid diagram visualizes the structure of the ECE227 Final Project following our recent integrations. It demonstrates how `main.py` orchestrates the graph building, the baseline models, the semantic simulations, and the comprehensive metric logging.

```mermaid
graph TD
    classDef main fill:#d8b4e2,stroke:#333,stroke-width:2px,color:#000;
    classDef module fill:#a2c4c9,stroke:#333,stroke-width:1px,color:#000;
    classDef logic fill:#cfe2f3,stroke:#333,stroke-width:1px,color:#000;
    classDef data fill:#d9ead3,stroke:#333,stroke-width:1px,color:#000;
    classDef output_node fill:#fce5cd,stroke:#333,stroke-width:2px,color:#000;

    Main["main.py<br>(CLI Orchestrator)"]:::main
    
    subgraph Data Loading
        Config["src/config.py"]:::data
        LoadNodes["src/load_nodes.py"]:::data
        JSONData[("data/*.json<br>Personas")]:::data
    end

    subgraph Graph Topologies
        ER["src/graphs/er.py<br>(Erdős-Rényi)"]:::module
        RGGLR["src/graphs/rgg_long_range.py<br>(RGGLR)"]:::module
    end

    subgraph Simulation Engines
        DeGroot["src/degroot.py<br>(Scalar Baseline)"]:::logic
        SimSemantic["src/simulation.py<br>(Semantic: No Bot)"]:::logic
        SimIntervention["src/intervention.py<br>(Semantic: With Bot)"]:::logic
    end

    subgraph Agent Core
        Agent["src/agent.py<br>(Agent State Tracking)"]:::logic
        LLM["src/llm_client.py<br>(LLM API Prompts)"]:::logic
    end

    subgraph Measurement
        MeasurementModule["src/measurement.py<br>(SBERT Polarization & Drift)"]:::module
        GraphMetrics["main.py:_graph_structure_metrics<br>(NetworkX Analytics)"]:::module
    end

    OutDir[("outputs/<br>(Plots, CSVs, JSONs)")]:::output_node

    %% Run Flow Orchestration
    Main -->|Parses Arguments| Config
    Main -->|1. Build Graph| ER
    Main -->|1. Build Graph| RGGLR
    
    ER --> LoadNodes
    RGGLR --> LoadNodes
    LoadNodes --> JSONData

    Main -->|2. Run Baseline| DeGroot
    Main -->|3. Get Graph NetworkX Stats| GraphMetrics
    
    Main -->|4a. If --bot off| SimSemantic
    Main -->|4b. If --bot on| SimIntervention

    %% Semantic Engine Dependencies
    SimSemantic --> Agent
    SimIntervention --> Agent
    Agent -->|Calls| LLM

    SimSemantic -->|Measures Iterations| MeasurementModule
    SimIntervention -->|Measures Iterations| MeasurementModule

    %% Outputs Output
    DeGroot -.->|Baseline Variances| OutDir
    GraphMetrics -.->|Topology Specs| OutDir
    SimSemantic -.->|Semantic Arrays| OutDir
    SimIntervention -.->|Semantic Arrays| OutDir
```

## Key Workflows

1. **Initialization:** `main.py` parses arguments (run or matrix) and calls the graph builders (`er.py` or `rgg_long_range.py`). These builders use `load_nodes.py` to populate nodes with text personas from the `data` directory.
2. **DeGroot Baseline:** Before any LLM calls occur, `main.py` extracts the ideological blocks of the graph nodes, translates them to simple scalars using `config.py`, and runs the standard DeGroot consensus simulation via `degroot.py`.
3. **Semantic Semantic Execution:** Depending on the `--bot` flag, `main.py` hands the graph over to either `simulation.py` or `intervention.py`. 
4. **Agent Interaction:** During semantic execution, graph agents (`agent.py`) observe their neighbors and formulate new opinions by querying LLM models (`llm_client.py`).
5. **Continuous Measurement:** After each step, the execution loop sends all opinions to `measurement.py`, which uses SBERT to calculate `semantic_variance`, `opinion_polarization`, and `persona_drift`. 
6. **Data Aggregation:** Finally, `main.py` calculates structural centrality scores (`_graph_structure_metrics`) and dumps all arrays and properties to respective CSV files and `.png` charts in the `outputs/` directory.
