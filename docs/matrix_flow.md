# Matrix Run Execution Loop

This diagram illustrates how `main_matrix` systematically sweeps through all permutations of Graph topology, Persona Data, and Initial random seed.

```mermaid
graph TD
    classDef start fill:#d8b4e2,stroke:#333,stroke-width:2px,color:#000;
    classDef loop_layer fill:#f9cb9c,stroke:#333,stroke-width:2px,color:#000;
    classDef sim fill:#cfe2f3,stroke:#333,stroke-width:1px,color:#000;
    classDef output_node fill:#fce5cd,stroke:#333,stroke-width:2px,color:#000;

    Start((Start Matrix Run)):::start
    
    Start --> LoopGraph[For Graph in: ER, RGGLR]:::loop_layer

    subgraph Outer Execution
        LoopGraph --> LoopPersona[For Persona Set in: personas, senate]:::loop_layer
        LoopPersona --> LoopSeed[For Seed in: SEED_LIST]:::loop_layer
        
        subgraph Graph Initialization [Seed Iteration]
            LoopSeed --> BuildGraph["_build_graph(graph_type, seed, persona_set)<br>Returns fully populated topology"]
            BuildGraph --> BaseMetrics["_graph_structure_metrics(G)"]
            
            subgraph Parallel Models
                BaseMetrics --> DeGroot["run_degroot()<br>Baseline"]:::sim
                BaseMetrics --> SemOff["run_semantic()<br>Bot: Off"]:::sim
                BaseMetrics --> SemOn["run_with_bot_on_graph()<br>Bot: On"]:::sim
            end
            
            DeGroot --> AppendRowD[Append Matrix Row]
            SemOff --> AppendRowOff[Append Matrix Row]
            SemOn --> AppendRowOn[Append Matrix Row]
        end
    end
    
    AppendRowD -.-> MatrixCSV[("matrix_results.csv")]:::output_node
    AppendRowOff -.-> MatrixCSV
    AppendRowOn -.-> MatrixCSV
    
    AppendRowOff -.-> TransitionAcc["Accumulate Side Transitions"]
    AppendRowOn -.-> TransitionAcc
    
    LoopSeed -- "Next Seed" --> LoopSeed
    LoopPersona -- "Next Set" --> LoopPersona
    LoopGraph -- "Next Graph" --> LoopGraph
    
    LoopGraph -- "Complete" --> Finalize[Finalize Analysis]
    
    Finalize --> MatrixCSV
    Finalize --> Json[("matrix_summary.json")]:::output_node
    Finalize --> Plots[/"Plots<br>(Heatmaps, Bars, Trajectories)"/]:::output_node
```

## Total Permutations
The script evaluates every combination automatically. With the defaults (2 graph types, 2 persona sets, 3 seeds), the script initializes **12 unique graphs**.

For each of those 12 graphs, it runs 3 separate simulation variations (DeGroot, Semantic No Bot, Semantic With Bot), producing **36 independent rows** in the final `matrix_results.csv` output.
