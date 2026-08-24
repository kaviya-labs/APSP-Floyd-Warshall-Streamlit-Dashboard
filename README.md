# APSP Floyd-Warshall Streamlit Dashboard

This app turns my Floyd–Warshall/APSP practical into an interactive Streamlit dashboard.

## Included
1. Teaching, negative-edge, sparse and dense graph scenarios
2. Final APSP distance matrix
3. Interactive Floyd-Warshall matrix snapshots
4. Relaxation/update log
5. Shortest-path prediction for a selected source/target
6. Rule-based algorithm recommendation panel
7. NetworkX / Dijkstra / Bellman–Ford / Johnson validation
8. Path-recovery completeness and path-weight correctness
9. Optional deterministic runtime benchmarks for 10/20/30/40 vertices
10. Comparative algorithm table

## Run if running on local system

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Important points

The "prediction panel" is kept as a transparent rule-based decision aid, not an ML model. It predicts:
1. the shortest route and distance for a selected source/target, and
2. an appropriate algorithm based on graph properties and the study's findings.