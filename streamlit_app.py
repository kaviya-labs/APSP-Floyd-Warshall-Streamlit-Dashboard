import math
import time
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="APSP • Floyd–Warshall Visual Analytics",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Theme / lightweight styling
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .metric-card {
        padding: 0.85rem 1rem;
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 12px;
        background: rgba(128,128,128,.05);
    }
    .small-note {font-size: .86rem; opacity: .78;}
    .prediction {
        padding: 1rem 1.1rem;
        border-radius: 12px;
        border: 1px solid rgba(50,120,200,.35);
        background: rgba(50,120,200,.07);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔗 All-Pairs Shortest Paths : Interactive Dashboard")
st.caption(
    "Floyd–Warshall study dashboard based on the submitted dissertation and practical notebook. "
    "It exposes the matrix evolution, relaxation events, path reconstruction, validation, benchmarking, "
    "and a rule-based algorithm prediction panel."
)


# -----------------------------
# Graph construction
# -----------------------------
def build_directed_graph(
    nodes: Sequence[Hashable],
    edges: Sequence[Tuple[Hashable, Hashable, float]],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for source, target, weight in edges:
        graph.add_edge(source, target, weight=float(weight))
    return graph


TEACHING_NODES = ["A", "B", "C", "D"]
TEACHING_EDGES = [
    ("A", "B", 5), ("A", "D", 10), ("B", "C", 3),
    ("B", "D", 9), ("C", "D", 1), ("D", "A", 2), ("D", "C", 4),
]
TEACHING_GRAPH = build_directed_graph(TEACHING_NODES, TEACHING_EDGES)

NEGATIVE_NODES = ["A", "B", "C", "D", "E"]
NEGATIVE_EDGES = [
    ("A", "B", 3), ("A", "C", 8), ("A", "E", -4),
    ("B", "D", 1), ("B", "E", 7), ("C", "B", 4),
    ("D", "A", 2), ("D", "C", -5), ("E", "D", 6),
]
NEGATIVE_GRAPH = build_directed_graph(NEGATIVE_NODES, NEGATIVE_EDGES)


def make_sparse_graph(node_count: int = 12) -> nx.DiGraph:
    nodes = list(range(node_count))
    edges = []
    for i in nodes:
        edges.append((i, (i + 1) % node_count, float((3 * i + 2) % 9 + 1)))
    for i in nodes:
        j = (i + 4) % node_count
        edges.append((i, j, float((5 * i + 3) % 11 + 2)))
    return build_directed_graph(nodes, edges)


def make_dense_graph(node_count: int = 9) -> nx.DiGraph:
    nodes = list(range(node_count))
    edges = []
    for i in nodes:
        for j in nodes:
            if i != j:
                edges.append((i, j, float(((7 * i + 3 * j + i * j) % 14) + 1)))
    return build_directed_graph(nodes, edges)


SPARSE_GRAPH = make_sparse_graph()
DENSE_GRAPH = make_dense_graph()

GRAPH_CASES = {
    "Teaching graph": (TEACHING_GRAPH, TEACHING_NODES),
    "Negative-edge graph": (NEGATIVE_GRAPH, NEGATIVE_NODES),
    "Sparse graph": (SPARSE_GRAPH, list(SPARSE_GRAPH.nodes())),
    "Dense graph": (DENSE_GRAPH, list(DENSE_GRAPH.nodes())),
}


# -----------------------------
# Floyd–Warshall explainable engine
# -----------------------------
@dataclass
class FloydWarshallResult:
    nodes: List[Hashable]
    distance: np.ndarray
    predecessor: np.ndarray
    snapshots: List[np.ndarray]
    updates_by_stage: List[List[dict]]
    negative_cycle_nodes: List[Hashable]


def initialise_distance_and_predecessor(
    graph: nx.DiGraph,
    nodes: Sequence[Hashable],
):
    node_index = {node: index for index, node in enumerate(nodes)}
    size = len(nodes)
    distance = np.full((size, size), np.inf, dtype=float)
    predecessor = np.full((size, size), None, dtype=object)
    np.fill_diagonal(distance, 0.0)

    for source, target, data in graph.edges(data=True):
        i, j = node_index[source], node_index[target]
        weight = float(data["weight"])
        if weight < distance[i, j]:
            distance[i, j] = weight
            predecessor[i, j] = source

    return distance, predecessor, node_index


def floyd_warshall_explainable(
    graph: nx.DiGraph,
    nodes: Sequence[Hashable],
) -> FloydWarshallResult:
    """
    Explainable custom Floyd-Warshall implementation.

    Why k is the outer loop:
    k represents the set of vertices currently allowed as intermediate vertices.
    At stage k, every d[i][j] asks one precise question:
        "Is i -> j shorter if the newly allowed vertex k is used?"
    Keeping k outside the i/j loops guarantees that the matrix evolves through
    well-defined dynamic-programming stages and that each snapshot is auditable.

    Why the triple loop:
    k: choose the newly permitted intermediate vertex.
    i: choose the source vertex.
    j: choose the target vertex.
    The relaxation compares the current d[i][j] with
    d[i][k] + d[k][j].
    This checks every ordered source-target pair for every possible intermediate,
    giving the standard O(V^3) Floyd-Warshall time complexity.
    """
    distance, predecessor, node_index = initialise_distance_and_predecessor(graph, nodes)
    snapshots = [distance.copy()]
    updates_by_stage = []

    # k must be the outer loop because each stage means:
    # "allow nodes[0:k+1] as intermediate vertices".
    for k, intermediate in enumerate(nodes):
        stage_updates = []

        # i and j inspect every ordered source-target pair for this k.
        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                via_k_left = distance[i, k]
                via_k_right = distance[k, j]

                # If either half of i -> k -> j is unreachable, this candidate
                # cannot improve the current source-target distance.
                if not np.isfinite(via_k_left) or not np.isfinite(via_k_right):
                    continue

                candidate = via_k_left + via_k_right

                # Dynamic-programming relaxation:
                # keep the old route unless going through k is strictly shorter.
                if candidate < distance[i, j]:
                    old_distance = distance[i, j]
                    distance[i, j] = candidate
                    predecessor[i, j] = predecessor[k, j]

                    stage_updates.append(
                        {
                            "Stage": k + 1,
                            "Intermediate": intermediate,
                            "Source": source,
                            "Target": target,
                            "Old distance": old_distance,
                            "New distance": candidate,
                            "Improvement": (
                                old_distance - candidate
                                if np.isfinite(old_distance)
                                else np.inf
                            ),
                        }
                    )

        updates_by_stage.append(stage_updates)
        snapshots.append(distance.copy())

    negative_cycle_nodes = [
        nodes[i] for i in range(len(nodes)) if distance[i, i] < 0
    ]

    return FloydWarshallResult(
        nodes=list(nodes),
        distance=distance,
        predecessor=predecessor,
        snapshots=snapshots,
        updates_by_stage=updates_by_stage,
        negative_cycle_nodes=negative_cycle_nodes,
    )



def reconstruct_path(
    result: FloydWarshallResult,
    source: Hashable,
    target: Hashable,
) -> Optional[List[Hashable]]:
    if result.negative_cycle_nodes:
        raise ValueError(
            "Shortest paths are undefined because a negative cycle is present."
        )

    index = {node: i for i, node in enumerate(result.nodes)}
    i, j = index[source], index[target]

    if source == target:
        return [source]
    if result.predecessor[i, j] is None:
        return None

    path = [target]
    current = target

    for _ in range(len(result.nodes) + 1):
        if current == source:
            return list(reversed(path))
        current = result.predecessor[i, index[current]]
        if current is None:
            return None
        path.append(current)

    raise RuntimeError("Path reconstruction exceeded the safe predecessor limit.")


def path_weight(graph: nx.DiGraph, path: Sequence[Hashable]) -> float:
    if len(path) <= 1:
        return 0.0
    return float(sum(graph[u][v]["weight"] for u, v in zip(path[:-1], path[1:])))


# -----------------------------
# Validation helpers
# -----------------------------
def matrix_from_distance_dictionary(distances: Dict, nodes: Sequence[Hashable]):
    matrix = np.full((len(nodes), len(nodes)), np.inf, dtype=float)
    for i, source in enumerate(nodes):
        matrix[i, i] = 0.0
        for target, value in distances.get(source, {}).items():
            matrix[i, nodes.index(target)] = float(value)
    return matrix


def networkx_floyd_matrix(graph, nodes):
    return matrix_from_distance_dictionary(
        dict(nx.floyd_warshall(graph, weight="weight")), list(nodes)
    )


def repeated_dijkstra_matrix(graph, nodes):
    if any(data["weight"] < 0 for _, _, data in graph.edges(data=True)):
        raise ValueError("Dijkstra is not applicable to negative edge weights.")
    distances = {
        source: nx.single_source_dijkstra_path_length(graph, source, weight="weight")
        for source in nodes
    }
    return matrix_from_distance_dictionary(distances, nodes)


def repeated_bellman_ford_matrix(graph, nodes):
    if nx.negative_edge_cycle(graph, weight="weight"):
        raise nx.NetworkXUnbounded("Negative cycle.")
    distances = {
        source: nx.single_source_bellman_ford_path_length(graph, source, weight="weight")
        for source in nodes
    }
    return matrix_from_distance_dictionary(distances, nodes)


def johnson_distance_matrix(graph, nodes):
    paths = nx.johnson(graph, weight="weight")
    matrix = np.full((len(nodes), len(nodes)), np.inf, dtype=float)
    for i, source in enumerate(nodes):
        matrix[i, i] = 0.0
        for j, target in enumerate(nodes):
            if target in paths[source]:
                matrix[i, j] = nx.path_weight(graph, paths[source][target], weight="weight")
    return matrix


def matrices_match(left, right):
    if not np.array_equal(np.isfinite(left), np.isfinite(right)):
        return False
    mask = np.isfinite(left)
    return bool(np.allclose(left[mask], right[mask]))


@st.cache_data(show_spinner=False)
def run_validation(graph_name: str):
    graph, nodes = GRAPH_CASES[graph_name]
    result = floyd_warshall_explainable(graph, nodes)
    rows = []

    checks = [
        ("NetworkX Floyd–Warshall", lambda: networkx_floyd_matrix(graph, nodes)),
        ("Repeated Dijkstra", lambda: repeated_dijkstra_matrix(graph, nodes)),
        ("Repeated Bellman–Ford", lambda: repeated_bellman_ford_matrix(graph, nodes)),
        ("Johnson", lambda: johnson_distance_matrix(graph, nodes)),
    ]

    for method, fn in checks:
        try:
            comparison = matrices_match(result.distance, fn())
            rows.append(
                {
                    "Method": method,
                    "Applicable": True,
                    "Matches custom matrix": comparison,
                    "Status": "Agreement" if comparison else "Mismatch",
                }
            )
        except (ValueError, nx.NetworkXUnbounded):
            rows.append(
                {
                    "Method": method,
                    "Applicable": False,
                    "Matches custom matrix": None,
                    "Status": "Not applicable",
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def run_path_evaluation(graph_name: str):
    graph, nodes = GRAPH_CASES[graph_name]
    result = floyd_warshall_explainable(graph, nodes)
    finite_pairs = recovered_pairs = correct_weights = 0

    if result.negative_cycle_nodes:
        return pd.DataFrame(
            [{
                "Graph": graph_name,
                "Finite pairs": 0,
                "Recovered pairs": 0,
                "Correct path weights": 0,
                "Path recovery completeness (%)": np.nan,
                "Path weight correctness (%)": np.nan,
            }]
        )

    for i, source in enumerate(nodes):
        for j, target in enumerate(nodes):
            if np.isfinite(result.distance[i, j]):
                finite_pairs += 1
                path = reconstruct_path(result, source, target)
                if path and path[0] == source and path[-1] == target:
                    recovered_pairs += 1
                    if math.isclose(
                        path_weight(graph, path),
                        result.distance[i, j],
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ):
                        correct_weights += 1

    return pd.DataFrame(
        [{
            "Graph": graph_name,
            "Finite pairs": finite_pairs,
            "Recovered pairs": recovered_pairs,
            "Correct path weights": correct_weights,
            "Path recovery completeness (%)": (
                100 * recovered_pairs / finite_pairs if finite_pairs else np.nan
            ),
            "Path weight correctness (%)": (
                100 * correct_weights / finite_pairs if finite_pairs else np.nan
            ),
        }]
    )


# -----------------------------
# Benchmark implementation
# -----------------------------
def make_benchmark_graph(node_count: int, density_label: str):
    nodes = list(range(node_count))
    edges = []

    for i in nodes:
        edges.append(
            (i, (i + 1) % node_count, float(((11 * i + 5) % 17) + 1))
        )

    if density_label == "Sparse":
        for i in nodes:
            for offset in (3, 7):
                if offset < node_count:
                    j = (i + offset) % node_count
                    edges.append(
                        (i, j, float(((13 * i + 7 * j + offset) % 23) + 1))
                    )
    else:
        for i in nodes:
            for j in nodes:
                if i != j:
                    edges.append(
                        (i, j, float(((17 * i + 19 * j + i * j) % 29) + 1))
                    )

    return build_directed_graph(nodes, edges)


def time_callable(function, repeats=3):
    measurements = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        measurements.append(time.perf_counter() - start)
    return float(np.median(measurements))


@st.cache_data(show_spinner=False)
def benchmark_algorithms(sizes=(10, 20, 30, 40), repeats=3):
    rows = []
    for density in ("Sparse", "Dense"):
        for n in sizes:
            graph = make_benchmark_graph(n, density)
            nodes = list(graph.nodes())
            algorithms = {
                "Custom Floyd–Warshall": lambda: floyd_warshall_explainable(graph, nodes),
                "Repeated Dijkstra": lambda: [
                    nx.single_source_dijkstra_path_length(graph, source, weight="weight")
                    for source in nodes
                ],
                "Repeated Bellman–Ford": lambda: [
                    nx.single_source_bellman_ford_path_length(graph, source, weight="weight")
                    for source in nodes
                ],
                "Johnson": lambda: nx.johnson(graph, weight="weight"),
            }

            for name, fn in algorithms.items():
                rows.append(
                    {
                        "Density": density,
                        "Nodes": n,
                        "Edges": graph.number_of_edges(),
                        "Algorithm": name,
                        "Median seconds": time_callable(fn, repeats),
                    }
                )
    return pd.DataFrame(rows)


# -----------------------------
# Display helpers
# -----------------------------
def format_matrix(matrix, nodes):
    df = pd.DataFrame(matrix, index=nodes, columns=nodes)
    return df.map(lambda x: "∞" if not np.isfinite(x) else round(float(x), 3))


def final_distance_table_for_all_graphs():
    """Return the final APSP matrix for every controlled graph case."""
    rows = []
    for graph_name, (case_graph, case_nodes) in GRAPH_CASES.items():
        case_result = floyd_warshall_explainable(case_graph, case_nodes)
        if case_result.negative_cycle_nodes:
            rows.append(
                {
                    "Graph": graph_name,
                    "Status": "Negative cycle detected; shortest distances undefined",
                    "Final APSP distance matrix": "Rejected",
                }
            )
        else:
            rows.append(
                {
                    "Graph": graph_name,
                    "Status": "Valid final APSP matrix",
                    "Final APSP distance matrix": format_matrix(
                        case_result.distance, case_nodes
                    ),
                }
            )
    return rows


def graph_summary_df():
    rows = []
    for name, (graph, nodes) in GRAPH_CASES.items():
        possible = len(nodes) * (len(nodes) - 1)
        rows.append(
            {
                "Graph": name,
                "Vertices": graph.number_of_nodes(),
                "Edges": graph.number_of_edges(),
                "Density": graph.number_of_edges() / possible if possible else 0,
                "Negative edge": any(
                    data["weight"] < 0 for _, _, data in graph.edges(data=True)
                ),
                "Negative cycle": nx.negative_edge_cycle(graph, weight="weight"),
            }
        )
    return pd.DataFrame(rows)


def draw_graph(graph, title, path_edges=None):
    positions = nx.circular_layout(graph)
    emphasized = set(path_edges or [])
    widths = [3.8 if (u, v) in emphasized else 1.0 for u, v in graph.edges()]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    nx.draw_networkx(
        graph,
        pos=positions,
        with_labels=True,
        arrows=True,
        width=widths,
        node_size=1300,
        ax=ax,
    )
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=nx.get_edge_attributes(graph, "weight"),
        ax=ax,
        font_size=9,
    )
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Dashboard controls")
    selected_graph_name = st.selectbox("Graph scenario", list(GRAPH_CASES.keys()))
    graph, nodes = GRAPH_CASES[selected_graph_name]

    show_benchmarks = st.checkbox(
        "Enable runtime benchmark",
        value=False,
        help="Runs deterministic 10/20/30/40 vertex benchmark cases using the same methodology as the practical.",
    )

    st.divider()
    st.markdown("**Study focus**")
    st.write("1. APSP + Floyd–Warshall")
    st.write("2. Matrix evolution")
    st.write("3. Relaxation logging")
    st.write("4. Path reconstruction")
    st.write("5. Validation")
    st.write("6. Runtime comparison")
    st.write("7. Algorithm prediction")

result = floyd_warshall_explainable(graph, nodes)
summary = graph_summary_df()
selected_summary = summary[summary["Graph"] == selected_graph_name].iloc[0]

# -----------------------------
# KPI row
# -----------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Vertices", int(selected_summary["Vertices"]))
c2.metric("Edges", int(selected_summary["Edges"]))
c3.metric("Density", f"{selected_summary['Density']:.3f}")
c4.metric("Relaxation updates", sum(len(x) for x in result.updates_by_stage))
c5.metric(
    "Negative cycle",
    "Detected" if result.negative_cycle_nodes else "None",
)

st.divider()

tabs = st.tabs(
    [
        "📊 Dashboard",
        "🧮 Matrix Evolution",
        "🛣️ Prediction Panel",
        "✅ Validation",
        "⏱️ Benchmarks",
        "📋 Algorithm Comparison",
    ]
)

# -----------------------------
# Dashboard
# -----------------------------
with tabs[0]:
    with st.expander("What is happening in the Dashboard?", expanded=True):
        st.write(
            "The Dashboard is the overview of the selected experimental graph. The sidebar selection "
            "controls which deterministic graph is analysed. The graph view shows the directed weighted "
            "structure, the final APSP matrix shows the shortest distance between every ordered pair, "
            "the status message reports negative-edge or negative-cycle conditions, and the KPI row "
            "summarises vertices, edges, density, relaxation updates and cycle status. The expandable "
            "matrices below expose the final result for all four controlled graph cases, while the relaxation "
            "chart shows where useful matrix changes occurred."
        )

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Selected graph")
        draw_graph(graph, f"{selected_graph_name} - directed weighted graph")

    with right:
        st.subheader("Final APSP distance matrix")
        st.dataframe(format_matrix(result.distance, nodes), use_container_width=True)

        if result.negative_cycle_nodes:
            st.error(
                "Shortest-path output is not accepted because a negative cycle was detected."
            )
        elif any(data["weight"] < 0 for _, _, data in graph.edges(data=True)):
            st.info(
                "Negative edges are present, but the selected graph has no negative cycle. "
                "Floyd–Warshall remains applicable; Dijkstra is excluded."
            )
        else:
            st.success("No negative edge is present in this scenario.")

    st.subheader("Final shortest distances for every graph")
    st.caption(
        "This section shows the final APSP result for all controlled graph scenarios, "
        "not only the graph selected in the sidebar."
    )
    for graph_name, (case_graph, case_nodes) in GRAPH_CASES.items():
        case_result = floyd_warshall_explainable(case_graph, case_nodes)
        with st.expander(f"{graph_name} — final shortest-distance matrix"):
            if case_result.negative_cycle_nodes:
                st.error(
                    "Negative cycle detected. A finite shortest-distance matrix is not "
                    "accepted for this graph."
                )
            else:
                st.dataframe(
                    format_matrix(case_result.distance, case_nodes),
                    use_container_width=True,
                )

    st.subheader("Relaxation behaviour")
    counts = pd.DataFrame(
        {
            "Intermediate vertex": nodes,
            "Improved cells": [len(x) for x in result.updates_by_stage],
        }
    ).set_index("Intermediate vertex")
    st.bar_chart(counts)

    st.caption(
        "The study explicitly treats update counts as an interpretive measure of useful matrix changes, "
        "not as a replacement for the O(V³) complexity of Floyd–Warshall."
    )

# -----------------------------
# Matrix evolution
# -----------------------------
with tabs[1]:
    with st.expander("What is happening in Matrix Evolution?", expanded=True):
        st.write(
            "This section exposes the internal dynamic-programming stages of Floyd–Warshall. "
            "The stage selector chooses the initial matrix or a snapshot after a particular intermediate "
            "vertex k is allowed. For that stage, the table and heatmap show the current distance matrix, "
            "and the update table identifies every successful relaxation by source, target, old value, "
            "new value and improvement. The implementation keeps k as the outer loop so that each snapshot "
            "has a clear meaning: a new intermediate vertex has just been added to the set of permitted "
            "intermediate vertices."
        )

    st.subheader("Interactive matrix snapshot viewer")
    stage_labels = ["Initial - no intermediate vertex"]
    stage_labels += [f"After {n} is allowed as intermediate" for n in nodes]

    selected_label = st.selectbox("Select matrix stage", stage_labels)
    stage_index = stage_labels.index(selected_label)

    st.info(
        "**Why k is the outer loop:** k is the newly allowed intermediate vertex. "
        "For each k, the inner i and j loops test every source-target pair using "
        "the candidate route i → k → j. The three loops therefore implement the "
        "dynamic-programming recurrence and give the O(V³) structure of "
        "Floyd–Warshall. The snapshots below make each k-stage visible."
    )

    st.dataframe(
        format_matrix(result.snapshots[stage_index], nodes),
        use_container_width=True,
    )

    finite = result.snapshots[stage_index][np.isfinite(result.snapshots[stage_index])]
    if finite.size:
        display_cap = float(finite.max() + max(1, finite.std()))
    else:
        display_cap = 1.0

    visible = np.where(np.isfinite(result.snapshots[stage_index]), result.snapshots[stage_index], display_cap)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    image = ax.imshow(visible, aspect="auto")
    fig.colorbar(image, ax=ax, label="Distance (∞ display-capped)")
    ax.set_xticks(range(len(nodes)), nodes)
    ax.set_yticks(range(len(nodes)), nodes)
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    ax.set_title(selected_label)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            label = "∞" if not np.isfinite(result.snapshots[stage_index][i, j]) else f"{result.snapshots[stage_index][i, j]:g}"
            ax.text(j, i, label, ha="center", va="center")
    st.pyplot(fig, clear_figure=True)

    updates = result.updates_by_stage[stage_index - 1] if stage_index > 0 else []
    st.subheader("Updates introduced at this stage")
    if updates:
        update_df = pd.DataFrame(updates)
        st.dataframe(
            update_df.style.format(
                {
                    "Old distance": lambda x: "∞" if not np.isfinite(x) else f"{x:g}",
                    "New distance": "{:.3g}",
                    "Improvement": lambda x: "newly reachable" if not np.isfinite(x) else f"{x:.3g}",
                }
            ),
            use_container_width=True,
        )
    else:
        st.info("No matrix cells improved at this stage.")

# -----------------------------
# Prediction panel
# -----------------------------
with tabs[2]:
    with st.expander("What is happening in the Prediction & Decision Panel?", expanded=True):
        st.write(
            "The first part predicts the shortest route for the selected source and target using the "
            "predecessor matrix and verifies the route by independently summing its edge weights. The graph "
            "then highlights that route. The second part is a rule-based algorithm decision aid: the user "
            "chooses whether interpretability, measured speed or APSP suitability is the priority, and the "
            "application recommends an algorithm using the graph's negative-edge condition, density, size "
            "and the findings of this study. It is a transparent decision rule, not a machine-learning model."
        )

    st.subheader("🔮 Prediction & decision panel")

    if result.negative_cycle_nodes:
        st.error(
            f"Prediction disabled for shortest-path output: negative cycle detected at "
            f"{result.negative_cycle_nodes}."
        )
    else:
        p1, p2, p3 = st.columns(3)
        source = p1.selectbox("Source", nodes, key=f"source_{selected_graph_name}")
        target = p2.selectbox("Target", nodes, key=f"target_{selected_graph_name}")
        objective = p3.selectbox(
            "Decision objective",
            ["Best interpretability", "Best measured speed", "Best APSP fit"],
        )

        path = reconstruct_path(result, source, target)
        if path is None:
            st.warning(f"No finite route exists from {source} to {target}.")
        else:
            distance = result.distance[nodes.index(source), nodes.index(target)]
            weight = path_weight(graph, path)

            st.markdown(
                f"""
                <div class="prediction">
                <b>Predicted shortest route</b><br>
                {' → '.join(map(str, path))}<br>
                <b>Predicted distance:</b> {distance:g}
                &nbsp; | &nbsp;
                <b>Verified path weight:</b> {weight:g}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.success("Path weight agrees with the selected APSP matrix value.")

            draw_graph(
                graph,
                f"Predicted route: {' → '.join(map(str, path))}",
                list(zip(path[:-1], path[1:])),
            )

        # Rule-based algorithm selection prediction.
        has_negative = any(data["weight"] < 0 for _, _, data in graph.edges(data=True))
        density = float(selected_summary["Density"])
        n = int(selected_summary["Vertices"])

        if has_negative:
            predicted = "Floyd–Warshall" if objective != "Best measured speed" else "Bellman–Ford / Johnson"
            rationale = (
                "Negative edges are present, so Dijkstra is excluded. "
                "For this study, Floyd–Warshall provides the clearest complete matrix evolution; "
                "Bellman–Ford and Johnson remain applicable when there is no negative cycle."
            )
        elif objective == "Best interpretability":
            predicted = "Floyd–Warshall"
            rationale = "The study's main contribution is transparent, auditable matrix evolution."
        elif objective == "Best measured speed":
            predicted = "Repeated Dijkstra"
            rationale = (
                "Repeated Dijkstra was the fastest measured method in the controlled positive-weight "
                "benchmark cases in the practical."
            )
        elif density < 0.35 and n >= 10:
            predicted = "Johnson"
            rationale = (
                "The study identifies Johnson as a strong APSP alternative for sparse graphs, "
                "especially when repeated single-source computation is suitable."
            )
        else:
            predicted = "Floyd–Warshall"
            rationale = (
                "The selected graph is a good fit when complete pairwise distances and matrix-based "
                "interpretability are more important than raw execution speed."
            )

        st.markdown("### Algorithm recommendation prediction")
        st.markdown(
            f"""
            <div class="prediction">
            <b>Predicted method: {predicted}</b><br>
            {rationale}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "This is a rule-based prediction/decision aid derived from the study's graph assumptions "
            "and measured findings. "
        )

# -----------------------------
# Validation
# -----------------------------
with tabs[3]:
    with st.expander("What is happening in Validation?", expanded=True):
        st.write(
            "The Validation section checks the custom implementation against independent NetworkX shortest-"
            "path routines. Methods that violate the selected graph's assumptions are marked not applicable "
            "rather than treated as implementation failures. The section also checks whether predecessor-"
            "based paths can be recovered and whether each recovered path has the same total weight as the "
            "corresponding APSP matrix entry."
        )

    st.subheader("Validation against applicable reference algorithms")
    validation = run_validation(selected_graph_name)
    st.dataframe(validation, use_container_width=True, hide_index=True)

    applicable = validation[validation["Applicable"]]
    if len(applicable):
        agreement = 100 * applicable["Matches custom matrix"].mean()
        st.metric("Agreement among applicable comparisons", f"{agreement:.0f}%")

    evaluation = run_path_evaluation(selected_graph_name)
    st.subheader("Path-recovery validation")
    st.dataframe(evaluation, use_container_width=True, hide_index=True)

    if not result.negative_cycle_nodes:
        cols = st.columns(2)
        cols[0].metric(
            "Path recovery completeness",
            f"{evaluation.iloc[0]['Path recovery completeness (%)']:.0f}%",
        )
        cols[1].metric(
            "Path-weight correctness",
            f"{evaluation.iloc[0]['Path weight correctness (%)']:.0f}%",
        )

    st.caption(
        "The dissertation reports 100% agreement for the 15 applicable algorithm/graph comparisons "
        "and 100% recovery plus weight correctness across 266 finite source-target pairs in the controlled cases."
    )

# -----------------------------
# Benchmarks
# -----------------------------
with tabs[4]:
    with st.expander("What is happening in Benchmarks?", expanded=False):
        st.write(
            "The benchmark object is optional because it runs additional deterministic experiments. When "
            "enabled, the application measures the four algorithms on sparse and dense graph families with "
            "10, 20, 30 and 40 vertices. Three executions are used for each case and the median time is "
            "displayed so the result is consistent with the dissertation's runtime methodology."
        )

    st.subheader("Controlled runtime benchmark")

    if not show_benchmarks:
        st.info(
            "Benchmark execution is disabled. Enable **Runtime benchmark** in the sidebar to run "
            "deterministic 10/20/30/40 vertex cases."
        )
    else:
        with st.spinner("Running deterministic benchmark cases..."):
            benchmark = benchmark_algorithms()

        density_choice = st.selectbox("Benchmark family", ["Sparse", "Dense"])
        subset = benchmark[benchmark["Density"] == density_choice]

        chart_df = subset.pivot(
            index="Nodes", columns="Algorithm", values="Median seconds"
        )
        st.line_chart(chart_df)

        st.dataframe(
            subset.round({"Median seconds": 6}),
            use_container_width=True,
            hide_index=True,
        )

        fastest = (
            benchmark.sort_values("Median seconds")
            .groupby(["Density", "Nodes"], as_index=False)
            .first()
        )
        st.subheader("Fastest measured method by case")
        st.dataframe(
            fastest[
                ["Density", "Nodes", "Edges", "Algorithm", "Median seconds"]
            ].round({"Median seconds": 6}),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Timing is environment-dependent. The study uses runtime as one criterion rather than "
            "a universal ranking of algorithms."
        )

# -----------------------------
# Algorithm comparison
# -----------------------------
with tabs[5]:
    with st.expander("What is happening in Algorithm Comparison?", expanded=True):
        st.write(
            "This view connects the application's computational results to algorithm-selection reasoning. "
            "It compares Floyd–Warshall, repeated Dijkstra, repeated Bellman–Ford and Johnson in terms of "
            "APSP/SSSP orientation, negative-edge support, density suitability, theoretical complexity and "
            "the main measured finding from this study. The purpose is not to declare one universally best "
            "algorithm, but to show why suitability changes with graph conditions and the analysis objective."
        )

    st.subheader("Comparative algorithm view")

    comparison = pd.DataFrame(
        [
            {
                "Algorithm": "Floyd–Warshall",
                "APSP directly": "Yes",
                "Negative edges": "Yes, without negative cycle",
                "Complete matrix evolution": "High",
                "Interpretability in study": "Very high",
                "Best fit in study": "Complete APSP, teaching, auditability",
                "Typical time": "O(V³)",
                "Typical space": "O(V²)",
            },
            {
                "Algorithm": "Repeated Dijkstra",
                "APSP directly": "No - repeated SSSP",
                "Negative edges": "No",
                "Complete matrix evolution": "Low",
                "Interpretability in study": "Moderate",
                "Best fit in study": "Sparse non-negative graphs",
                "Typical time": "Implementation-dependent",
                "Typical space": "Graph + per-source structures",
            },
            {
                "Algorithm": "Repeated Bellman–Ford",
                "APSP directly": "No - repeated SSSP",
                "Negative edges": "Yes",
                "Complete matrix evolution": "Low",
                "Interpretability in study": "Moderate",
                "Best fit in study": "Negative-edge single-source computations",
                "Typical time": "O(V²E) when repeated",
                "Typical space": "Graph + per-source structures",
            },
            {
                "Algorithm": "Johnson",
                "APSP directly": "Yes",
                "Negative edges": "Yes, without negative cycle",
                "Complete matrix evolution": "Low",
                "Interpretability in study": "Moderate",
                "Best fit in study": "Sparse APSP",
                "Typical time": "Usually O(VE log V)",
                "Typical space": "Graph + APSP result",
            },
        ]
    )

    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.info(
        "Study conclusion: the question is not simply 'which algorithm is best?' but "
        "'which algorithm is most appropriate for this graph, objective, and evaluation criterion?'"
    )

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "Study-aligned scope: weighted directed graphs, deterministic inputs, Floyd–Warshall as the "
    "main algorithm, comparison with Dijkstra/Bellman–Ford/Johnson, visual matrix evolution, "
    "path reconstruction, negative-cycle safeguarding, validation, and controlled runtime analysis."
)
