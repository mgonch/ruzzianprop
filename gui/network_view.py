"""
Network visualization helpers for the Streamlit GUI.

Builds on the existing GraphBuilder + CommunityDetector and produces
Plotly figures that embed cleanly in Streamlit.

Figures provided
----------------
network_graph()          – interactive node-link diagram (Plotly scatter)
community_sunburst()     – sunburst of communities → accounts by bot risk
activity_timeline()      – tweet volume over time, coloured by classification
coordination_heatmap()   – accounts × posting-hours grid (spots coordinated bursts)
community_stats_bars()   – per-community avg bot score + size bar chart
edge_type_sankey()       – sankey of interaction types (RT / mention / co-hashtag)
top_accounts_table()     – DataFrame of top accounts by PageRank, sortable in UI
"""

import math
import logging
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

# Classification → hex colour
CLF_COLOR = {
    "bot":       "#ef4444",  # red
    "suspected": "#f59e0b",  # amber
    "human":     "#22c55e",  # green
    "unknown":   "#94a3b8",  # slate
}

# Community palette (up to 20 communities)
_COMMUNITY_PALETTE = [
    "#6366f1", "#ec4899", "#14b8a6", "#f97316", "#8b5cf6",
    "#06b6d4", "#84cc16", "#f43f5e", "#a855f7", "#10b981",
    "#eab308", "#3b82f6", "#ef4444", "#22c55e", "#64748b",
    "#d946ef", "#0ea5e9", "#fb923c", "#4ade80", "#a78bfa",
]


def _community_color(comm_id: int) -> str:
    return _COMMUNITY_PALETTE[comm_id % len(_COMMUNITY_PALETTE)]


def _bot_score_color(score: float) -> str:
    """Continuous red-green gradient for bot score [0, 1]."""
    r = int(239 * score + 34 * (1 - score))
    g = int(68  * score + 197 * (1 - score))
    b = int(68  * score + 94  * (1 - score))
    return f"rgb({r},{g},{b})"


def _hex_to_rgba(hex_color: str, alpha: float = 0.55) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

def compute_layout(
    G: nx.DiGraph,
    communities: dict[int, list[str]],
    algorithm: str = "spring",
) -> dict[str, tuple[float, float]]:
    """
    Compute 2-D node positions.

    algorithm: "spring" | "community" | "circular" | "kamada_kawai"

    The "community" layout places community centroids evenly around a
    circle, then spreads nodes within each community using a smaller
    spring layout.  This gives a clear visual cluster separation.
    """
    if len(G) == 0:
        return {}

    if algorithm == "community" and communities:
        pos = {}
        n_comm = max(len(communities), 1)
        radius_outer = 3.0
        for idx, (comm_id, members) in enumerate(communities.items()):
            angle = 2 * math.pi * idx / n_comm
            cx = radius_outer * math.cos(angle)
            cy = radius_outer * math.sin(angle)

            sub = G.subgraph([m for m in members if G.has_node(m)])
            if len(sub) == 0:
                continue
            if len(sub) == 1:
                node = list(sub.nodes())[0]
                pos[node] = (cx, cy)
            else:
                try:
                    sub_pos = nx.spring_layout(
                        sub, seed=comm_id,
                        k=0.8 / math.sqrt(len(sub)),
                    )
                except Exception:
                    sub_pos = {n: (0.0, 0.0) for n in sub.nodes()}
                spread = min(1.5, radius_outer / 2)
                for node, (x, y) in sub_pos.items():
                    pos[node] = (cx + x * spread, cy + y * spread)
        return pos

    if algorithm == "kamada_kawai":
        try:
            return nx.kamada_kawai_layout(G)
        except Exception:
            pass

    if algorithm == "circular":
        return nx.circular_layout(G)

    # Default: spring
    k = 1.5 / math.sqrt(max(len(G), 1))
    return nx.spring_layout(G, k=k, iterations=60, seed=42)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Interactive network graph
# ─────────────────────────────────────────────────────────────────────────────

def network_graph(
    G: nx.DiGraph,
    communities: dict[int, list[str]],
    pos: dict[str, tuple[float, float]],
    color_by: str = "classification",   # "classification" | "community" | "bot_score"
    size_by: str = "pagerank",          # "pagerank" | "followers" | "tweets" | "uniform"
    show_edge_labels: bool = False,
    min_edge_weight: int = 1,
    height: int = 650,
) -> go.Figure:
    """
    Build the main interactive node-link graph.

    Clicking a node shows its attributes in a hover card.
    """
    if len(G) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No network data", showarrow=False, font_size=18)
        return fig

    # Community lookup  node → comm_id
    node_community: dict[str, int] = {}
    for comm_id, members in communities.items():
        for m in members:
            node_community[m] = comm_id

    # ── Edge traces (one trace per relation type for legend) ──────────────────
    edge_traces: list[go.Scatter] = []
    relation_edges: dict[str, list] = defaultdict(list)

    for u, v, data in G.edges(data=True):
        if data.get("weight", 1) < min_edge_weight:
            continue
        if u not in pos or v not in pos:
            continue
        rel = next(iter(data.get("relation_types", {"interaction"})), "interaction")
        relation_edges[rel].append((u, v, data.get("weight", 1)))

    rel_style = {
        "mention":    dict(color="rgba(100,116,139,0.35)", dash="solid",  width_scale=1.0),
        "co_hashtag": dict(color="rgba(99,102,241,0.25)",  dash="dot",    width_scale=0.8),
        "retweet":    dict(color="rgba(239,68,68,0.30)",   dash="solid",  width_scale=1.2),
        "reply":      dict(color="rgba(34,197,94,0.30)",   dash="dash",   width_scale=0.9),
        "interaction":dict(color="rgba(148,163,184,0.25)", dash="solid",  width_scale=0.8),
    }

    for rel, edges in relation_edges.items():
        style = rel_style.get(rel, rel_style["interaction"])
        xe, ye = [], []
        for u, v, w in edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            xe += [x0, x1, None]
            ye += [y0, y1, None]

        edge_traces.append(go.Scatter(
            x=xe, y=ye,
            mode="lines",
            name=rel,
            line=dict(color=style["color"], dash=style["dash"], width=1),
            hoverinfo="skip",
            showlegend=True,
        ))

    # ── Node trace ────────────────────────────────────────────────────────────
    node_x, node_y = [], []
    node_colors, node_sizes = [], []
    node_text, node_hover = [], []

    max_pagerank = max(
        (G.nodes[n].get("pagerank", 0) for n in G.nodes()), default=1e-9
    ) or 1e-9
    max_followers = max(
        (G.nodes[n].get("followers_count", 0) for n in G.nodes()), default=1
    ) or 1

    for node in G.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        d = G.nodes[node]
        clf = d.get("classification", "unknown")
        score = float(d.get("bot_score", 0.0))
        comm_id = node_community.get(node, 0)

        # Color
        if color_by == "community":
            node_colors.append(_community_color(comm_id))
        elif color_by == "bot_score":
            node_colors.append(_bot_score_color(score))
        else:  # classification
            node_colors.append(CLF_COLOR.get(clf, CLF_COLOR["unknown"]))

        # Size
        if size_by == "pagerank":
            raw = G.nodes[node].get("pagerank", 0) / max_pagerank
            sz = 8 + raw * 30
        elif size_by == "followers":
            raw = math.log1p(G.nodes[node].get("followers_count", 0)) / math.log1p(max_followers)
            sz = 8 + raw * 28
        elif size_by == "tweets":
            tc = G.nodes[node].get("tweet_count", 0) or 0
            sz = 8 + min(tc / 1000, 1) * 25
        else:
            sz = 12
        node_sizes.append(sz)

        username = d.get("username") or node
        node_text.append(f"@{username}")

        # Hover card
        narratives = ", ".join(d.get("top_narratives", [])[:3]) or "—"
        hover = (
            f"<b>@{username}</b><br>"
            f"Classification: <b>{clf.upper()}</b><br>"
            f"Bot score: <b>{score*100:.1f}%</b><br>"
            f"Community: #{comm_id}<br>"
            f"PageRank: {d.get('pagerank', 0):.5f}<br>"
            f"Betweenness: {d.get('betweenness_centrality', 0):.4f}<br>"
            f"Followers: {d.get('followers_count', 0):,}<br>"
            f"Following: {d.get('following_count', 0):,}<br>"
            f"Degree (in/out): {d.get('in_degree', 0)}/{d.get('out_degree', 0)}<br>"
            f"Top narratives: {narratives}"
        )
        node_hover.append(hover)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=9, color="#1e293b"),
        hovertext=node_hover,
        hoverinfo="text",
        name="Accounts",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1.5, color="#ffffff"),
            opacity=0.92,
        ),
    )

    fig = go.Figure(data=[*edge_traces, node_trace])
    fig.update_layout(
        height=height,
        showlegend=True,
        legend=dict(
            orientation="v", x=1.01, y=1, bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e2e8f0", borderwidth=1,
        ),
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Community sunburst
# ─────────────────────────────────────────────────────────────────────────────

def community_sunburst(
    G: nx.DiGraph,
    communities: dict[int, list[str]],
    top_n_accounts: int = 5,
) -> go.Figure:
    """
    Sunburst: inner ring = communities, outer ring = top accounts per community.
    Colour encodes avg bot score (red = high).
    """
    ids, labels, parents, values, colors, hover = [], [], [], [], [], []

    root = "Network"
    ids.append(root); labels.append(root); parents.append("")
    values.append(len(G)); colors.append("#6366f1"); hover.append("")

    for comm_id, members in sorted(communities.items(), key=lambda x: -len(x[1])):
        valid = [m for m in members if G.has_node(m)]
        if not valid:
            continue
        scores = [G.nodes[m].get("bot_score", 0.0) for m in valid]
        avg = sum(scores) / len(scores)

        cid = f"c{comm_id}"
        n_bots = sum(1 for m in valid if G.nodes[m].get("classification") == "bot")
        n_susp = sum(1 for m in valid if G.nodes[m].get("classification") == "suspected")

        ids.append(cid)
        labels.append(f"Community {comm_id}<br>({len(valid)} accounts)")
        parents.append(root)
        values.append(len(valid))
        colors.append(_bot_score_color(avg))
        hover.append(
            f"<b>Community {comm_id}</b><br>"
            f"Accounts: {len(valid)}<br>"
            f"Avg bot score: {avg*100:.1f}%<br>"
            f"Bots: {n_bots} | Suspected: {n_susp}"
        )

        # Top accounts by PageRank
        top = sorted(valid, key=lambda n: G.nodes[n].get("pagerank", 0), reverse=True)[:top_n_accounts]
        for node in top:
            d = G.nodes[node]
            username = d.get("username", node)
            score = d.get("bot_score", 0.0)
            ids.append(f"c{comm_id}__{node}")
            labels.append(f"@{username}")
            parents.append(cid)
            values.append(1)
            colors.append(_bot_score_color(score))
            hover.append(
                f"@{username}<br>"
                f"Bot score: {score*100:.1f}%<br>"
                f"Classification: {d.get('classification','?')}<br>"
                f"PageRank: {d.get('pagerank',0):.5f}"
            )

    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        marker=dict(colors=colors, line=dict(color="#fff", width=1)),
        hovertext=hover, hoverinfo="text",
        branchvalues="total",
        maxdepth=2,
    ))
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Activity timeline
# ─────────────────────────────────────────────────────────────────────────────

def activity_timeline(
    records: list[dict],
    scores: list[dict],
    bin_hours: int = 6,
) -> go.Figure:
    """
    Stacked area chart: tweet volume over time broken down by
    bot / suspected / human classification.
    """
    score_map = {
        s.get("user_id"): s.get("classification", "unknown")
        for s in scores
    }

    rows = []
    for r in records:
        uid = (r.get("author") or {}).get("id") or r.get("author_id", "")
        ts_str = r.get("created_at")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        except ValueError:
            continue
        clf = score_map.get(uid, "unknown")
        rows.append({"ts": ts, "classification": clf})

    if not rows:
        fig = go.Figure()
        fig.add_annotation(text="No timestamped tweets available", showarrow=False)
        return fig

    df = pd.DataFrame(rows)
    df["bin"] = df["ts"].dt.floor(f"{bin_hours}h")

    pivot = (
        df.groupby(["bin", "classification"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    fig = go.Figure()
    for clf, color in CLF_COLOR.items():
        if clf in pivot.columns:
            fill_color = (
                color.replace(")", ",0.55)").replace("rgb", "rgba")
                if color.startswith("rgb")
                else _hex_to_rgba(color, 0.55)
            )
            fig.add_trace(go.Scatter(
                x=pivot["bin"], y=pivot[clf],
                name=clf.capitalize(),
                mode="lines",
                stackgroup="one",
                fillcolor=fill_color,
                line=dict(color=color, width=1),
                hovertemplate=f"{clf}: %{{y}}<extra></extra>",
            ))

    fig.update_layout(
        title=f"Tweet Activity Timeline (binned every {bin_hours}h)",
        xaxis_title="Time",
        yaxis_title="Tweets",
        height=320,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=10, r=10, t=40, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Coordination heatmap
# ─────────────────────────────────────────────────────────────────────────────

def coordination_heatmap(
    records: list[dict],
    scores: list[dict],
    top_n: int = 30,
) -> go.Figure:
    """
    Heatmap: rows = top accounts by tweet count, columns = hours of day.
    Cell value = tweet count in that hour.
    Highly uniform rows signal robotic, 24/7 automation.
    """
    score_map = {
        s.get("user_id"): s for s in scores
    }

    # Accumulate (user, hour) → count
    data: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        uid = (r.get("author") or {}).get("id") or r.get("author_id", "")
        ts_str = r.get("created_at")
        if not uid or not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            data[uid][ts.hour] += 1
        except ValueError:
            continue

    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No timestamped tweets available", showarrow=False)
        return fig

    # Pick top_n most active accounts
    totals = {uid: sum(h.values()) for uid, h in data.items()}
    top_uids = sorted(totals, key=lambda u: totals[u], reverse=True)[:top_n]

    matrix = []
    ylabels = []
    for uid in top_uids:
        row = [data[uid].get(h, 0) for h in range(24)]
        matrix.append(row)
        s = score_map.get(uid, {})
        username = (s.get("author") or {}).get("username", uid[:8])
        clf = s.get("classification", "?")[0].upper()  # B / S / H / ?
        ylabels.append(f"[{clf}] @{username}")

    z = np.array(matrix)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"{h:02d}:00" for h in range(24)],
        y=ylabels,
        colorscale=[
            [0.0,  "#f0fdf4"],
            [0.3,  "#fef9c3"],
            [0.6,  "#fed7aa"],
            [1.0,  "#ef4444"],
        ],
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>Hour %{x}: %{z} tweets<extra></extra>",
        showscale=True,
        colorbar=dict(title="Tweets", thickness=12),
    ))
    fig.update_layout(
        title=f"Posting Hour Distribution (top {len(top_uids)} accounts)<br>"
              "<sup>[B]=bot  [S]=suspected  [H]=human</sup>",
        xaxis_title="Hour of day (UTC)",
        yaxis=dict(autorange="reversed"),
        height=max(250, 20 * len(top_uids) + 80),
        margin=dict(l=10, r=80, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Community stats bar chart
# ─────────────────────────────────────────────────────────────────────────────

def community_stats_bars(
    community_summaries: list[dict],
) -> go.Figure:
    """
    Grouped bar: one group per community showing size and avg bot score.
    """
    if not community_summaries:
        fig = go.Figure()
        fig.add_annotation(text="No communities detected", showarrow=False)
        return fig

    summaries = sorted(community_summaries, key=lambda c: c["size"], reverse=True)[:15]
    labels = [f"Community {c['community_id']}" for c in summaries]
    sizes  = [c["size"] for c in summaries]
    scores = [c["avg_bot_score_pct"] for c in summaries]
    colors = [_community_color(c["community_id"]) for c in summaries]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accounts in community",
        x=labels, y=sizes,
        marker_color=colors,
        yaxis="y",
        offsetgroup=1,
        text=sizes, textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Avg bot score %",
        x=labels, y=scores,
        marker_color=["#ef4444" if s >= 60 else "#f59e0b" if s >= 35 else "#22c55e"
                      for s in scores],
        yaxis="y2",
        offsetgroup=2,
        text=[f"{s:.0f}%" for s in scores], textposition="outside",
    ))
    fig.update_layout(
        title="Community Summary",
        barmode="group",
        yaxis=dict(title="# Accounts", showgrid=True, gridcolor="#e2e8f0"),
        yaxis2=dict(title="Avg Bot Score %", overlaying="y", side="right", range=[0, 120]),
        height=360,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=40, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Top accounts DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def top_accounts_dataframe(
    G: nx.DiGraph,
    communities: dict[int, list[str]],
    n: int = 50,
) -> pd.DataFrame:
    """Build a sortable DataFrame of top accounts by PageRank."""
    node_community = {}
    for comm_id, members in communities.items():
        for m in members:
            node_community[m] = comm_id

    rows = []
    for node in G.nodes():
        d = G.nodes[node]
        clf = d.get("classification", "unknown")
        score = d.get("bot_score", 0.0)
        rows.append({
            "username":     "@" + (d.get("username") or node),
            "class":        clf,
            "bot_score_%":  round(score * 100, 1),
            "community":    node_community.get(node, "?"),
            "pagerank":     round(d.get("pagerank", 0), 6),
            "betweenness":  round(d.get("betweenness_centrality", 0), 5),
            "in_degree":    d.get("in_degree", 0),
            "out_degree":   d.get("out_degree", 0),
            "followers":    d.get("followers_count", 0),
            "tweets":       d.get("tweet_count", 0),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("pagerank", ascending=False).head(n).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Narrative co-occurrence chord / bar
# ─────────────────────────────────────────────────────────────────────────────

def narrative_bar(
    G: nx.DiGraph,
    top_n: int = 15,
) -> go.Figure:
    """Horizontal bar of the most-amplified disinformation narratives."""
    counter: Counter = Counter()
    for node in G.nodes():
        for narrative in G.nodes[node].get("top_narratives", []):
            clf = G.nodes[node].get("classification", "unknown")
            weight = 2 if clf == "bot" else 1.2 if clf == "suspected" else 0.5
            counter[narrative] += weight

    if not counter:
        fig = go.Figure()
        fig.add_annotation(text="No narratives detected in this dataset", showarrow=False)
        return fig

    items = counter.most_common(top_n)
    names  = [i[0] for i in reversed(items)]
    values = [i[1] for i in reversed(items)]
    colors = [
        "#ef4444" if v >= max(values) * 0.6 else
        "#f59e0b" if v >= max(values) * 0.3 else "#6366f1"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        y=names, x=values,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Most-Amplified Disinformation Narratives<br><sup>(weighted by account classification)</sup>",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=max(300, top_n * 26 + 80),
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. Helper: build graph + compute everything in one call
# ─────────────────────────────────────────────────────────────────────────────

def build_full_network(
    records: list[dict],
    scores: list[dict],
    config: dict,
) -> tuple[nx.DiGraph, dict[int, list[str]], list[dict]]:
    """
    Build graph, compute metrics, detect communities, describe them.

    Returns (G, communities_dict, community_summaries_list).
    """
    from src.network.graph_builder import GraphBuilder
    from src.network.community_detector import CommunityDetector

    builder = GraphBuilder(config)
    G = builder.build(records, scores)

    if len(G) == 0:
        return G, {}, []

    builder.compute_metrics(G)

    detector = CommunityDetector(config)
    communities = detector.detect(G)

    # Attach community_id to each node
    for comm_id, members in communities.items():
        for m in members:
            if G.has_node(m):
                G.nodes[m]["community_id"] = comm_id

    summaries = detector.describe(G, communities)
    return G, communities, summaries
