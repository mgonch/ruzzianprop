"""
Network visualizer.

Produces two complementary outputs:
  1. Static PNG/SVG via Matplotlib   – suitable for reports and print
  2. Interactive HTML via PyVis      – zoomable, hoverable, filterable

Node appearance
---------------
  Size    : proportional to PageRank (influence) within the network
  Color   : classification (bot=red, suspected=orange, human=green, unknown=grey)
  Border  : thicker border for high disinformation scores

Edge appearance
---------------
  Width   : proportional to interaction weight
  Color   : grey (co_hashtag) vs blue (mention/retweet)
"""

import os
import json
import logging
from pathlib import Path

import networkx as nx
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex
import numpy as np

logger = logging.getLogger(__name__)

# Classification → hex colour
CLASSIFICATION_COLORS = {
    "bot": "#e74c3c",        # Red
    "suspected": "#e67e22",  # Orange
    "human": "#2ecc71",      # Green
    "unknown": "#95a5a6",    # Grey
}


class NetworkVisualizer:
    """Render bot network graphs as static images and interactive HTML."""

    def __init__(self, config: dict | None = None, output_dir: str = "./output"):
        cfg = (config or {}).get("visualization", {})
        self.output_dir = Path(cfg.get("output_dir", output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.static_format = cfg.get("static_format", "png")
        self.interactive = cfg.get("interactive", True)
        colors = cfg.get("color_scheme", {})
        self.colors = {**CLASSIFICATION_COLORS, **colors}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def render_all(
        self,
        G: nx.DiGraph,
        communities: dict[int, list[str]] | None = None,
        prefix: str = "network",
    ) -> dict[str, str]:
        """
        Render both static and interactive visualizations.

        Returns dict of {output_type: file_path}.
        """
        outputs: dict[str, str] = {}

        static_path = self.render_static(G, communities, prefix)
        outputs["static"] = static_path

        if self.interactive:
            html_path = self.render_interactive(G, communities, prefix)
            outputs["interactive"] = html_path

        return outputs

    def render_static(
        self,
        G: nx.DiGraph,
        communities: dict[int, list[str]] | None = None,
        prefix: str = "network",
    ) -> str:
        """Render a static Matplotlib figure and save it."""
        if len(G) == 0:
            logger.warning("Graph is empty – skipping static render")
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(22, 11))
        fig.patch.set_facecolor("#1a1a2e")

        # Panel 1: full network coloured by bot classification
        ax1 = axes[0]
        self._draw_network(G, ax1, color_by="classification")
        ax1.set_title("Bot Classification Network", color="white", fontsize=14, pad=10)
        self._add_classification_legend(ax1)

        # Panel 2: community view
        ax2 = axes[1]
        self._draw_network(G, ax2, color_by="community")
        ax2.set_title("Community Structure", color="white", fontsize=14, pad=10)
        if communities:
            self._add_community_legend(ax2, communities, G)

        plt.tight_layout(pad=2.0)
        out_path = str(self.output_dir / f"{prefix}.{self.static_format}")
        plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Static network saved to %s", out_path)
        return out_path

    def render_interactive(
        self,
        G: nx.DiGraph,
        communities: dict[int, list[str]] | None = None,
        prefix: str = "network",
    ) -> str:
        """Render an interactive HTML network using PyVis."""
        try:
            from pyvis.network import Network
        except ImportError:
            logger.warning("PyVis not installed; skipping interactive render.")
            return ""

        if len(G) == 0:
            logger.warning("Graph is empty – skipping interactive render")
            return ""

        net = Network(
            height="900px",
            width="100%",
            bgcolor="#1a1a2e",
            font_color="white",
            directed=True,
        )
        net.barnes_hut(
            gravity=-8000,
            central_gravity=0.3,
            spring_length=150,
            spring_strength=0.001,
            damping=0.09,
        )

        # Compute layout once for consistent placement
        undirected = G.to_undirected()
        try:
            pos = nx.spring_layout(undirected, seed=42, k=2.0 / (len(G) ** 0.5))
        except Exception:
            pos = {n: (0, 0) for n in G.nodes()}

        max_pr = max(
            (G.nodes[n].get("pagerank", 0.0) for n in G.nodes()), default=1e-9
        )
        max_pr = max(max_pr, 1e-9)

        for node_id in G.nodes():
            attrs = G.nodes[node_id]
            classification = attrs.get("classification", "unknown")
            color = self.colors.get(classification, "#95a5a6")
            username = attrs.get("username", node_id)
            bot_pct = attrs.get("bot_score_pct", 0.0)
            dis_score = attrs.get("avg_dis_score", 0.0)

            # Size by PageRank (range 15–60px)
            pr = attrs.get("pagerank", 0.0)
            size = 15 + 45 * (pr / max_pr)

            tooltip = (
                f"@{username}\n"
                f"Bot Score: {bot_pct:.1f}%\n"
                f"Classification: {classification}\n"
                f"Dis. Score: {dis_score:.2%}\n"
                f"Followers: {attrs.get('followers_count', 0):,}\n"
                f"Following: {attrs.get('following_count', 0):,}\n"
                f"Tweets: {attrs.get('tweet_count', 0):,}\n"
                f"Age: {attrs.get('account_age_days', '?')} days\n"
                f"Community: {attrs.get('community_id', '?')}\n"
                f"PageRank: {pr:.6f}\n"
                f"Betweenness: {attrs.get('betweenness_centrality', 0):.4f}"
            )

            # Border thickness by dis score
            border_width = 1 + int(dis_score * 5)

            net.add_node(
                node_id,
                label=f"@{username}",
                title=tooltip,
                color={
                    "background": color,
                    "border": "#f1c40f" if dis_score > 0.3 else color,
                },
                size=size,
                borderWidth=border_width,
                x=float(pos.get(node_id, (0, 0))[0]) * 1000,
                y=float(pos.get(node_id, (0, 0))[1]) * 1000,
            )

        for src, tgt, data in G.edges(data=True):
            weight = data.get("weight", 1)
            rel_types = data.get("relation_types", [])
            color = "#3498db" if "mention" in rel_types else "#7f8c8d"
            net.add_edge(
                src,
                tgt,
                width=min(1 + weight * 0.5, 8),
                color=color,
                title=f"Weight: {weight}\nTypes: {', '.join(rel_types)}",
                arrows="to",
            )

        # Inject custom legend HTML
        legend_html = self._build_html_legend()

        out_path = str(self.output_dir / f"{prefix}.html")
        net.save_graph(out_path)

        # Inject legend into the HTML file
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace("</body>", legend_html + "\n</body>")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info("Interactive network saved to %s", out_path)
        return out_path

    def render_similarity_heatmap(
        self, scores: list[dict], prefix: str = "similarity"
    ) -> str:
        """
        Render a heatmap of bot signal similarity between top-scoring accounts.
        """
        if not scores:
            return ""

        top = scores[:30]  # Limit for readability
        usernames = [s.get("username", s.get("user_id", "?")) for s in top]
        signal_keys = list((top[0].get("signals") or {}).keys())

        data = np.array([
            [s.get("signals", {}).get(k, 0.0) for k in signal_keys]
            for s in top
        ])

        fig, ax = plt.subplots(figsize=(len(signal_keys) * 1.1 + 2, len(top) * 0.5 + 2))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        im = ax.imshow(data, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Signal Score")

        ax.set_xticks(range(len(signal_keys)))
        ax.set_xticklabels(
            [k.replace("_", "\n") for k in signal_keys],
            color="white", fontsize=8, rotation=0,
        )
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(
            [f"@{u}" for u in usernames],
            color="white", fontsize=8,
        )
        ax.set_title("Bot Detection Signal Heatmap", color="white", fontsize=13, pad=12)

        # Annotate cells
        for i in range(len(top)):
            for j in range(len(signal_keys)):
                ax.text(
                    j, i, f"{data[i, j]:.1f}",
                    ha="center", va="center",
                    color="white" if data[i, j] > 0.5 else "black",
                    fontsize=6,
                )

        plt.tight_layout()
        out_path = str(self.output_dir / f"{prefix}_heatmap.{self.static_format}")
        plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Similarity heatmap saved to %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_network(
        self,
        G: nx.DiGraph,
        ax: plt.Axes,
        color_by: str = "classification",
    ) -> None:
        ax.set_facecolor("#16213e")

        if len(G) == 0:
            ax.text(0.5, 0.5, "No data", color="white", ha="center", va="center")
            return

        undirected = G.to_undirected()

        try:
            pos = nx.spring_layout(undirected, seed=42, k=2.5 / (len(G) ** 0.5))
        except Exception:
            pos = nx.random_layout(G, seed=42)

        # Node sizes by PageRank
        max_pr = max(
            (G.nodes[n].get("pagerank", 0.0) for n in G.nodes()), default=1e-9
        )
        max_pr = max(max_pr, 1e-9)
        node_sizes = [
            max(80, 2000 * G.nodes[n].get("pagerank", 0.0) / max_pr)
            for n in G.nodes()
        ]

        # Node colours
        if color_by == "classification":
            node_colors = [
                self.colors.get(G.nodes[n].get("classification", "unknown"), "#95a5a6")
                for n in G.nodes()
            ]
        else:
            # Community colours
            all_comm_ids = sorted(
                set(G.nodes[n].get("community_id", -1) for n in G.nodes())
            )
            community_palette = plt.cm.tab20.colors
            comm_color_map = {
                cid: to_hex(community_palette[i % len(community_palette)])
                for i, cid in enumerate(all_comm_ids)
            }
            node_colors = [
                comm_color_map.get(G.nodes[n].get("community_id", -1), "#95a5a6")
                for n in G.nodes()
            ]

        # Edge widths
        edge_widths = [
            min(0.3 + 0.3 * G[u][v].get("weight", 1), 3.0)
            for u, v in G.edges()
        ]

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=edge_widths,
            edge_color="#4a4a6a",
            alpha=0.5,
            arrows=False,
        )
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.85,
            linewidths=0.5,
            edgecolors="#ffffff40",
        )

        # Labels only for high-PageRank nodes to avoid clutter
        pr_threshold = sorted(
            (G.nodes[n].get("pagerank", 0.0) for n in G.nodes()), reverse=True
        )[:min(20, len(G))][-1] if G.nodes() else 0

        label_nodes = {
            n: f"@{G.nodes[n].get('username', n)}"
            for n in G.nodes()
            if G.nodes[n].get("pagerank", 0.0) >= pr_threshold
        }
        nx.draw_networkx_labels(
            G, pos, labels=label_nodes, ax=ax,
            font_size=6, font_color="white", font_weight="bold",
        )
        ax.axis("off")

    def _add_classification_legend(self, ax: plt.Axes) -> None:
        patches = [
            mpatches.Patch(color=color, label=label.capitalize())
            for label, color in self.colors.items()
        ]
        ax.legend(
            handles=patches, loc="lower left",
            facecolor="#0f3460", labelcolor="white",
            fontsize=8, framealpha=0.8,
        )

    def _add_community_legend(
        self, ax: plt.Axes, communities: dict[int, list[str]], G: nx.DiGraph
    ) -> None:
        palette = plt.cm.tab20.colors
        patches = []
        for i, (cid, nodes) in enumerate(sorted(communities.items())[:10]):
            size = len([n for n in nodes if G.has_node(n)])
            patches.append(
                mpatches.Patch(
                    color=to_hex(palette[i % len(palette)]),
                    label=f"C{cid} ({size} accounts)",
                )
            )
        if patches:
            ax.legend(
                handles=patches, loc="lower left",
                facecolor="#0f3460", labelcolor="white",
                fontsize=8, framealpha=0.8,
            )

    @staticmethod
    def _build_html_legend() -> str:
        return """
<div id="legend" style="
  position:fixed; bottom:20px; left:20px; z-index:9999;
  background:rgba(10,10,30,0.9); border-radius:8px;
  padding:12px 16px; color:white; font-family:sans-serif; font-size:13px;
  border:1px solid #334;
">
  <b style="font-size:14px;">Classification</b><br><br>
  <span style="color:#e74c3c;">&#9679;</span> Bot (&ge;60%)<br>
  <span style="color:#e67e22;">&#9679;</span> Suspected (&ge;40%)<br>
  <span style="color:#2ecc71;">&#9679;</span> Human (&lt;40%)<br>
  <span style="color:#95a5a6;">&#9679;</span> Unknown<br><br>
  <b>Node size</b>: PageRank influence<br>
  <b>Gold border</b>: High disinfo score<br>
  <b>Blue edge</b>: Mention/Retweet<br>
  <b>Grey edge</b>: Shared hashtag
</div>
"""
