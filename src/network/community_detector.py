"""
Community detection using the Louvain method.

Identifies tightly-knit clusters of accounts within the interaction graph.
Each community is described by:
  - id
  - size
  - avg bot score
  - dominant narratives
  - top accounts (by PageRank or bot score)
  - bridge accounts (high betweenness centrality between communities)
"""

import logging
from collections import Counter, defaultdict

import networkx as nx

try:
    import community as community_louvain  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    logging.warning(
        "python-louvain not installed. Community detection will use a basic "
        "connected-components fallback. Install with: pip install python-louvain"
    )

logger = logging.getLogger(__name__)


class CommunityDetector:
    """Detect and describe communities within an account interaction graph."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("network", {})
        self.resolution = cfg.get("community_resolution", 1.0)

    def detect(self, G: nx.DiGraph) -> dict[int, list[str]]:
        """
        Return a mapping of {community_id: [node_id, ...]} for all nodes in G.
        """
        undirected = G.to_undirected()

        if LOUVAIN_AVAILABLE and len(G) > 0:
            partition = community_louvain.best_partition(
                undirected, resolution=self.resolution
            )
            # partition maps node -> community_id
            communities: dict[int, list[str]] = defaultdict(list)
            for node, comm_id in partition.items():
                communities[comm_id].append(node)
            logger.info(
                "Louvain detected %d communities in %d-node graph",
                len(communities),
                len(G),
            )
        else:
            # Fallback: connected components
            communities = {}
            for i, component in enumerate(nx.connected_components(undirected)):
                communities[i] = list(component)
            logger.info(
                "Connected-components found %d communities in %d-node graph",
                len(communities),
                len(G),
            )

        return dict(communities)

    def describe(
        self, G: nx.DiGraph, communities: dict[int, list[str]]
    ) -> list[dict]:
        """
        Return a list of community summary dicts, sorted by size descending.
        """
        summaries = []
        for comm_id, nodes in communities.items():
            subgraph_nodes = [n for n in nodes if G.has_node(n)]
            if not subgraph_nodes:
                continue

            node_data = [G.nodes[n] for n in subgraph_nodes]

            scores = [d.get("bot_score", 0.0) for d in node_data]
            avg_bot_score = sum(scores) / len(scores) if scores else 0.0

            classifications = Counter(d.get("classification", "unknown") for d in node_data)

            all_narratives = []
            for d in node_data:
                all_narratives.extend(d.get("top_narratives", []))
            top_narratives = [n for n, _ in Counter(all_narratives).most_common(5)]

            # Top accounts by PageRank within community
            top_accounts = sorted(
                subgraph_nodes,
                key=lambda n: G.nodes[n].get("pagerank", 0.0),
                reverse=True,
            )[:5]

            # Bridge accounts (high betweenness)
            bridge_accounts = sorted(
                subgraph_nodes,
                key=lambda n: G.nodes[n].get("betweenness_centrality", 0.0),
                reverse=True,
            )[:3]

            summaries.append({
                "community_id": comm_id,
                "size": len(subgraph_nodes),
                "avg_bot_score": round(avg_bot_score, 4),
                "avg_bot_score_pct": round(avg_bot_score * 100, 1),
                "classifications": dict(classifications),
                "top_narratives": top_narratives,
                "top_accounts": [
                    {
                        "id": n,
                        "username": G.nodes[n].get("username", ""),
                        "bot_score_pct": G.nodes[n].get("bot_score_pct", 0.0),
                        "pagerank": G.nodes[n].get("pagerank", 0.0),
                    }
                    for n in top_accounts
                ],
                "bridge_accounts": [
                    {
                        "id": n,
                        "username": G.nodes[n].get("username", ""),
                        "betweenness_centrality": G.nodes[n].get("betweenness_centrality", 0.0),
                    }
                    for n in bridge_accounts
                ],
            })

        summaries.sort(key=lambda s: s["size"], reverse=True)
        return summaries

    def assign_to_graph(
        self, G: nx.DiGraph, communities: dict[int, list[str]]
    ) -> None:
        """Attach community_id attribute to each node in G."""
        for comm_id, nodes in communities.items():
            for node in nodes:
                if G.has_node(node):
                    G.nodes[node]["community_id"] = comm_id
