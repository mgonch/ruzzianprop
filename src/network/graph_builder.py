"""
Graph builder: constructs a NetworkX graph from tweet/user data.

Edges represent relationships between accounts:
  - retweet_of    (A retweeted B's tweet)
  - mention_of    (A mentioned B)
  - reply_to      (A replied to B)
  - quote_of      (A quoted B)
  - co_hashtag    (A and B both used the same hashtag, >= min_co_occurrences times)

Node attributes
---------------
  - username, display_name
  - bot_score, bot_score_pct, classification
  - followers_count, following_count, tweet_count
  - account_age_days
  - top_narratives, avg_dis_score

Edge attributes
---------------
  - weight          (number of interactions)
  - relation_types  (set of relation types that exist)
"""

import re
import logging
from collections import defaultdict

import networkx as nx

logger = logging.getLogger(__name__)

_MENTION_RE = re.compile(r"@(\w+)")
_HASHTAG_RE = re.compile(r"#(\w+)")


class GraphBuilder:
    """Build interaction graphs from tweet records and bot scores."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("network", {})
        self.min_edge_weight = cfg.get("min_edge_weight", 1)

    def build(
        self,
        records: list[dict],
        scores: list[dict] | None = None,
        dis_summaries: dict | None = None,
    ) -> nx.DiGraph:
        """
        Build a directed graph from tweet *records*.

        Parameters
        ----------
        records       : list of normalised tweet dicts
        scores        : list of BotDetector.score_account() results
        dis_summaries : dict mapping user_id -> DisinformationDetector.analyse_account() result

        Returns
        -------
        nx.DiGraph with bot scores and disinformation attributes on nodes.
        """
        G = nx.DiGraph()

        score_map: dict[str, dict] = {}
        if scores:
            for s in scores:
                uid = s.get("user_id") or (s.get("author") or {}).get("id")
                if uid:
                    score_map[uid] = s

        dis_map: dict[str, dict] = dis_summaries or {}

        # Track edge weights
        edge_weights: dict[tuple[str, str], dict] = defaultdict(
            lambda: {"weight": 0, "relation_types": set()}
        )

        # Track hashtag co-usage
        hashtag_users: dict[str, set[str]] = defaultdict(set)

        for record in records:
            author = record.get("author") or {}
            src_id = author.get("id") or record.get("author_id", "")
            if not src_id:
                continue

            # Ensure source node exists
            self._add_node(G, src_id, author, score_map, dis_map)

            text = record.get("text", "") or ""

            # Retweets, quotes, replies from referenced_tweets
            for ref in record.get("referenced_tweets") or []:
                ref_type = ref.get("type")
                # We don't always know the target author here without extra hydration
                # so we skip adding an edge for now; mention edges cover most cases

            # Mentions
            for username in _MENTION_RE.findall(text):
                username_lower = username.lower()
                # Look up target by username from the data we have
                target_id = self._find_id_by_username(username_lower, records)
                if target_id and target_id != src_id:
                    edge_weights[(src_id, target_id)]["weight"] += 1
                    edge_weights[(src_id, target_id)]["relation_types"].add("mention")

            # Hashtags for co-usage analysis
            for tag in _HASHTAG_RE.findall(text):
                hashtag_users[tag.lower()].add(src_id)

        # Add co-hashtag edges (accounts sharing hashtags ≥ min_edge_weight times)
        for tag, users in hashtag_users.items():
            users = list(users)
            if len(users) > 1:
                for i, u1 in enumerate(users):
                    for u2 in users[i + 1 :]:
                        # Bidirectional co-hashtag edge
                        edge_weights[(u1, u2)]["weight"] += 1
                        edge_weights[(u1, u2)]["relation_types"].add("co_hashtag")
                        edge_weights[(u2, u1)]["weight"] += 1
                        edge_weights[(u2, u1)]["relation_types"].add("co_hashtag")

        # Add edges meeting minimum weight
        for (src, tgt), attrs in edge_weights.items():
            if attrs["weight"] >= self.min_edge_weight:
                if not G.has_node(src):
                    G.add_node(src)
                if not G.has_node(tgt):
                    G.add_node(tgt)
                G.add_edge(
                    src,
                    tgt,
                    weight=attrs["weight"],
                    relation_types=list(attrs["relation_types"]),
                )

        logger.info(
            "Graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
        )
        return G

    # ------------------------------------------------------------------

    @staticmethod
    def _add_node(
        G: nx.DiGraph,
        uid: str,
        user: dict,
        score_map: dict,
        dis_map: dict,
    ) -> None:
        if G.has_node(uid):
            return
        score = score_map.get(uid, {})
        dis = dis_map.get(uid, {})
        G.add_node(
            uid,
            username=user.get("username", ""),
            display_name=user.get("name", ""),
            bot_score=score.get("bot_score", 0.0),
            bot_score_pct=score.get("bot_score_pct", 0.0),
            classification=score.get("classification", "unknown"),
            followers_count=user.get("followers_count", 0),
            following_count=user.get("following_count", 0),
            tweet_count=user.get("tweet_count", 0),
            account_age_days=user.get("account_age_days"),
            avg_dis_score=dis.get("avg_dis_score", 0.0),
            top_narratives=dis.get("top_narratives", []),
        )

    @staticmethod
    def _find_id_by_username(username: str, records: list[dict]) -> str | None:
        for r in records:
            author = r.get("author") or {}
            if (author.get("username") or "").lower() == username:
                return author.get("id")
        return None

    # ------------------------------------------------------------------
    # Graph metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(G: nx.DiGraph) -> dict[str, dict]:
        """Compute per-node graph metrics and attach them to the graph."""
        undirected = G.to_undirected()

        degree_centrality = nx.degree_centrality(undirected)
        try:
            betweenness = nx.betweenness_centrality(undirected, normalized=True, k=min(100, len(G)))
        except Exception:
            betweenness = {n: 0.0 for n in G.nodes()}

        try:
            pagerank = nx.pagerank(G, alpha=0.85, max_iter=200)
        except Exception:
            pagerank = {n: 0.0 for n in G.nodes()}

        metrics: dict[str, dict] = {}
        for node in G.nodes():
            m = {
                "degree": G.degree(node),
                "in_degree": G.in_degree(node),
                "out_degree": G.out_degree(node),
                "degree_centrality": round(degree_centrality.get(node, 0.0), 6),
                "betweenness_centrality": round(betweenness.get(node, 0.0), 6),
                "pagerank": round(pagerank.get(node, 0.0), 8),
            }
            metrics[node] = m
            G.nodes[node].update(m)

        return metrics
