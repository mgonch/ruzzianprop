"""Tests for the GraphBuilder and CommunityDetector."""

import pytest
import networkx as nx
from src.network.graph_builder import GraphBuilder
from src.network.community_detector import CommunityDetector


def make_record(author_id: str, username: str, text: str,
                created_at: str = "2024-01-01T12:00:00+00:00") -> dict:
    return {
        "id": f"tweet_{author_id}",
        "text": text,
        "author_id": author_id,
        "created_at": created_at,
        "referenced_tweets": [],
        "entities": {},
        "author": {
            "id": author_id,
            "username": username,
            "name": f"User {username}",
            "account_age_days": 365,
            "description": "Test user",
            "location": "NYC",
            "profile_image_url": "https://pic.jpg",
            "verified": False,
            "protected": False,
            "followers_count": 100,
            "following_count": 50,
            "tweet_count": 500,
            "listed_count": 2,
        },
    }


class TestGraphBuilder:
    def test_nodes_created_from_authors(self):
        builder = GraphBuilder()
        records = [
            make_record("u1", "alice", "Hello world"),
            make_record("u2", "bob", "Hi there"),
        ]
        G = builder.build(records)
        assert G.has_node("u1")
        assert G.has_node("u2")

    def test_mention_creates_edge(self):
        builder = GraphBuilder(config={"network": {"min_edge_weight": 1}})
        records = [
            make_record("u1", "alice", "Hello @bob how are you?"),
            make_record("u2", "bob", "Just tweeting"),
        ]
        G = builder.build(records)
        assert G.has_edge("u1", "u2")

    def test_co_hashtag_creates_edge(self):
        builder = GraphBuilder(config={"network": {"min_edge_weight": 1}})
        records = [
            make_record("u1", "alice", "Check out #testhashtag today"),
            make_record("u2", "bob", "I love #testhashtag too"),
        ]
        G = builder.build(records)
        # co-hashtag edges should exist between u1 and u2
        assert G.has_edge("u1", "u2") or G.has_edge("u2", "u1")

    def test_node_has_bot_score_from_scores(self):
        builder = GraphBuilder()
        records = [make_record("u1", "alice", "Hello")]
        scores = [{"user_id": "u1", "bot_score": 0.8, "bot_score_pct": 80.0, "classification": "bot"}]
        G = builder.build(records, scores=scores)
        assert G.nodes["u1"]["bot_score"] == 0.8
        assert G.nodes["u1"]["classification"] == "bot"

    def test_empty_records_empty_graph(self):
        builder = GraphBuilder()
        G = builder.build([])
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0

    def test_compute_metrics(self):
        builder = GraphBuilder()
        records = [
            make_record("u1", "alice", "Hello @bob"),
            make_record("u2", "bob", "@alice hi back"),
        ]
        G = builder.build(records, scores=[])
        metrics = builder.compute_metrics(G)
        assert "u1" in metrics or "u2" in metrics
        for node in G.nodes():
            assert "degree" in G.nodes[node]
            assert "pagerank" in G.nodes[node]


class TestCommunityDetector:
    def test_detect_returns_nonempty_for_connected_graph(self):
        G = nx.DiGraph()
        G.add_nodes_from(["u1", "u2", "u3"])
        G.add_edges_from([("u1", "u2"), ("u2", "u3")])
        for n in G.nodes():
            G.nodes[n].update({
                "bot_score": 0.5, "bot_score_pct": 50.0,
                "classification": "suspected", "top_narratives": [],
                "pagerank": 0.33, "betweenness_centrality": 0.0,
            })
        detector = CommunityDetector()
        communities = detector.detect(G)
        assert len(communities) >= 1
        total_nodes = sum(len(v) for v in communities.values())
        assert total_nodes == 3

    def test_assign_community_id_to_nodes(self):
        G = nx.DiGraph()
        G.add_nodes_from(["u1", "u2"])
        for n in G.nodes():
            G.nodes[n].update({
                "bot_score": 0.0, "bot_score_pct": 0.0,
                "classification": "human", "top_narratives": [],
                "pagerank": 0.5, "betweenness_centrality": 0.0,
            })
        detector = CommunityDetector()
        communities = detector.detect(G)
        detector.assign_to_graph(G, communities)
        for node in G.nodes():
            assert "community_id" in G.nodes[node]

    def test_describe_returns_list(self):
        G = nx.DiGraph()
        G.add_node("u1", bot_score=0.9, bot_score_pct=90.0, classification="bot",
                   top_narratives=["nato_narratives"], pagerank=0.5,
                   betweenness_centrality=0.1, avg_dis_score=0.3, username="testbot")
        communities = {0: ["u1"]}
        detector = CommunityDetector()
        summaries = detector.describe(G, communities)
        assert len(summaries) == 1
        assert summaries[0]["avg_bot_score_pct"] == 90.0
