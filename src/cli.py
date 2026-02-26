"""
RuzzianProp CLI – Russian disinformation bot network analyser.

Usage examples
--------------
# Analyse from a JSON data file
ruzzianprop analyse --input data/sample.json --output ./output

# Search Twitter/X API live
ruzzianprop analyse --query "NATO aggression" --max-results 500 --output ./output

# Run demo with synthetic data
ruzzianprop demo

# Show scores only (no network graph)
ruzzianprop analyse --input data/sample.json --no-graph
"""

import json
import logging
import os
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    p = Path(config_path)
    if p.exists():
        with open(p, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


# ------------------------------------------------------------------
# CLI root
# ------------------------------------------------------------------

@click.group()
@click.version_option("0.1.0", prog_name="ruzzianprop")
def cli():
    """RuzzianProp – Russian disinformation bot network analyser."""
    pass


# ------------------------------------------------------------------
# analyse command
# ------------------------------------------------------------------

@cli.command()
@click.option("--input", "-i", "input_path", default=None,
              help="Path to a JSON or CSV file with tweet data.")
@click.option("--query", "-q", default=None,
              help="Twitter search query (requires API credentials).")
@click.option("--max-results", "-n", default=200, show_default=True,
              help="Maximum number of tweets to fetch from the API.")
@click.option("--output", "-o", default="./output", show_default=True,
              help="Output directory for reports and visualizations.")
@click.option("--config", "-c", default="config/settings.yaml", show_default=True,
              help="Path to YAML configuration file.")
@click.option("--no-graph", is_flag=True, default=False,
              help="Skip network graph generation.")
@click.option("--no-interactive", is_flag=True, default=False,
              help="Skip interactive HTML graph generation.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Enable verbose logging.")
def analyse(input_path, query, max_results, output, config, no_graph, no_interactive, verbose):
    """Analyse a dataset for Russian disinformation bots."""
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    cfg = load_config(config)

    if not input_path and not query:
        console.print("[red]Error:[/] Provide --input or --query.", style="bold")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------------
    console.print(Panel("[bold cyan]RuzzianProp[/bold cyan] – Disinformation Bot Analyser", expand=False))

    records: list[dict] = []

    if input_path:
        from src.collectors.file_collector import FileCollector
        console.print(f"[cyan]Loading data from[/cyan] {input_path}...")
        collector = FileCollector(cfg)
        records = collector.load(input_path)
        console.print(f"  [green]Loaded {len(records):,} records[/green]")

    if query:
        from src.collectors.twitter_collector import TwitterCollector
        console.print(f"[cyan]Querying Twitter API:[/cyan] {query!r} (max {max_results})")
        try:
            collector = TwitterCollector(cfg)
            api_records = collector.search_recent(query, max_results=max_results)
            records.extend(api_records)
            console.print(f"  [green]Fetched {len(api_records):,} tweets[/green]")
        except EnvironmentError as e:
            console.print(f"[red]Twitter API error:[/red] {e}")
            sys.exit(1)

    if not records:
        console.print("[yellow]No records to analyse.[/yellow]")
        sys.exit(0)

    # ----------------------------------------------------------------
    # 2. Bot detection
    # ----------------------------------------------------------------
    console.print("\n[cyan]Running bot detection...[/cyan]")
    from src.detectors.bot_detector import BotDetector
    from src.detectors.disinformation_detector import DisinformationDetector

    keywords = (cfg.get("disinformation") or {}).get("keywords", {})
    detector = BotDetector(config=cfg, keywords=keywords)
    scores = detector.score_all(records)

    _print_scores_table(scores)

    # ----------------------------------------------------------------
    # 3. Disinformation content analysis
    # ----------------------------------------------------------------
    console.print("\n[cyan]Analysing disinformation content...[/cyan]")
    dis_detector = DisinformationDetector(config=cfg)
    corpus_summary = dis_detector.analyse_corpus(records)
    _print_disinfo_summary(corpus_summary)

    # Per-account dis summaries for graph annotation
    from collections import defaultdict
    by_author: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        uid = (r.get("author") or {}).get("id") or r.get("author_id", "")
        by_author[uid].append(r)
    dis_summaries = {uid: dis_detector.analyse_account(tweets) for uid, tweets in by_author.items()}

    # ----------------------------------------------------------------
    # 4. Network analysis
    # ----------------------------------------------------------------
    if not no_graph:
        console.print("\n[cyan]Building interaction network...[/cyan]")
        from src.network.graph_builder import GraphBuilder
        from src.network.community_detector import CommunityDetector

        builder = GraphBuilder(config=cfg)
        G = builder.build(records, scores=scores, dis_summaries=dis_summaries)
        metrics = builder.compute_metrics(G)

        console.print(f"  Nodes: [bold]{G.number_of_nodes():,}[/bold]  "
                      f"Edges: [bold]{G.number_of_edges():,}[/bold]")

        # Community detection
        comm_detector = CommunityDetector(config=cfg)
        communities = comm_detector.detect(G)
        comm_detector.assign_to_graph(G, communities)
        community_summaries = comm_detector.describe(G, communities)
        _print_community_summary(community_summaries)

        # ----------------------------------------------------------------
        # 5. Visualizations
        # ----------------------------------------------------------------
        console.print("\n[cyan]Generating visualizations...[/cyan]")
        from src.visualization.network_visualizer import NetworkVisualizer

        viz_cfg = dict(cfg)
        if no_interactive:
            viz_cfg.setdefault("visualization", {})["interactive"] = False

        visualizer = NetworkVisualizer(config=viz_cfg, output_dir=output)
        output_files = visualizer.render_all(G, communities=communities)

        if scores:
            heatmap_path = visualizer.render_similarity_heatmap(scores)
            if heatmap_path:
                output_files["heatmap"] = heatmap_path

        for output_type, path in output_files.items():
            console.print(f"  [green]Saved {output_type}:[/green] {path}")

    # ----------------------------------------------------------------
    # 6. Save JSON report
    # ----------------------------------------------------------------
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"

    report = {
        "summary": {
            "total_tweets": len(records),
            "total_accounts": len(scores),
            "bots": sum(1 for s in scores if s["classification"] == "bot"),
            "suspected": sum(1 for s in scores if s["classification"] == "suspected"),
            "human": sum(1 for s in scores if s["classification"] == "human"),
        },
        "disinfo_corpus": corpus_summary,
        "accounts": [
            {
                "username": s.get("username"),
                "user_id": s.get("user_id"),
                "bot_score_pct": s.get("bot_score_pct"),
                "classification": s.get("classification"),
                "signals": s.get("signals"),
                "disinfo": dis_summaries.get(s.get("user_id"), {}),
            }
            for s in scores[:100]  # Top 100
        ],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    console.print(f"\n  [green]Report saved:[/green] {report_path}")
    console.print("\n[bold green]Analysis complete.[/bold green]")


# ------------------------------------------------------------------
# demo command
# ------------------------------------------------------------------

@cli.command()
@click.option("--output", "-o", default="./output", show_default=True)
@click.option("--accounts", default=60, show_default=True,
              help="Number of synthetic accounts to generate.")
def demo(output, accounts):
    """Run a demonstration with synthetic data (no API key required)."""
    console.print(Panel("[bold cyan]RuzzianProp DEMO[/bold cyan] – Synthetic data", expand=False))
    console.print(f"Generating {accounts} synthetic accounts...\n")

    from src.demo import generate_demo_data
    records = generate_demo_data(n_accounts=accounts)

    demo_path = Path(output) / "demo_data.json"
    Path(output).mkdir(parents=True, exist_ok=True)
    with open(demo_path, "w") as f:
        json.dump(records, f, indent=2)
    console.print(f"[green]Demo data written to:[/green] {demo_path}")

    # Invoke analyse with the generated file
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(analyse, [
        "--input", str(demo_path),
        "--output", output,
        "--verbose",
    ], catch_exceptions=False)
    console.print(result.output)


# ------------------------------------------------------------------
# Rich output helpers
# ------------------------------------------------------------------

def _print_scores_table(scores: list[dict]) -> None:
    table = Table(
        title="Top Accounts by Bot Score",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Rank", style="dim", width=5)
    table.add_column("Username", style="cyan", no_wrap=True)
    table.add_column("Bot Score", justify="right")
    table.add_column("Classification")
    table.add_column("Age (days)", justify="right")
    table.add_column("Followers", justify="right")
    table.add_column("Top Signal", style="dim")

    for i, score in enumerate(scores[:25], 1):
        classification = score.get("classification", "unknown")
        color = {"bot": "red", "suspected": "yellow", "human": "green"}.get(classification, "white")
        signals = score.get("signals") or {}
        top_signal = max(signals, key=signals.get, default="-") if signals else "-"
        author = score.get("author") or {}
        table.add_row(
            str(i),
            f"@{score.get('username', '?')}",
            f"[{color}]{score.get('bot_score_pct', 0):.1f}%[/{color}]",
            f"[{color}]{classification.upper()}[/{color}]",
            str(author.get("account_age_days", "?")),
            f"{author.get('followers_count', 0):,}",
            top_signal,
        )

    console.print(table)
    bots = sum(1 for s in scores if s["classification"] == "bot")
    suspected = sum(1 for s in scores if s["classification"] == "suspected")
    human = sum(1 for s in scores if s["classification"] == "human")
    console.print(
        f"  Total accounts: [bold]{len(scores)}[/bold]  "
        f"| [red]Bots: {bots}[/red]  "
        f"| [yellow]Suspected: {suspected}[/yellow]  "
        f"| [green]Human: {human}[/green]"
    )


def _print_disinfo_summary(summary: dict) -> None:
    table = Table(title="Top Disinformation Narratives", box=box.SIMPLE_HEAVY)
    table.add_column("Narrative", style="yellow")
    table.add_column("Count", justify="right")

    for item in summary.get("top_narratives", [])[:10]:
        table.add_row(item["narrative"], str(item["count"]))
    console.print(table)

    kw_table = Table(title="Top Disinformation Keywords", box=box.SIMPLE_HEAVY)
    kw_table.add_column("Keyword", style="red")
    kw_table.add_column("Occurrences", justify="right")
    for item in summary.get("top_keywords", [])[:10]:
        kw_table.add_row(item["keyword"], str(item["count"]))
    console.print(kw_table)


def _print_community_summary(summaries: list[dict]) -> None:
    table = Table(title="Network Communities", box=box.ROUNDED)
    table.add_column("Community", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Avg Bot Score", justify="right")
    table.add_column("Top Narrative", style="yellow")
    table.add_column("Top Account", style="dim")

    for s in summaries[:10]:
        avg = s.get("avg_bot_score_pct", 0)
        color = "red" if avg >= 60 else "yellow" if avg >= 40 else "green"
        top_account = ""
        if s.get("top_accounts"):
            top_account = "@" + s["top_accounts"][0].get("username", "?")
        top_narrative = (s.get("top_narratives") or ["—"])[0]
        table.add_row(
            f"C{s['community_id']}",
            str(s["size"]),
            f"[{color}]{avg:.1f}%[/{color}]",
            top_narrative,
            top_account,
        )
    console.print(table)


if __name__ == "__main__":
    cli()
