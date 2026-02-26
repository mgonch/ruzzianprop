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
# gui command
# ------------------------------------------------------------------

@cli.command()
@click.option("--port", default=8501, show_default=True, help="Port to run on.")
@click.option("--host", default="localhost", show_default=True, help="Host address.")
@click.option("--no-browser", is_flag=True, default=False, help="Don't open browser automatically.")
def gui(port, host, no_browser):
    """Launch the Streamlit web GUI for interactive account analysis."""
    import subprocess
    from pathlib import Path

    app_path = Path(__file__).parent.parent / "gui" / "app.py"
    if not app_path.exists():
        console.print("[red]GUI app not found at gui/app.py[/red]")
        raise SystemExit(1)

    console.print(Panel(
        f"[bold cyan]RuzzianProp GUI[/bold cyan]\n"
        f"Launching at [link=http://{host}:{port}]http://{host}:{port}[/link]",
        expand=False,
    ))

    cmd = [
        "streamlit", "run", str(app_path),
        "--server.port", str(port),
        "--server.address", host,
        "--theme.primaryColor", "#6366f1",
        "--theme.backgroundColor", "#ffffff",
        "--theme.secondaryBackgroundColor", "#f8fafc",
        "--theme.textColor", "#1e293b",
    ]
    if no_browser:
        cmd += ["--server.headless", "true"]

    subprocess.run(cmd)


# ------------------------------------------------------------------
# train command
# ------------------------------------------------------------------

@cli.command()
@click.option("--input", "-i", "input_path", default=None,
              help="Data file to pseudo-label (JSON/CSV). If omitted, uses synthetic data only.")
@click.option("--bootstrap", is_flag=True, default=False,
              help="Generate synthetic training data even if --input is provided.")
@click.option("--feedback", default="data/feedback_labels.json", show_default=True,
              help="Path to human-confirmed labels (FeedbackStore file).")
@click.option("--models-dir", default="models", show_default=True,
              help="Directory to save trained model artifacts.")
@click.option("--config", "-c", default="config/settings.yaml", show_default=True)
@click.option("--min-samples", default=30, show_default=True,
              help="Minimum labeled samples required before training.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def train(input_path, bootstrap, feedback, models_dir, config, min_samples, verbose):
    """Train (or retrain) the ML bot detection model."""
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    cfg = load_config(config)
    console.print(Panel("[bold cyan]RuzzianProp ML Trainer[/bold cyan]", expand=False))

    from src.ml.training_data import TrainingDataBuilder
    from src.ml.trainer import BotClassifier

    builder = TrainingDataBuilder(cfg)
    frames = []

    # 1. Human-confirmed labels (highest priority)
    console.print("[cyan]Loading human-confirmed labels...[/cyan]")
    fb_df = builder.from_feedback(feedback)
    if not fb_df.empty:
        console.print(f"  [green]{len(fb_df)} human labels loaded[/green]")
        frames.append(fb_df)
    else:
        console.print("  [dim]No human labels found yet[/dim]")

    # 2. Input file pseudo-labels
    if input_path:
        console.print(f"[cyan]Loading data from {input_path} for pseudo-labelling...[/cyan]")
        from src.collectors.file_collector import FileCollector
        from src.detectors.bot_detector import BotDetector
        records = FileCollector(cfg).load(input_path)
        detector = BotDetector(
            config=cfg,
            keywords=cfg.get("disinformation", {}).get("keywords", {}),
        )
        heuristic_scores = detector.score_all(records)
        pseudo_df = builder.from_heuristic_pseudolabels(records, heuristic_scores)
        if not pseudo_df.empty:
            console.print(f"  [green]{len(pseudo_df)} pseudo-labelled samples[/green]")
            frames.append(pseudo_df)

    # 3. Synthetic data (always included for bootstrap or when we need more data)
    total_so_far = sum(len(df) for df in frames)
    if bootstrap or total_so_far < min_samples:
        n_synthetic = max(200, min_samples * 2)
        console.print(f"[cyan]Generating {n_synthetic} synthetic training samples...[/cyan]")
        syn_df = builder.from_synthetic(n_accounts=n_synthetic)
        console.print(f"  [green]{len(syn_df)} synthetic samples[/green]")
        frames.append(syn_df)

    if not frames:
        console.print("[red]No training data available.[/red]")
        raise SystemExit(1)

    training_df = TrainingDataBuilder.merge(*frames)
    console.print(
        f"\n[bold]Training set:[/bold] {len(training_df)} samples  "
        f"| Bots: {(training_df['label']==1).sum()}  "
        f"| Humans: {(training_df['label']==0).sum()}"
    )

    # Train
    console.print("\n[cyan]Training ensemble model (RF + GBT + LR)...[/cyan]")
    clf = BotClassifier(cfg)
    try:
        metrics = clf.train(training_df, min_samples=min_samples)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    _print_training_metrics(metrics)

    # Feature importances
    if clf.feature_importances_:
        top = sorted(clf.feature_importances_.items(), key=lambda x: x[1], reverse=True)[:10]
        imp_table = Table(title="Top 10 Feature Importances", box=box.SIMPLE_HEAVY)
        imp_table.add_column("Feature", style="cyan")
        imp_table.add_column("Importance", justify="right")
        for feat, imp in top:
            imp_table.add_row(feat, f"{imp:.4f}")
        console.print(imp_table)

    # Save
    model_path = clf.save(models_dir)
    console.print(f"\n[green]Model saved:[/green] {model_path}")
    console.print(
        "\n[bold]Next step:[/bold] Run [cyan]ruzzianprop review[/cyan] to label uncertain "
        "accounts and improve precision."
    )


# ------------------------------------------------------------------
# review command (active learning)
# ------------------------------------------------------------------

@cli.command()
@click.option("--input", "-i", "input_path", required=True,
              help="Data file to analyse (JSON/CSV).")
@click.option("--feedback", default="data/feedback_labels.json", show_default=True,
              help="Path to FeedbackStore file.")
@click.option("--models-dir", default="models", show_default=True)
@click.option("--config", "-c", default="config/settings.yaml", show_default=True)
@click.option("--n", default=20, show_default=True,
              help="Number of accounts to review per session.")
@click.option("--auto-retrain", is_flag=True, default=False,
              help="Automatically retrain model after review session.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def review(input_path, feedback, models_dir, config, n, auto_retrain, verbose):
    """
    Interactive active learning review.

    Shows accounts where the ML model is most uncertain and prompts
    you to label them as bot or human. Labels are saved to the
    FeedbackStore and used to retrain the model.
    """
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    cfg = load_config(config)
    console.print(Panel("[bold cyan]RuzzianProp Active Learning Review[/bold cyan]", expand=False))

    from src.collectors.file_collector import FileCollector
    from src.detectors.bot_detector import BotDetector
    from src.ml.predictor import MLPredictor
    from src.ml.feedback_store import FeedbackStore

    records = FileCollector(cfg).load(input_path)
    detector = BotDetector(
        config=cfg,
        keywords=cfg.get("disinformation", {}).get("keywords", {}),
    )
    heuristic_scores = detector.score_all(records)

    predictor = MLPredictor(
        config=cfg,
        models_dir=models_dir,
        feedback_path=feedback,
        auto_retrain=False,
    )
    store = FeedbackStore(path=feedback)

    if predictor.ml_model is None:
        console.print(
            "[yellow]No trained ML model found.[/yellow] "
            "Run [cyan]ruzzianprop train[/cyan] first, or continue to label "
            "accounts for the initial training set.\n"
        )
        # Fall back to reviewing top heuristic scores
        candidates = _heuristic_review_queue(heuristic_scores, store, n)
    else:
        console.print(f"[green]ML model loaded.[/green] Selecting {n} uncertain accounts...\n")
        candidates = predictor.get_review_queue(records, heuristic_scores, n=n)

    if not candidates:
        console.print("[green]Review queue is empty – all accounts have been labelled.[/green]")
        return

    labelled_this_session = 0
    from src.ml.active_learner import ReviewCandidate

    for idx, candidate in enumerate(candidates, 1):
        console.rule(f"[bold]Account {idx}/{len(candidates)}[/bold]")

        # Account info panel
        uid = candidate.user_id if isinstance(candidate, ReviewCandidate) else candidate.get("user_id", "")
        username = candidate.username if isinstance(candidate, ReviewCandidate) else candidate.get("username", uid)
        ml_pct = candidate.ml_pct if isinstance(candidate, ReviewCandidate) else candidate.get("heuristic_pct", 0)
        h_pct = candidate.heuristic_pct if isinstance(candidate, ReviewCandidate) else candidate.get("heuristic_pct", 0)
        signals = candidate.signals if isinstance(candidate, ReviewCandidate) else {}
        top_tweets = candidate.top_tweets if isinstance(candidate, ReviewCandidate) else []
        features = candidate.features if isinstance(candidate, ReviewCandidate) else {}
        classification = candidate.classification if isinstance(candidate, ReviewCandidate) else "unknown"

        color = {"bot": "red", "suspected": "yellow", "human": "green"}.get(classification, "white")

        console.print(f"  [bold]@{username}[/bold]  (ID: {uid})")
        console.print(f"  ML score:        [{color}]{ml_pct:.1f}%[/{color}]")
        console.print(f"  Heuristic score: [{color}]{h_pct:.1f}%[/{color}]")

        if isinstance(candidate, ReviewCandidate) and candidate.disagreement:
            console.print("  [yellow]⚠ ML and heuristic strongly disagree[/yellow]")

        # Show top signals
        if signals:
            top_sigs = sorted(signals.items(), key=lambda x: x[1], reverse=True)[:4]
            console.print("  Top signals: " + ", ".join(f"{k}={v:.2f}" for k, v in top_sigs))

        # Show sample tweets
        if top_tweets:
            console.print("\n  [dim]Sample tweets:[/dim]")
            for tweet in top_tweets[:3]:
                console.print(f"    • {tweet[:120]}")

        console.print()

        # Prompt
        choice = click.prompt(
            "  Label this account",
            type=click.Choice(["b", "h", "s", "q"], case_sensitive=False),
            default="s",
            show_choices=True,
            prompt_suffix=" [b=bot, h=human, s=skip, q=quit] > ",
        )

        if choice == "q":
            console.print("[yellow]Quitting review session.[/yellow]")
            break
        elif choice == "s":
            store.skip(uid)
            console.print("  [dim]Skipped.[/dim]")
            continue
        else:
            label = 1 if choice == "b" else 0
            note = click.prompt("  Optional note (press Enter to skip)", default="")
            needs_retrain = store.add_label(
                user_id=uid,
                username=username,
                label=label,
                features=features,
                heuristic_score=h_pct / 100,
                reviewer_note=note,
            )
            labelled_this_session += 1
            label_str = "[red]BOT[/red]" if label == 1 else "[green]HUMAN[/green]"
            console.print(f"  Labelled as {label_str}. Total labels: {store.count()}")

            if needs_retrain:
                console.print(
                    f"\n  [bold yellow]Retrain trigger reached "
                    f"({store.retrain_trigger} new labels).[/bold yellow]"
                )
                if auto_retrain or click.confirm("  Retrain model now?", default=True):
                    console.print("  [cyan]Retraining...[/cyan]")
                    from click.testing import CliRunner
                    runner = CliRunner()
                    result = runner.invoke(train, [
                        "--input", input_path, "--config", config,
                        "--models-dir", models_dir, "--feedback", feedback,
                    ], catch_exceptions=False)
                    console.print(result.output)
                    store.mark_retrained()

    # Session summary
    console.rule()
    console.print(
        f"\n[bold green]Session complete.[/bold green] "
        f"Labelled [bold]{labelled_this_session}[/bold] accounts. "
        f"Total labels in store: [bold]{store.count()}[/bold] "
        f"({store.count_by_label()})"
    )
    if store.new_since_train() > 0:
        console.print(
            f"  [dim]{store.new_since_train()} new labels since last training. "
            f"Run [cyan]ruzzianprop train[/cyan] to update the model.[/dim]"
        )


def _heuristic_review_queue(
    heuristic_scores: list[dict],
    store,
    n: int,
) -> list[dict]:
    """Fallback review queue when no ML model is available."""
    candidates = []
    for s in heuristic_scores:
        uid = s.get("user_id", "")
        if store.is_labelled(uid) or store.is_skipped(uid):
            continue
        # Surface suspected + borderline accounts first
        score = s.get("bot_score", 0.0)
        if 0.30 <= score <= 0.80:
            candidates.append({
                "user_id": uid,
                "username": (s.get("author") or {}).get("username", uid),
                "ml_pct": score * 100,
                "heuristic_pct": score * 100,
                "classification": s.get("classification", "unknown"),
                "signals": s.get("signals", {}),
                "top_tweets": [],
                "features": {},
            })
    candidates.sort(key=lambda c: abs(c["heuristic_pct"] / 100 - 0.5))
    return candidates[:n]


# ------------------------------------------------------------------
# model-info command
# ------------------------------------------------------------------

@cli.command("model-info")
@click.option("--models-dir", default="models", show_default=True)
def model_info(models_dir):
    """Show information about the trained ML model."""
    from src.ml.trainer import BotClassifier
    from src.ml.feature_extractor import FEATURE_NAMES

    if not BotClassifier.is_trained(models_dir):
        console.print("[yellow]No trained model found. Run 'ruzzianprop train' first.[/yellow]")
        return

    clf = BotClassifier.load(models_dir)

    console.print(Panel("[bold cyan]ML Model Information[/bold cyan]", expand=False))

    # CV metrics
    m = clf.cv_metrics_
    console.print(f"  Training samples : [bold]{m.get('n_samples', '?')}[/bold]")
    console.print(f"  Bots / Humans    : {m.get('n_bots', '?')} / {m.get('n_humans', '?')}")
    console.print(f"  CV ROC-AUC       : [green]{m.get('cv_roc_auc_mean', 0):.3f}[/green] "
                  f"± {m.get('cv_roc_auc_std', 0):.3f}")
    console.print(f"  CV F1            : [green]{m.get('cv_f1_mean', 0):.3f}[/green] "
                  f"± {m.get('cv_f1_std', 0):.3f}")
    console.print(f"  CV Avg Precision : [green]{m.get('cv_avg_precision_mean', 0):.3f}[/green]")

    # Feature importances
    if clf.feature_importances_:
        top = sorted(clf.feature_importances_.items(), key=lambda x: x[1], reverse=True)[:15]
        imp_table = Table(title="Feature Importances (RF)", box=box.SIMPLE)
        imp_table.add_column("Feature", style="cyan")
        imp_table.add_column("Importance", justify="right")
        imp_table.add_column("Bar")
        max_imp = top[0][1] if top else 1.0
        for feat, imp in top:
            bar = "█" * int(imp / max_imp * 30)
            imp_table.add_row(feat, f"{imp:.4f}", f"[blue]{bar}[/blue]")
        console.print(imp_table)

    # Training history
    hist_path = Path(models_dir) / "training_history.json"
    if hist_path.exists():
        import json as _json
        with open(hist_path) as f:
            history = _json.load(f)
        hist_table = Table(title="Training History", box=box.SIMPLE)
        hist_table.add_column("Version", style="dim")
        hist_table.add_column("Samples", justify="right")
        hist_table.add_column("ROC-AUC", justify="right")
        hist_table.add_column("F1", justify="right")
        for entry in history[-5:]:
            v = entry.get("model_file", "?")
            hist_table.add_row(
                v,
                str(entry.get("n_samples", "?")),
                f"{entry.get('cv_roc_auc_mean', 0):.3f}",
                f"{entry.get('cv_f1_mean', 0):.3f}",
            )
        console.print(hist_table)


# ------------------------------------------------------------------
# Rich output helpers
# ------------------------------------------------------------------

def _print_training_metrics(metrics: dict) -> None:
    table = Table(title="Cross-Validation Metrics (5-fold)", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Std", justify="right", style="dim")
    table.add_row("ROC-AUC",
                  f"[green]{metrics.get('cv_roc_auc_mean', 0):.3f}[/green]",
                  f"±{metrics.get('cv_roc_auc_std', 0):.3f}")
    table.add_row("F1",
                  f"[green]{metrics.get('cv_f1_mean', 0):.3f}[/green]",
                  f"±{metrics.get('cv_f1_std', 0):.3f}")
    table.add_row("Avg Precision",
                  f"[green]{metrics.get('cv_avg_precision_mean', 0):.3f}[/green]",
                  "—")
    table.add_row("Samples", str(metrics.get("n_samples", "?")), "—")
    table.add_row("Bots", str(metrics.get("n_bots", "?")), "—")
    table.add_row("Humans", str(metrics.get("n_humans", "?")), "—")
    console.print(table)


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
