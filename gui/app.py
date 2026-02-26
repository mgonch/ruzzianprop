"""
RuzzianProp GUI
===============
Streamlit web interface for bot & disinformation detection.

Run with:
    streamlit run gui/app.py
  or:
    ruzzianprop gui

Two modes
---------
Live mode   – enter a @username; the app fetches data via the Twitter API.
              Requires TWITTER_BEARER_TOKEN env var.
Manual mode – paste account details + sample tweets; no API key needed.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Ensure project root is on the path when running via `streamlit run gui/app.py`
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detectors.bot_detector import BotDetector
from src.detectors.disinformation_detector import DisinformationDetector

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RuzzianProp – Bot Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
/* Gauge label sizing */
.metric-label { font-size: 0.85rem !important; }

/* Classification badge */
.badge-bot         { background:#ef4444; color:#fff; padding:4px 14px; border-radius:9999px; font-weight:700; }
.badge-suspected   { background:#f59e0b; color:#fff; padding:4px 14px; border-radius:9999px; font-weight:700; }
.badge-human       { background:#22c55e; color:#fff; padding:4px 14px; border-radius:9999px; font-weight:700; }
.badge-state_actor { background:#dc2626; color:#fff; padding:4px 14px; border-radius:9999px; font-weight:700; }

/* Source tag */
.source-ml        { font-size:0.75rem; color:#6366f1; }
.source-heuristic { font-size:0.75rem; color:#64748b; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config + model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_config() -> dict:
    cfg_path = ROOT / "config" / "settings.yaml"
    if cfg_path.exists():
        import yaml
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


@st.cache_resource
def load_ml_model():
    """Load the ML model if available. Returns (model, loaded: bool)."""
    try:
        from src.ml.trainer import BotClassifier
        if BotClassifier.is_trained(ROOT / "models"):
            return BotClassifier.load(ROOT / "models"), True
    except Exception as e:
        logger.warning("ML model load failed: %s", e)
    return None, False


@st.cache_resource
def load_feature_extractor(config: dict):
    from src.ml.feature_extractor import FeatureExtractor
    dis = config.get("disinformation", {})
    return FeatureExtractor(
        keywords=dis.get("keywords", {}),
        state_media=set(dis.get("state_media_domains", [])),
    )


def load_feedback_store():
    from src.ml.feedback_store import FeedbackStore
    return FeedbackStore(path=str(ROOT / "data" / "feedback_labels.json"))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def gauge_chart(pct: float, label: str, color: str, height: int = 250) -> go.Figure:
    """Circular gauge for a 0–100 score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "color": "#1e293b"}},
        title={"text": label, "font": {"size": 14, "color": "#475569"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#cbd5e1"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#f8fafc",
            "borderwidth": 2,
            "bordercolor": "#e2e8f0",
            "steps": [
                {"range": [0, 40],  "color": "#dcfce7"},
                {"range": [40, 60], "color": "#fef9c3"},
                {"range": [60, 100],"color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "#1e293b", "width": 3},
                "thickness": 0.8,
                "value": pct,
            },
        },
    ))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=40, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def signals_bar_chart(signals: dict) -> go.Figure:
    names = list(signals.keys())
    values = [round(v * 100, 1) for v in signals.values()]
    colors = ["#ef4444" if v >= 60 else "#f59e0b" if v >= 35 else "#22c55e" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Signal Scores",
        xaxis=dict(range=[0, 110], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(autorange="reversed"),
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def feature_importance_chart(importances: dict, top_n: int = 12) -> go.Figure:
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [f[0].replace("_", " ") for f in sorted_feats]
    vals = [f[1] for f in sorted_feats]
    fig = go.Figure(go.Bar(
        x=vals, y=names,
        orientation="h",
        marker_color="#6366f1",
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        title="ML Feature Importances (Random Forest)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(autorange="reversed"),
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def score_comparison_chart(ml_pct: float, heuristic_pct: float) -> go.Figure:
    fig = go.Figure()
    for label, val, color in [
        ("ML Model", ml_pct, "#6366f1"),
        ("Heuristic", heuristic_pct, "#64748b"),
    ]:
        fig.add_trace(go.Bar(
            name=label, x=[val], y=[label],
            orientation="h",
            marker_color=color,
            text=[f"{val:.1f}%"], textposition="outside",
            width=0.5,
        ))
    fig.update_layout(
        title="Score Comparison",
        xaxis=dict(range=[0, 115], showgrid=False, zeroline=False, showticklabels=False),
        barmode="group",
        height=160,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis engine
# ─────────────────────────────────────────────────────────────────────────────

def build_records(user: dict, tweets: list[dict]) -> list[dict]:
    """Package user + tweets into the internal record format."""
    uid = user.get("id", "manual_user")
    records = []
    for t in tweets:
        records.append({
            "id": t.get("id", "t0"),
            "text": t.get("text", ""),
            "author_id": uid,
            "created_at": t.get("created_at"),
            "referenced_tweets": t.get("referenced_tweets", []),
            "entities": t.get("entities", {}),
            "author": user,
        })
    if not records:
        records.append({
            "id": "placeholder",
            "text": "",
            "author_id": uid,
            "created_at": None,
            "referenced_tweets": [],
            "entities": {},
            "author": user,
        })
    return records


@st.cache_data(ttl=300)
def fetch_twitter_user(username: str) -> tuple[dict | None, list[dict], str | None]:
    """Fetch user + recent tweets from Twitter API. Returns (user, tweets, error)."""
    config = load_config()
    try:
        from src.collectors.twitter_collector import TwitterCollector
        collector = TwitterCollector(config)
        records = collector.get_user_timeline(
            username=username.lstrip("@"), max_results=50
        )
        if not records:
            return None, [], f"No data returned for @{username}"
        user = records[0].get("author", {})
        return user, records, None
    except Exception as e:
        return None, [], str(e)


def analyse(user: dict, tweets: list[dict], config: dict) -> dict:
    """Run heuristic + ML analysis and return unified result dict."""
    records = build_records(user, tweets)

    # Heuristic
    detector = BotDetector(
        config=config,
        keywords=config.get("disinformation", {}).get("keywords", {}),
    )
    heuristic_scores = detector.score_all(records)
    hs = heuristic_scores[0] if heuristic_scores else {}

    result = {
        "heuristic": hs,
        "ml": None,
        "ml_available": False,
        "features": {},
    }

    # Disinformation analysis – separate threat model from bot automation
    dis_detector = DisinformationDetector(config=config)
    dis_summary = dis_detector.analyse_account(records)
    dis_score = dis_summary.get("avg_dis_score", 0.0)
    result["dis"] = dis_summary
    result["dis_score"] = dis_score

    # ML
    ml_model, ml_loaded = load_ml_model()
    if ml_loaded:
        try:
            from src.ml.feature_extractor import FEATURE_NAMES
            extractor = load_feature_extractor(config)
            features_df = extractor.extract_batch(records, heuristic_scores=heuristic_scores)
            if not features_df.empty:
                X = features_df.values.astype(np.float32)
                unc = ml_model.predict_with_uncertainty(X)
                ml_prob = float(unc["prob"][0])
                result["ml"] = {
                    "prob": ml_prob,
                    "pct": round(ml_prob * 100, 1),
                    "uncertainty": float(unc["uncertainty"][0]),
                    "lower": round(float(unc["lower"][0]) * 100, 1),
                    "upper": round(float(unc["upper"][0]) * 100, 1),
                }
                result["features"] = dict(zip(FEATURE_NAMES, features_df.iloc[0].tolist()))
                result["ml_available"] = True
        except Exception as e:
            st.warning(f"ML scoring error: {e}")

    # Combined
    h_score = float(hs.get("bot_score", 0.0))
    if result["ml_available"] and result["ml"]:
        combined = 0.60 * result["ml"]["prob"] + 0.40 * h_score
        result["source"] = "ml+heuristic"
    else:
        combined = h_score
        result["source"] = "heuristic"

    combined_pct = round(combined * 100, 1)
    result["combined_score"] = combined
    result["combined_pct"] = combined_pct

    if combined >= 0.60:
        result["classification"] = "bot"
    elif combined >= 0.40:
        result["classification"] = "suspected"
    elif dis_score >= 0.35:
        # Human-operated account posting disinformation: not a bot, but not clean
        result["classification"] = "state_actor"
    else:
        result["classification"] = "human"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Result rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_results(result: dict, user: dict, tweets: list[dict]) -> None:
    clf = result["classification"]
    combined_pct = result["combined_pct"]
    hs = result["heuristic"]
    h_pct = round(float(hs.get("bot_score", 0.0)) * 100, 1)
    dis_score = result.get("dis_score", 0.0)
    dis_pct = round(dis_score * 100, 1)
    dis_color = "#ef4444" if dis_pct >= 60 else "#f59e0b" if dis_pct >= 35 else "#64748b"

    # ── Header ───────────────────────────────────────────────────────────────
    username = user.get("username", "unknown")
    badge_cls = f"badge-{clf}"
    _badge_labels = {
        "bot": "BOT", "suspected": "SUSPECTED",
        "human": "HUMAN", "state_actor": "STATE ACTOR",
    }
    badge_label = _badge_labels.get(clf, clf.upper())
    color_map = {
        "bot": "#ef4444", "suspected": "#f59e0b",
        "human": "#22c55e", "state_actor": "#dc2626",
    }
    gauge_color = color_map.get(clf, "#22c55e")

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
        <h2 style="margin:0;">@{username}</h2>
        <span class="{badge_cls}">{badge_label}</span>
        <span style="font-size:0.8rem; color:#64748b;">source: {result['source']}</span>
    </div>
    """, unsafe_allow_html=True)

    if clf == "state_actor":
        st.warning(
            "**State actor / influence operator detected.**  \n"
            "This account shows low bot-automation signals but was flagged for disinformation "
            "content. Human-operated state-affiliated accounts deliberately avoid bot-like "
            "behaviour — they post manually to evade detection. A high **Disinfo Score** is "
            "the primary signal here.",
            icon="🕵️",
        )

    # ── Gauges row ────────────────────────────────────────────────────────────
    if result["ml_available"] and result["ml"]:
        g_col1, g_col2, g_col3, g_col4 = st.columns(4)
        with g_col1:
            st.plotly_chart(
                gauge_chart(combined_pct, "Bot Score", gauge_color),
                use_container_width=True, key="gauge_combined",
            )
        with g_col2:
            st.plotly_chart(
                gauge_chart(dis_pct, "Disinfo Score", dis_color),
                use_container_width=True, key="gauge_dis",
            )
        with g_col3:
            ml = result["ml"]
            st.plotly_chart(
                gauge_chart(ml["pct"], "ML Model", "#6366f1"),
                use_container_width=True, key="gauge_ml",
            )
            st.caption(
                f"95% CI: {ml['lower']}% – {ml['upper']}%  "
                f"(uncertainty: ±{round(ml['uncertainty']*100,1)}%)"
            )
        with g_col4:
            st.plotly_chart(
                gauge_chart(h_pct, "Heuristic", "#64748b"),
                use_container_width=True, key="gauge_heuristic",
            )
    else:
        g_col1, g_col2, g_col3 = st.columns(3)
        with g_col1:
            st.plotly_chart(
                gauge_chart(combined_pct, "Bot Score", gauge_color),
                use_container_width=True, key="gauge_single",
            )
        with g_col2:
            st.plotly_chart(
                gauge_chart(dis_pct, "Disinfo Score", dis_color),
                use_container_width=True, key="gauge_dis",
            )
        with g_col3:
            st.info("No ML model trained yet. Run `ruzzianprop train` to enable ML scoring.", icon="ℹ️")

    # ── ML vs Heuristic bar ───────────────────────────────────────────────────
    if result["ml_available"] and result["ml"]:
        ml_pct = result["ml"]["pct"]
        diff = abs(ml_pct - h_pct)
        if diff >= 20:
            st.warning(
                f"ML ({ml_pct}%) and heuristic ({h_pct}%) disagree by {diff:.0f}%. "
                "This account is a good candidate for human review.",
                icon="⚠️",
            )
        st.plotly_chart(
            score_comparison_chart(ml_pct, h_pct),
            use_container_width=True, key="cmp_chart",
        )

    # ── Disinformation narrative breakdown ────────────────────────────────────
    dis = result.get("dis", {})
    if dis.get("top_narratives") or dis.get("state_media_share", 0) > 0 or dis.get("top_keywords"):
        narratives_str = ", ".join(dis.get("top_narratives", [])[:5]) or "—"
        keywords_str = "  |  ".join(f"`{k}`" for k in dis.get("top_keywords", [])[:8]) or "—"
        sm_pct = round(dis.get("state_media_share", 0) * 100, 1)
        st.error(
            f"**Disinformation signals detected**  \n"
            f"Narratives: **{narratives_str}**  \n"
            f"Matched keywords: {keywords_str}  \n"
            f"State media links: **{sm_pct}%** of tweets analysed",
            icon="🚨",
        )

    st.divider()

    # ── Signals + Feature importances ────────────────────────────────────────
    chart_col, detail_col = st.columns([1, 1])
    with chart_col:
        signals = hs.get("signals", {})
        if signals:
            st.plotly_chart(
                signals_bar_chart(signals),
                use_container_width=True, key="signals_chart",
            )
    with detail_col:
        ml_model, ml_loaded = load_ml_model()
        if ml_loaded and ml_model.feature_importances_:
            st.plotly_chart(
                feature_importance_chart(ml_model.feature_importances_),
                use_container_width=True, key="feat_imp_chart",
            )
        else:
            _render_account_info(user)

    # ── Tweets panel ─────────────────────────────────────────────────────────
    if tweets:
        with st.expander(f"Sample tweets ({len(tweets)} analysed)", expanded=False):
            for t in tweets[:10]:
                txt = t.get("text", "")
                is_rt = any(
                    r.get("type") == "retweeted"
                    for r in (t.get("referenced_tweets") or [])
                )
                prefix = "🔁 " if is_rt else "🐦 "
                st.markdown(f"{prefix} {txt}")
                st.caption(t.get("created_at", ""))
                st.divider()

    # ── Active learning label buttons ─────────────────────────────────────────
    st.subheader("Label this account")
    st.caption(
        "Your label is saved to the FeedbackStore and used to retrain the ML model. "
        f"Label trigger: every {st.session_state.get('retrain_trigger', 10)} new labels."
    )

    uid = user.get("id", "manual_user")
    features = result.get("features", {})
    h_score = float(hs.get("bot_score", 0.0))

    b_col, h_col, s_col = st.columns([1, 1, 2])
    with b_col:
        if st.button("🤖 Label as BOT", type="primary", use_container_width=True):
            _save_label(uid, username, 1, features, h_score)
    with h_col:
        if st.button("👤 Label as HUMAN", type="secondary", use_container_width=True):
            _save_label(uid, username, 0, features, h_score)
    with s_col:
        note = st.text_input("Optional note", key=f"note_{uid}", label_visibility="collapsed",
                             placeholder="Optional note about this account...")

    # ── Raw JSON expander ─────────────────────────────────────────────────────
    with st.expander("Raw score details (JSON)", expanded=False):
        st.json({
            "classification": result["classification"],
            "combined_pct": result["combined_pct"],
            "heuristic": {k: v for k, v in hs.items() if k != "author"},
            "ml": result.get("ml"),
            "features_top10": dict(
                sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            ) if features else {},
        })


def _render_account_info(user: dict) -> None:
    st.subheader("Account details")
    items = [
        ("Username", f"@{user.get('username', '?')}"),
        ("Account age", f"{user.get('account_age_days', '?')} days"),
        ("Followers", f"{user.get('followers_count', '?'):,}" if isinstance(user.get('followers_count'), int) else "?"),
        ("Following", f"{user.get('following_count', '?'):,}" if isinstance(user.get('following_count'), int) else "?"),
        ("Tweets", f"{user.get('tweet_count', '?'):,}" if isinstance(user.get('tweet_count'), int) else "?"),
        ("Verified", "✓" if user.get("verified") else "✗"),
        ("Has bio", "✓" if (user.get("description") or "").strip() else "✗"),
        ("Location", user.get("location") or "—"),
    ]
    for k, v in items:
        st.metric(k, v)


def _save_label(uid: str, username: str, label: int, features: dict, h_score: float) -> None:
    try:
        store = load_feedback_store()
        needs_retrain = store.add_label(uid, username, label, features, h_score)
        label_str = "BOT" if label == 1 else "HUMAN"
        st.success(f"Saved @{username} → {label_str}. Total labels: {store.count()}")
        if needs_retrain:
            st.warning(
                f"Retrain trigger reached ({store.retrain_trigger} new labels)! "
                "Run `ruzzianprop train` to update the model.",
                icon="🔄",
            )
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Failed to save label: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(config: dict) -> None:
    with st.sidebar:
        st.title("🔍 RuzzianProp")
        st.caption("Bot & Disinformation Detector")
        st.divider()

        # ML model status
        _, ml_loaded = load_ml_model()
        if ml_loaded:
            ml_model, _ = load_ml_model()
            m = ml_model.cv_metrics_
            st.success("ML model loaded", icon="✓")
            st.metric("CV ROC-AUC", f"{m.get('cv_roc_auc_mean', 0):.3f}")
            st.metric("CV F1",      f"{m.get('cv_f1_mean', 0):.3f}")
            st.metric("Training samples", m.get("n_samples", "?"))
        else:
            st.warning("No ML model trained yet", icon="⚠️")
            st.caption("Run `ruzzianprop train` to train the model.")

        st.divider()

        # Feedback store stats
        try:
            store = load_feedback_store()
            counts = store.count_by_label()
            st.metric("Labelled accounts", store.count())
            st.caption(f"Bots: {counts['bot']} | Humans: {counts['human']}")
            st.metric("New labels since train", store.new_since_train())
            trigger = store.retrain_trigger
            st.session_state["retrain_trigger"] = trigger
            st.progress(
                min(store.new_since_train() / max(trigger, 1), 1.0),
                text=f"Retrain in {max(trigger - store.new_since_train(), 0)} more labels",
            )
        except Exception:
            pass

        st.divider()

        # Twitter API status
        bearer = os.environ.get("TWITTER_BEARER_TOKEN", "")
        if bearer:
            st.success("Twitter API connected", icon="🐦")
        else:
            st.info("No Twitter API key – use Manual mode", icon="ℹ️")

        st.divider()
        st.caption("v0.3 · claude/ml-self-learning-dU0OJ")


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    config = load_config()
    render_sidebar(config)

    # Top-level navigation
    page = st.sidebar.radio(
        "Page",
        ["Account Analysis", "Network Explorer"],
        index=0,
        label_visibility="collapsed",
    )

    if page == "Network Explorer":
        render_network_page(config)
        return

    st.header("Account Analysis")

    tab_live, tab_manual, tab_history = st.tabs(
        ["🐦 Live (Twitter API)", "✏️ Manual Entry", "📋 Label History"]
    )

    # ── Tab 1: Live Twitter lookup ────────────────────────────────────────────
    with tab_live:
        st.subheader("Look up a Twitter/X account")
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            username_input = st.text_input(
                "Username", placeholder="@username or username",
                key="live_username", label_visibility="collapsed",
            )
        with col_btn:
            live_go = st.button("Analyse", type="primary", key="live_go", use_container_width=True)

        bearer = os.environ.get("TWITTER_BEARER_TOKEN", "")
        if not bearer:
            st.warning(
                "Set `TWITTER_BEARER_TOKEN` environment variable to enable live lookups. "
                "Use the **Manual Entry** tab to analyse without an API key.",
                icon="🔑",
            )

        if live_go and username_input:
            if not bearer:
                st.error("Twitter API key required for live lookup.")
            else:
                with st.spinner(f"Fetching @{username_input.lstrip('@')}..."):
                    user, tweets, err = fetch_twitter_user(username_input)
                if err:
                    st.error(f"API error: {err}")
                elif user:
                    with st.spinner("Analysing..."):
                        result = analyse(user, tweets, config)
                    render_results(result, user, tweets)

    # ── Tab 2: Manual entry ───────────────────────────────────────────────────
    with tab_manual:
        st.subheader("Enter account details manually")
        st.caption("No Twitter API key needed. Paste publicly visible information.")

        with st.form("manual_form"):
            c1, c2 = st.columns(2)
            with c1:
                m_username     = st.text_input("Username (without @)", value="")
                m_account_age  = st.number_input("Account age (days)", min_value=0, value=365, step=1)
                m_followers    = st.number_input("Followers", min_value=0, value=500, step=1)
                m_following    = st.number_input("Following", min_value=0, value=200, step=1)
                m_tweet_count  = st.number_input("Total tweets", min_value=0, value=1000, step=1)
            with c2:
                m_verified   = st.checkbox("Verified account", value=False)
                m_has_bio    = st.checkbox("Has profile bio", value=True)
                m_has_loc    = st.checkbox("Has location set", value=True)
                m_has_avatar = st.checkbox("Has custom avatar", value=True)
                m_bio_text   = st.text_area("Bio text (optional)", height=80)

            st.markdown("**Sample tweets** (one per line, paste up to 50)")
            m_tweets_raw = st.text_area(
                "Tweets", height=200,
                placeholder="Paste tweet texts here, one per line...",
                label_visibility="collapsed",
            )

            submitted = st.form_submit_button("Analyse", type="primary", use_container_width=True)

        if submitted:
            if not m_username:
                st.error("Please enter a username.")
            else:
                user = {
                    "id": f"manual_{m_username}",
                    "username": m_username,
                    "name": m_username,
                    "description": m_bio_text if m_has_bio else "",
                    "location": "Entered manually" if m_has_loc else "",
                    "profile_image_url": ("https://pic.example.com/img.jpg"
                                          if m_has_avatar else None),
                    "verified": m_verified,
                    "protected": False,
                    "account_age_days": int(m_account_age),
                    "followers_count": int(m_followers),
                    "following_count": int(m_following),
                    "tweet_count": int(m_tweet_count),
                    "listed_count": 0,
                }

                tweets = []
                if m_tweets_raw.strip():
                    lines = [l.strip() for l in m_tweets_raw.strip().splitlines() if l.strip()]
                    for i, line in enumerate(lines[:50]):
                        is_rt = line.startswith("RT ")
                        tweets.append({
                            "id": f"t{i}",
                            "text": line,
                            "author_id": user["id"],
                            "created_at": None,
                            "referenced_tweets": (
                                [{"type": "retweeted", "id": "0"}] if is_rt else []
                            ),
                            "entities": {},
                        })

                with st.spinner("Analysing..."):
                    result = analyse(user, tweets, config)
                render_results(result, user, tweets)

    # ── Tab 3: Label history ──────────────────────────────────────────────────
    with tab_history:
        st.subheader("Human-confirmed labels (FeedbackStore)")
        try:
            store = load_feedback_store()
            entries = store.get_all()
            if not entries:
                st.info("No labels yet. Analyse accounts and label them to build the training set.")
            else:
                df = pd.DataFrame([
                    {
                        "username": e.get("username"),
                        "label": "🤖 bot" if e["label"] == 1 else "👤 human",
                        "heuristic_score": f"{e.get('heuristic_score', 0)*100:.1f}%",
                        "note": e.get("reviewer_note", ""),
                        "reviewed_at": e.get("reviewed_at", ""),
                    }
                    for e in entries
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
                counts = store.count_by_label()
                m1, m2, m3 = st.columns(3)
                m1.metric("Total", store.count())
                m2.metric("Bots", counts["bot"])
                m3.metric("Humans", counts["human"])

                # Pie chart
                if store.count() > 0:
                    fig = go.Figure(go.Pie(
                        labels=["Bot", "Human"],
                        values=[counts["bot"], counts["human"]],
                        marker_colors=["#ef4444", "#22c55e"],
                        hole=0.4,
                    ))
                    fig.update_layout(
                        height=250,
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load label history: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Network Explorer page
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def _build_network_cached(
    records_json: str,
    scores_json: str,
    config_json: str,
):
    """Cache-friendly wrapper — takes JSON strings so Streamlit can hash them."""
    import json as _json
    records = _json.loads(records_json)
    scores  = _json.loads(scores_json)
    config  = _json.loads(config_json)
    from gui.network_view import build_full_network
    G, communities, summaries = build_full_network(records, scores, config)
    # Serialise graph to dict for caching
    return G, communities, summaries


def render_network_page(config: dict) -> None:
    from gui import network_view as nv
    import json as _json

    st.header("Network Explorer")
    st.caption(
        "Visualise how bot and disinfo accounts interact as a community. "
        "Upload a JSON data file (same format as `ruzzianprop analyse --input`) "
        "or use the Twitter API to collect a dataset first."
    )

    # ── Data source ───────────────────────────────────────────────────────────
    with st.expander("Load dataset", expanded=True):
        src_mode = st.radio(
            "Data source",
            ["Upload JSON file", "Use demo data"],
            horizontal=True,
        )

        records: list[dict] = []
        if src_mode == "Upload JSON file":
            uploaded = st.file_uploader(
                "Upload tweet records JSON",
                type=["json"],
                help="The JSON file produced by ruzzianprop analyse or ruzzianprop demo.",
            )
            if uploaded:
                try:
                    records = _json.load(uploaded)
                    st.success(f"Loaded {len(records)} records.")
                except Exception as e:
                    st.error(f"Could not parse file: {e}")
        else:
            n_accs = st.slider("Synthetic accounts", 30, 200, 80, step=10)
            if st.button("Generate demo data", type="primary"):
                with st.spinner("Generating..."):
                    from src.demo import generate_demo_data
                    records = generate_demo_data(n_accounts=n_accs)
                st.session_state["network_records"] = records
                st.success(f"Generated {len(records)} records for {n_accs} accounts.")

        if "network_records" in st.session_state and not records:
            records = st.session_state["network_records"]

    if not records:
        st.info("Load or generate a dataset above to begin.", icon="📂")
        return

    # ── Score accounts ────────────────────────────────────────────────────────
    with st.spinner("Scoring accounts & building graph..."):
        from src.detectors.bot_detector import BotDetector
        detector = BotDetector(
            config=config,
            keywords=config.get("disinformation", {}).get("keywords", {}),
        )
        scores = detector.score_all(records)
        G, communities, summaries = build_full_network(records, scores, config)

    if len(G) == 0:
        st.warning("Graph is empty – no interaction edges found in the dataset.")
        return

    # ── Summary metrics strip ─────────────────────────────────────────────────
    n_bots  = sum(1 for n in G.nodes() if G.nodes[n].get("classification") == "bot")
    n_susp  = sum(1 for n in G.nodes() if G.nodes[n].get("classification") == "suspected")
    n_human = sum(1 for n in G.nodes() if G.nodes[n].get("classification") == "human")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accounts",   len(G))
    m2.metric("Edges",      G.number_of_edges())
    m3.metric("Communities",len(communities))
    m4.metric("Bots",       n_bots,  delta=None)
    m5.metric("Suspected",  n_susp,  delta=None)
    m6.metric("Humans",     n_human, delta=None)

    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        color_by = st.selectbox(
            "Colour nodes by",
            ["classification", "community", "bot_score"],
            index=0,
        )
    with ctrl2:
        size_by = st.selectbox(
            "Node size by",
            ["pagerank", "followers", "tweets", "uniform"],
            index=0,
        )
    with ctrl3:
        layout_algo = st.selectbox(
            "Layout",
            ["community", "spring", "kamada_kawai", "circular"],
            index=0,
        )
    with ctrl4:
        min_edge = st.number_input(
            "Min edge weight",
            min_value=1, max_value=10, value=1, step=1,
        )

    # ── Compute layout (cached per algo) ─────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def _layout(_G_nodes, _G_edges, _communities_items, algo):
        """Wrap layout computation for caching."""
        # Rebuild minimal graph for layout
        _G = nx.DiGraph()
        _G.add_nodes_from(_G_nodes)
        _G.add_edges_from(_G_edges)
        _comm = dict(_communities_items)
        return nv.compute_layout(_G, _comm, algorithm=algo)

    import networkx as nx
    pos = _layout(
        list(G.nodes(data=True)),
        list(G.edges(data=True)),
        list(communities.items()),
        layout_algo,
    )

    # ── Main graph ────────────────────────────────────────────────────────────
    st.subheader("Interaction Graph")
    st.caption(
        "**Nodes** = accounts (size = PageRank influence). "
        "**Edges** = interactions: retweets (red), mentions (grey), co-hashtag (purple dashed). "
        "Hover any node for full details."
    )
    graph_fig = nv.network_graph(
        G, communities, pos,
        color_by=color_by,
        size_by=size_by,
        min_edge_weight=int(min_edge),
        height=680,
    )
    st.plotly_chart(graph_fig, use_container_width=True, key="main_network")

    st.divider()

    # ── Community breakdown ───────────────────────────────────────────────────
    comm_col, sun_col = st.columns([1, 1])
    with comm_col:
        st.subheader("Community Summary")
        if summaries:
            st.plotly_chart(
                nv.community_stats_bars(summaries),
                use_container_width=True, key="comm_bars",
            )

            # Expandable community details
            with st.expander("Community details"):
                for s in sorted(summaries, key=lambda x: -x["size"])[:10]:
                    top_names = ", ".join(
                        "@" + (G.nodes[u].get("username") or u)
                        for u in s.get("top_accounts", [])
                        if G.has_node(u)
                    )
                    bridge_names = ", ".join(
                        "@" + (G.nodes[u].get("username") or u)
                        for u in s.get("bridge_accounts", [])
                        if G.has_node(u)
                    )
                    clf_str = " | ".join(
                        f"{k}: {v}" for k, v in s.get("classifications", {}).items()
                    )
                    st.markdown(
                        f"**Community {s['community_id']}** — "
                        f"{s['size']} accounts | avg bot score: "
                        f"**{s['avg_bot_score_pct']:.1f}%**  \n"
                        f"Classifications: {clf_str}  \n"
                        f"Top accounts (by PageRank): {top_names}  \n"
                        f"Bridge accounts: {bridge_names}"
                    )
                    if s.get("top_narratives"):
                        st.caption("Narratives: " + ", ".join(s["top_narratives"]))
                    st.divider()

    with sun_col:
        st.subheader("Community Sunburst")
        st.caption("Inner ring = community (colour = avg bot score). Outer ring = top accounts.")
        st.plotly_chart(
            nv.community_sunburst(G, communities),
            use_container_width=True, key="sunburst",
        )

    st.divider()

    # ── Activity timeline ─────────────────────────────────────────────────────
    st.subheader("Activity Timeline")
    bin_h = st.select_slider(
        "Time bin size (hours)",
        options=[1, 3, 6, 12, 24],
        value=6,
    )
    st.plotly_chart(
        nv.activity_timeline(records, scores, bin_hours=bin_h),
        use_container_width=True, key="timeline",
    )

    st.divider()

    # ── Coordination heatmap ──────────────────────────────────────────────────
    st.subheader("Coordination Heatmap")
    st.caption(
        "Each row is an account; columns are hours of the day (UTC). "
        "A nearly-uniform row with activity across all 24 hours signals an automated account. "
        "Vertical stripes indicate coordinated bursts (many accounts posting at the same hour)."
    )
    top_n_heat = st.slider("Accounts to show", 10, 60, 30, step=5)
    st.plotly_chart(
        nv.coordination_heatmap(records, scores, top_n=top_n_heat),
        use_container_width=True, key="heatmap",
    )

    st.divider()

    # ── Narrative bar ─────────────────────────────────────────────────────────
    st.subheader("Disinformation Narratives")
    st.caption("Weighted by account classification: bots amplify narratives 2×, suspected 1.2×, humans 0.5×.")
    st.plotly_chart(
        nv.narrative_bar(G),
        use_container_width=True, key="narratives",
    )

    st.divider()

    # ── Top accounts table ────────────────────────────────────────────────────
    st.subheader("Top Accounts by Network Influence")
    top_df = nv.top_accounts_dataframe(G, communities, n=100)
    if not top_df.empty:
        # Colour-code rows by classification
        def _row_color(row):
            c = {"bot": "background-color:#fee2e2",
                 "suspected": "background-color:#fef9c3",
                 "human": "background-color:#dcfce7"}.get(row["class"], "")
            return [c] * len(row)

        st.dataframe(
            top_df.style.apply(_row_color, axis=1),
            use_container_width=True,
            hide_index=True,
            column_config={
                "bot_score_%": st.column_config.ProgressColumn(
                    "Bot score %", min_value=0, max_value=100, format="%.1f%%"
                ),
                "pagerank": st.column_config.NumberColumn("PageRank", format="%.6f"),
            },
        )

        csv = top_df.to_csv(index=False).encode()
        st.download_button(
            "Download as CSV", data=csv,
            file_name="network_top_accounts.csv", mime="text/csv",
        )


if __name__ == "__main__":
    main()
