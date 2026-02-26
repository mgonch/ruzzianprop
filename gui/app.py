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
.badge-bot      { background:#ef4444; color:#fff; padding:4px 14px; border-radius:9999px; font-weight:700; }
.badge-suspected{ background:#f59e0b; color:#fff; padding:4px 14px; border-radius:9999px; font-weight:700; }
.badge-human    { background:#22c55e; color:#fff; padding:4px 14px; border-radius:9999px; font-weight:700; }

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

    # ── Header ───────────────────────────────────────────────────────────────
    username = user.get("username", "unknown")
    badge_cls = f"badge-{clf}"
    badge_label = clf.upper()
    color_map = {"bot": "#ef4444", "suspected": "#f59e0b", "human": "#22c55e"}
    gauge_color = color_map[clf]

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
        <h2 style="margin:0;">@{username}</h2>
        <span class="{badge_cls}">{badge_label}</span>
        <span style="font-size:0.8rem; color:#64748b;">source: {result['source']}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Gauges row ────────────────────────────────────────────────────────────
    if result["ml_available"] and result["ml"]:
        g_col1, g_col2, g_col3 = st.columns(3)
        with g_col1:
            st.plotly_chart(
                gauge_chart(combined_pct, "Combined Score", gauge_color),
                use_container_width=True, key="gauge_combined",
            )
        with g_col2:
            ml = result["ml"]
            st.plotly_chart(
                gauge_chart(ml["pct"], "ML Model", "#6366f1"),
                use_container_width=True, key="gauge_ml",
            )
            st.caption(
                f"95% CI: {ml['lower']}% – {ml['upper']}%  "
                f"(uncertainty: ±{round(ml['uncertainty']*100,1)}%)"
            )
        with g_col3:
            st.plotly_chart(
                gauge_chart(h_pct, "Heuristic", "#64748b"),
                use_container_width=True, key="gauge_heuristic",
            )
    else:
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.plotly_chart(
                gauge_chart(combined_pct, "Bot Score (Heuristic)", gauge_color),
                use_container_width=True, key="gauge_single",
            )
        with g_col2:
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


if __name__ == "__main__":
    main()
