"""
Streamlit dashboard for the AI Talent Intelligence Platform.

Layout:
  ┌─────────────────────────────────────────────────────┐
  │  Sidebar: API base URL + pipeline settings          │
  ├─────────────────────────────────────────────────────┤
  │  Tab 1: Rank — paste JD → run pipeline → table      │
  │  Tab 2: Top 100 — cached ranking with score bars    │
  │  Tab 3: Analytics — score distribution charts       │
  └─────────────────────────────────────────────────────┘

Runs against the FastAPI server (no direct model calls from Streamlit).
Start with:
    streamlit run ui/app.py
"""

from __future__ import annotations

import time
from typing import Any

import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Talent Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e2030, #252840);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e2e8f0;
    }

    .rank-badge {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .score-bar-bg {
        background: #1e293b;
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
    }
    .score-bar-fill {
        background: linear-gradient(90deg, #6366f1, #06b6d4);
        border-radius: 6px;
        height: 8px;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        background: #1a1f2e !important;
    }
    div[data-testid="stTabs"] button {
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_url = st.text_input(
        "API Base URL",
        value="http://localhost:8000",
        help="FastAPI server address (must be running).",
    )
    retrieval_k = st.slider("Retrieval Top-K", 100, 2000, 500, 100)
    rerank_k = st.slider("Rerank Top-K", 50, 500, 200, 50)

    st.markdown("---")
    if st.button("🔍 Check API Health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                st.success(
                    f"✅ API online — {data['num_candidates']:,} candidates indexed"
                )
            else:
                st.error(f"API returned {r.status_code}")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")

    st.markdown("---")
    st.markdown(
        "<small style='color:#475569;'>AI Talent Intelligence Platform v1.0</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="hero-title">🎯 AI Talent Intelligence Platform</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-sub">Expert recruiter-grade candidate ranking '
    '— FAISS retrieval · Cross-encoder reranking · Hybrid scoring</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_rank, tab_top100, tab_analytics = st.tabs(
    ["🚀 Rank Candidates", "📋 Top 100", "📊 Analytics"]
)

# ── Tab 1: Rank ─────────────────────────────────────────────────────────────

with tab_rank:
    st.markdown("### Paste or type a Job Description")
    jd_text = st.text_area(
        "Job Description",
        height=220,
        placeholder=(
            "Senior AI / ML Engineer required for building production ranking systems.\n"
            "Must have: Python, FAISS, sentence-transformers, NLP, PyTorch.\n"
            "Nice to have: Kubernetes, MLOps, LLM fine-tuning experience.\n"
            "5–10 years of experience preferred."
        ),
        label_visibility="collapsed",
    )

    col_btn, col_spacer = st.columns([1, 4])
    with col_btn:
        run_pipeline = st.button("⚡ Run Pipeline", type="primary", use_container_width=True)

    if run_pipeline:
        if len(jd_text.strip()) < 50:
            st.warning("Please enter at least 50 characters of job description text.")
        else:
            with st.spinner("Running full pipeline (FAISS → CrossEncoder → Scoring)..."):
                t0 = time.time()
                try:
                    resp = requests.post(
                        f"{api_url}/rank",
                        json={
                            "jd_text": jd_text,
                            "retrieval_top_k": retrieval_k,
                            "rerank_top_k": rerank_k,
                        },
                        timeout=300,
                    )
                    if resp.status_code != 200:
                        st.error(f"API error {resp.status_code}: {resp.text[:300]}")
                    else:
                        data = resp.json()
                        st.session_state["last_rank_data"] = data
                        elapsed = time.time() - t0

                        # ── Metrics row
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric("Pipeline Time", f"{data['pipeline_time_ms']:.0f} ms")
                        with m2:
                            st.metric("Candidates Indexed", f"{data['num_candidates_indexed']:,}")
                        with m3:
                            st.metric("Honeypots Excluded", data["num_honeypots_excluded"])
                        with m4:
                            top_score = data["top_100"][0]["final_score"] if data["top_100"] else 0
                            st.metric("Top Score", f"{top_score:.1f}/100")

                        # ── Ranking table
                        st.markdown("### 🏆 Top 100 Candidates")
                        for entry in data["top_100"]:
                            score_pct = entry["final_score"]
                            bar_w = int(score_pct)
                            with st.expander(
                                f"#{entry['rank']}  ·  {entry['candidate_id']}  ·  "
                                f"Score: {score_pct:.1f}/100",
                                expanded=(entry["rank"] <= 3),
                            ):
                                c1, c2 = st.columns([2, 1])
                                with c1:
                                    st.markdown(
                                        f"<div class='score-bar-bg'>"
                                        f"<div class='score-bar-fill' style='width:{bar_w}%'></div>"
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )
                                    st.markdown(
                                        f"<small style='color:#94a3b8;'>{entry['reasoning']}</small>",
                                        unsafe_allow_html=True,
                                    )
                                with c2:
                                    st.markdown(
                                        f"| Signal | Score |\n|---|---|\n"
                                        f"| 🔵 Semantic | {entry['semantic_similarity']*100:.1f}% |\n"
                                        f"| 🟢 Cross-Enc | {entry['cross_encoder_score']*100:.1f}% |\n"
                                        f"| 🟠 Experience | {entry['experience_score']*100:.1f}% |\n"
                                        f"| 🟣 Redrob | {entry['redrob_signal_score']*100:.1f}% |\n"
                                        f"| 📚 Education | {entry['education_score']*100:.1f}% |\n"
                                        f"| 🏅 Certs | {entry['certification_score']*100:.1f}% |"
                                    )

                except requests.exceptions.ConnectionError:
                    st.error(
                        f"Cannot connect to API at {api_url}. "
                        "Start the server with: `uvicorn src.api.server:app --reload`"
                    )
                except Exception as exc:
                    st.exception(exc)

# ── Tab 2: Top 100 ──────────────────────────────────────────────────────────

with tab_top100:
    st.markdown("### Latest Cached Ranking")
    col_refresh, _ = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            try:
                r = requests.get(f"{api_url}/top100", timeout=10)
                if r.status_code == 200:
                    st.session_state["top100_data"] = r.json()
                else:
                    st.info("No ranking cached yet. Run the pipeline first.")
            except Exception as e:
                st.error(str(e))

    top100_data: list[dict[str, Any]] = st.session_state.get(
        "top100_data",
        st.session_state.get("last_rank_data", {}).get("top_100", []),
    )

    if top100_data:
        import pandas as pd

        df = pd.DataFrame(top100_data)[
            ["rank", "candidate_id", "final_score",
             "semantic_similarity", "cross_encoder_score",
             "experience_score", "education_score",
             "certification_score", "redrob_signal_score"]
        ]
        df.columns = [
            "Rank", "Candidate ID", "Final Score",
            "Semantic", "Cross-Enc",
            "Experience", "Education", "Certs", "Redrob Signal",
        ]
        for col in df.columns[2:]:
            df[col] = df[col].round(4)
        st.dataframe(df, use_container_width=True, height=600)
    else:
        st.info("Run the pipeline on the **Rank Candidates** tab to populate this view.")

# ── Tab 3: Analytics ────────────────────────────────────────────────────────

with tab_analytics:
    st.markdown("### Score Distribution Analytics")

    chart_data: list[dict[str, Any]] = st.session_state.get(
        "last_rank_data", {}
    ).get("top_100", [])

    if not chart_data:
        st.info("Run the pipeline first to view analytics.")
    else:
        import pandas as pd

        df = pd.DataFrame(chart_data)

        c1, c2 = st.columns(2)

        with c1:
            fig_hist = px.histogram(
                df,
                x="final_score",
                nbins=20,
                title="Final Score Distribution (Top 100)",
                labels={"final_score": "Final Score (0–100)"},
                color_discrete_sequence=["#6366f1"],
                template="plotly_dark",
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_family="Inter",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            components = {
                "Semantic (40%)": df["semantic_similarity"].mean() * 100,
                "Cross-Enc (30%)": df["cross_encoder_score"].mean() * 100,
                "Experience (10%)": df["experience_score"].mean() * 100,
                "Redrob (10%)": df["redrob_signal_score"].mean() * 100,
                "Education (5%)": df["education_score"].mean() * 100,
                "Certs (5%)": df["certification_score"].mean() * 100,
            }
            fig_bar = px.bar(
                x=list(components.keys()),
                y=list(components.values()),
                title="Average Component Scores — Top 100",
                labels={"x": "Component", "y": "Avg Score (%)"},
                color=list(components.values()),
                color_continuous_scale="Viridis",
                template="plotly_dark",
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_family="Inter",
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Scatter: semantic vs cross-encoder
        fig_scatter = px.scatter(
            df,
            x="semantic_similarity",
            y="cross_encoder_score",
            size="final_score",
            color="final_score",
            hover_data=["candidate_id", "rank", "final_score"],
            title="Semantic Similarity vs Cross-Encoder Score",
            labels={
                "semantic_similarity": "Semantic Similarity",
                "cross_encoder_score": "Cross-Encoder Score",
                "final_score": "Final Score",
            },
            color_continuous_scale="Plasma",
            template="plotly_dark",
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
