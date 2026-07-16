"""
Boiler Component Weight Estimator - Streamlit App (Phase 2)

Loads the trained Gradient Boosting model bundle and provides an
interactive, dark-mode, decision-support interface for conceptual /
proposal-stage boiler weight estimation.

Run with: streamlit run app.py
"""

import io
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

MODEL_PATH = "models/boiler_weight_models.joblib"

BOILER_TYPES = ["CFBC", "AFBC", "Bi-Drum", "Two-Pass"]
CAPACITIES_MW = [50, 100, 200, 250, 270, 500, 660, 800]
STEAM_FLOW_TPH = [150, 200, 300, 500, 1000, 2000]
PRESSURE_KGCM2 = [100, 150, 200, 250, 300, 325]
TEMPERATURE_C = [250, 300, 350, 400]

CATEGORY_COLORS = {
    "Pressure Parts": "#3DA9FC",
    "Structural": "#F2994A",
    "Thermal": "#EB5757",
    "Mechanical Equipment": "#27AE60",
    "Instrumentation": "#9B51E0",
    "Electrical": "#F2C94C",
    "Chemical Systems": "#56CCF2",
    "Miscellaneous": "#828282",
}

# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Boiler Weight Estimator",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; letter-spacing: -0.01em; }
    h1 { font-weight: 700 !important; }

    .app-header {
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #14181F 0%, #1B2130 100%);
        border: 1px solid #262B36;
        margin-bottom: 1.5rem;
    }
    .app-subtitle { color: #9AA4B2; font-size: 0.95rem; margin-top: -0.4rem; }

    div[data-testid="stMetric"] {
        background: #161A22;
        border: 1px solid #262B36;
        border-radius: 12px;
        padding: 1rem 1.2rem 0.8rem 1.2rem;
    }
    div[data-testid="stMetricLabel"] { color: #9AA4B2 !important; }

    .insight-card {
        background: #161A22;
        border: 1px solid #262B36;
        border-left: 4px solid #3DA9FC;
        border-radius: 10px;
        padding: 0.95rem 1.2rem;
        margin-bottom: 0.85rem;
    }
    .insight-card.warn { border-left-color: #F2994A; }
    .insight-card.good { border-left-color: #27AE60; }
    .insight-card.alert { border-left-color: #EB5757; }
    .insight-title { font-weight: 600; color: #E6E8EB; margin-bottom: 0.25rem; font-size: 0.98rem; }
    .insight-body { color: #B4BCC8; font-size: 0.9rem; line-height: 1.45; }

    .conf-badge {
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600; margin-left: 0.5rem;
    }
    .conf-high { background: rgba(39,174,96,0.18); color: #4FD684; }
    .conf-med  { background: rgba(242,153,74,0.18); color: #F5A85A; }
    .conf-low  { background: rgba(235,87,87,0.18); color: #F17E7E; }

    footer, #MainMenu { visibility: hidden; }
    .disclaimer {
        color: #6C7686; font-size: 0.8rem; border-top: 1px solid #262B36;
        padding-top: 0.9rem; margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading + prediction helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_bundle():
    """Try loading the pre-trained model bundle. If that fails for any
    reason (most commonly a scikit-learn version mismatch between the
    environment the .joblib was created in and the current one), fall
    back to training fresh, in-process, against the currently installed
    scikit-learn -- this is what makes the app portable across hosting
    environments without needing exact version pinning."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(
            f"Could not load the pre-trained model file ({type(e).__name__}). "
            "Training fresh models in this environment instead — this happens "
            "once and takes under a minute.",
            icon="⚠️",
        )
        from train_model import train_bundle
        return train_bundle(data_path="data/boiler_synthetic_dataset.csv", verbose=False)


def build_feature_row(bundle, boiler_type, capacity, steam_flow, pressure, temperature):
    row = {c: 0 for c in bundle["feature_cols"]}
    row["Capacity_MW"] = capacity
    row["Steam_Flow_TPH"] = steam_flow
    row["Rated_Pressure_kgcm2"] = pressure
    row["Rated_Steam_Temperature_C"] = temperature
    row[f"Type_{boiler_type}"] = 1
    return pd.DataFrame([row])[bundle["feature_cols"]]


def predict_all(bundle, X_row):
    preds = {}
    for target, model in bundle["models"].items():
        preds[target] = float(model.predict(X_row)[0])
    return preds


def category_totals(bundle, preds):
    totals = {}
    for cat, cols in bundle["category_groups"].items():
        totals[cat] = sum(preds[c] for c in cols)
    return totals


def confidence_label(avg_r2):
    if avg_r2 >= 0.95:
        return "High", "conf-high"
    elif avg_r2 >= 0.85:
        return "Moderate", "conf-med"
    return "Indicative", "conf-low"


def fmt_kg(v):
    if v >= 1_000_000:
        return f"{v/1_000_000:,.2f} kt"
    return f"{v:,.0f} kg"


def fmt_t(v):
    return f"{v/1000:,.1f} t"


# ---------------------------------------------------------------------------
# Sidebar - inputs
# ---------------------------------------------------------------------------

bundle = load_bundle()

with st.sidebar:
    st.markdown("### ⚙️ Boiler Configuration")
    st.caption("Select the conceptual-stage design parameters.")

    boiler_type = st.selectbox("Boiler Type", BOILER_TYPES, index=0)
    capacity = st.selectbox("Capacity (MW)", CAPACITIES_MW, index=6)
    steam_flow = st.selectbox("Steam Flow (TPH)", STEAM_FLOW_TPH, index=4)
    pressure = st.selectbox("Rated Pressure (kg/cm²)", PRESSURE_KGCM2, index=3)
    temperature = st.selectbox("Rated Steam Temperature (°C)", TEMPERATURE_C, index=3)

    st.markdown("---")
    run = st.button("🔍 Generate Estimate", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption(
        f"Model: Gradient Boosting Regressor · {len(bundle['all_targets'])} components\n\n"
        f"Trained on {bundle['training_rows']:,} synthetic samples, "
        f"validated on {bundle['test_rows']:,}."
    )

if "has_run" not in st.session_state:
    st.session_state.has_run = False
if run:
    st.session_state.has_run = True
    st.session_state.inputs = (boiler_type, capacity, steam_flow, pressure, temperature)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="app-header">
    <h1>🏭 Boiler Component Weight Estimator</h1>
    <div class="app-subtitle">
        Machine Learning based weight estimation for conceptual & proposal engineering ·
        Preliminary costing, transport planning & procurement support.
        Disclaimer:  Developed for EQPOWER as a proof of concept ;Not for prodcution.
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.has_run:
    st.info("👈 Set a boiler configuration in the sidebar and click **Generate Estimate** to begin.")
    st.stop()

boiler_type, capacity, steam_flow, pressure, temperature = st.session_state.inputs
X_row = build_feature_row(bundle, boiler_type, capacity, steam_flow, pressure, temperature)
preds = predict_all(bundle, X_row)
cat_totals = category_totals(bundle, preds)
total_weight = sum(preds.values())

avg_r2 = np.mean([bundle["metrics"][t]["r2"] for t in bundle["all_targets"]])
conf_text, conf_class = confidence_label(avg_r2)

type_stats = bundle["dataset_stats"]["total_weight_by_type"].get(boiler_type)
kg_per_mw = total_weight / capacity

# ---------------------------------------------------------------------------
# Key metrics
# ---------------------------------------------------------------------------

st.markdown(f"#### Estimate Summary — {boiler_type} · {capacity} MW · {steam_flow} TPH")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Estimated Weight", fmt_t(total_weight))
c2.metric("Weight-to-Capacity Ratio", f"{kg_per_mw:,.0f} kg/MW")
if type_stats:
    delta_vs_avg = (total_weight - type_stats["mean"]) / type_stats["mean"] * 100
    c3.metric(f"vs. {boiler_type} Average", fmt_t(type_stats["mean"]), f"{delta_vs_avg:+.1f}%")
else:
    c3.metric(f"{boiler_type} Average", "n/a")
c4.metric("Model Confidence", conf_text)
st.markdown(
    f'<span class="conf-badge {conf_class}">Avg. component R² = {avg_r2:.2f} · '
    f'MAPE ≈ {np.mean([bundle["metrics"][t]["mape"] for t in bundle["all_targets"]]):.1f}%</span>',
    unsafe_allow_html=True,
)

st.markdown("")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_breakdown, tab_details, tab_insights, tab_model = st.tabs(
    ["📊 Overview", "🧩 Category Breakdown", "📋 Component Detail", "💡 Insights", "🎯 Model Performance"]
)

# --- Overview tab ---
with tab_overview:
    col_a, col_b = st.columns([1.1, 1])

    with col_a:
        df_cat = pd.DataFrame(
            {"Category": list(cat_totals.keys()), "Weight_kg": list(cat_totals.values())}
        ).sort_values("Weight_kg", ascending=True)
        fig = go.Figure(go.Bar(
            x=df_cat["Weight_kg"],
            y=df_cat["Category"],
            orientation="h",
            marker=dict(color=[CATEGORY_COLORS[c] for c in df_cat["Category"]]),
            text=[fmt_t(v) for v in df_cat["Weight_kg"]],
            textposition="outside",
        ))
        fig.update_layout(
            title="Weight by Category",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
            margin=dict(l=10, r=60, t=50, b=10),
            xaxis_title="kg",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        df_pie = pd.DataFrame(
            {"Category": list(cat_totals.keys()), "Weight_kg": list(cat_totals.values())}
        )
        fig2 = go.Figure(go.Pie(
            labels=df_pie["Category"],
            values=df_pie["Weight_kg"],
            hole=0.55,
            marker=dict(colors=[CATEGORY_COLORS[c] for c in df_pie["Category"]]),
            textinfo="percent",
        ))
        fig2.update_layout(
            title="Share of Total Weight",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            annotations=[dict(text=fmt_t(total_weight), x=0.5, y=0.5,
                               font_size=18, showarrow=False, font_color="#E6E8EB")],
        )
        st.plotly_chart(fig2, use_container_width=True)

    top5 = pd.Series(preds).sort_values(ascending=False).head(5)
    st.markdown("##### Heaviest Individual Components")
    top5_df = pd.DataFrame({
        "Component": [c.replace("_kg", "").replace("_", " ") for c in top5.index],
        "Weight": [fmt_t(v) for v in top5.values],
        "% of Total": [f"{v/total_weight*100:.1f}%" for v in top5.values],
    })
    st.dataframe(top5_df, hide_index=True, use_container_width=True)

# --- Category breakdown tab ---
with tab_breakdown:
    for cat, cols in bundle["category_groups"].items():
        with st.expander(f"**{cat}** — {fmt_t(cat_totals[cat])} ({cat_totals[cat]/total_weight*100:.1f}% of total)"):
            sub = pd.DataFrame({
                "Component": [c.replace("_kg", "").replace("_", " ") for c in cols],
                "Weight (kg)": [f"{preds[c]:,.0f}" for c in cols],
                "Weight (t)": [f"{preds[c]/1000:,.2f}" for c in cols],
            })
            st.dataframe(sub, hide_index=True, use_container_width=True)

# --- Detail table tab ---
with tab_details:
    detail_df = pd.DataFrame({
        "Component": [c.replace("_kg", "").replace("_", " ") for c in bundle["all_targets"]],
        "Category": [cat for c in bundle["all_targets"]
                     for cat, cols in bundle["category_groups"].items() if c in cols],
        "Estimated Weight (kg)": [round(preds[c], 1) for c in bundle["all_targets"]],
        "Component R²": [bundle["metrics"][c]["r2"] for c in bundle["all_targets"]],
    }).sort_values("Estimated Weight (kg)", ascending=False)

    st.dataframe(detail_df, hide_index=True, use_container_width=True, height=520)

    csv_buf = io.StringIO()
    export_df = detail_df.copy()
    export_df.insert(0, "Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M"))
    export_df.insert(1, "Boiler_Type", boiler_type)
    export_df.insert(2, "Capacity_MW", capacity)
    export_df.insert(3, "Steam_Flow_TPH", steam_flow)
    export_df.insert(4, "Pressure_kgcm2", pressure)
    export_df.insert(5, "Temperature_C", temperature)
    export_df.to_csv(csv_buf, index=False)

    st.download_button(
        "⬇️ Download Full Estimate (CSV)",
        data=csv_buf.getvalue(),
        file_name=f"boiler_weight_estimate_{boiler_type}_{capacity}MW.csv",
        mime="text/csv",
        use_container_width=True,
    )

# --- Insights tab ---
with tab_insights:
    st.markdown("##### Decision-Support Insights")

    # Insight 1: dominant category
    dominant_cat = max(cat_totals, key=cat_totals.get)
    dominant_share = cat_totals[dominant_cat] / total_weight * 100
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">🏗️ {dominant_cat} drives this estimate</div>
        <div class="insight-body">{dominant_cat} accounts for {dominant_share:.1f}% of total weight
        ({fmt_t(cat_totals[dominant_cat])}). This is typically the largest lever for cost and schedule —
        prioritize vendor quotes and engineering review here first.</div>
    </div>
    """, unsafe_allow_html=True)

    # Insight 2: heaviest single component -> logistics/crane flag
    heaviest_series = pd.Series(preds).sort_values(ascending=False)
    heaviest_name, heaviest_val = heaviest_series.index[0], float(heaviest_series.iloc[0])
    heaviest_label = heaviest_name.replace("_kg", "").replace("_", " ")
    logistics_class = "warn" if heaviest_val > 3_000_000 else ""
    st.markdown(f"""
    <div class="insight-card {logistics_class}">
        <div class="insight-title">🚛 Heaviest single item: {heaviest_label}</div>
        <div class="insight-body">Estimated at {fmt_t(heaviest_val)}. If this exceeds site crane or
        transport corridor limits, plan for sub-assembly / modular delivery early — this is one of the
        first constraints that shapes site logistics planning.</div>
    </div>
    """, unsafe_allow_html=True)

    # Insight 3: comparison vs boiler-type average
    if type_stats:
        delta = (total_weight - type_stats["mean"]) / type_stats["mean"] * 100
        cls = "good" if abs(delta) < 10 else "warn"
        direction = "above" if delta > 0 else "below"
        st.markdown(f"""
        <div class="insight-card {cls}">
            <div class="insight-title">📈 {abs(delta):.1f}% {direction} the {boiler_type} fleet average</div>
            <div class="insight-body">Across the training dataset, {boiler_type} boilers average
            {fmt_t(type_stats['mean'])} (range {fmt_t(type_stats['min'])} – {fmt_t(type_stats['max'])}).
            A large deviation is expected at capacity extremes; use this as a sanity check against your
            own reference projects.</div>
        </div>
        """, unsafe_allow_html=True)

    # Insight 4: type-specific structural note
    type_notes = {
        "CFBC": "CFBC design typically carries heavier buckstay and sealing systems due to the refractory-lined "
                "combustor and solids-recycle system — verify structural steel and refractory budgets accordingly.",
        "Bi-Drum": "Bi-Drum design carries a materially heavier steam drum assembly than single-drum types — "
                   "confirm crane capacity and drum transport route early, as this is often a critical-path item.",
        "AFBC": "AFBC is the baseline configuration in this model — no additional structural premium applied.",
        "Two-Pass": "Two-Pass arrangement trends lighter on structural/general weight due to a more compact "
                    "layout — this can translate into reduced foundation and steel costs versus other types.",
    }
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">🔧 {boiler_type}-specific consideration</div>
        <div class="insight-body">{type_notes[boiler_type]}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### What-If: Sensitivity to Design Choices")
    st.caption("Holding all other parameters fixed, how does total weight change if you adjust one variable?")

    sens_col1, sens_col2 = st.columns(2)

    with sens_col1:
        st.markdown("**Sensitivity to Capacity**")
        rows = []
        for cap in CAPACITIES_MW:
            Xr = build_feature_row(bundle, boiler_type, cap, steam_flow, pressure, temperature)
            p = predict_all(bundle, Xr)
            rows.append({"Capacity_MW": cap, "Total_Weight_t": sum(p.values()) / 1000})
        sens_df = pd.DataFrame(rows)
        fig3 = px.line(sens_df, x="Capacity_MW", y="Total_Weight_t", markers=True)
        fig3.add_scatter(x=[capacity], y=[total_weight/1000], mode="markers",
                          marker=dict(size=14, color="#EB5757"), name="Current selection")
        fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)", height=320,
                            margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                            yaxis_title="Total weight (t)")
        st.plotly_chart(fig3, use_container_width=True)

    with sens_col2:
        st.markdown("**Sensitivity to Boiler Type** (same capacity/flow/pressure/temp)")
        rows = []
        for bt in BOILER_TYPES:
            Xr = build_feature_row(bundle, bt, capacity, steam_flow, pressure, temperature)
            p = predict_all(bundle, Xr)
            rows.append({"Boiler_Type": bt, "Total_Weight_t": sum(p.values()) / 1000})
        sens_df2 = pd.DataFrame(rows)
        colors = ["#EB5757" if bt == boiler_type else "#3DA9FC" for bt in sens_df2["Boiler_Type"]]
        fig4 = go.Figure(go.Bar(x=sens_df2["Boiler_Type"], y=sens_df2["Total_Weight_t"],
                                 marker_color=colors,
                                 text=[f"{v:,.0f} t" for v in sens_df2["Total_Weight_t"]],
                                 textposition="outside"))
        fig4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)", height=320,
                            margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Total weight (t)")
        st.plotly_chart(fig4, use_container_width=True)

# --- Model performance tab ---
with tab_model:
    st.markdown("##### Global Feature Importance")
    st.caption("Averaged across all 33 component models — shows what drives predictions overall.")

    fi = bundle["feature_importance"]
    fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
    fig5 = go.Figure(go.Bar(
        x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
        marker_color="#3DA9FC",
    ))
    fig5.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", height=380,
                        margin=dict(l=10, r=10, t=10, b=10),
                        yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("##### Per-Component Model Accuracy (held-out test set)")
    metrics_df = pd.DataFrame(bundle["metrics"]).T.reset_index()
    metrics_df.columns = ["Component", "R²", "MAE (kg)", "MAPE (%)"]
    metrics_df["Component"] = metrics_df["Component"].str.replace("_kg", "").str.replace("_", " ")
    metrics_df = metrics_df.sort_values("R²", ascending=False)
    st.dataframe(metrics_df, hide_index=True, use_container_width=True, height=420)

    st.caption(
        f"Trained on {bundle['training_rows']:,} rows · validated on {bundle['test_rows']:,} held-out rows. "
        "MAPE near 7–8% is expected and consistent with the ±5–10% engineering variability built into the "
        "synthetic training data — this reflects the noise injected in dataset generation, not a modeling flaw."
    )

st.markdown(
    '<div class="disclaimer">Estimates are generated by a Gradient Boosting model trained on a synthetic, '
    'engineering-assumption-based dataset (see Phase 1 documentation). Treat outputs as conceptual-stage '
    'planning figures only — validate against vendor quotations and detailed engineering before committing '
    'to procurement, transport, or cost decisions.</div>',
    unsafe_allow_html=True,
)
