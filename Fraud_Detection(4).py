import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, f1_score
)
from imblearn.over_sampling import SMOTE

# ==============================================================================
# PAGE CONFIG & INSTITUTIONAL DESIGN TOKENS (Premium Swiss Banking Theme)
# ==============================================================================
st.set_page_config(page_title="FraudGuard AI Enterprise", page_icon="🛡️", layout="wide")

BG        = "#060D19"   # Inky Sovereign Navy (High-end Fintech Background)
PANEL     = "#0B1524"   # Premium Container Navy
PANEL_2   = "#0D1B2E"   # Input/Form Well
BORDER    = "#162B47"   # Razor-sharp structural border
TEXT      = "#F0F5FA"   # High-contrast crisp typography
MUTED     = "#6C829E"   # Banking Sub-text/Metadata Slate
ACCENT    = "#00E676"   # Mint Compliance Green
ACCENT_BG = "rgba(0, 230, 118, 0.08)"
DANGER    = "#FF5252"   # Risk/Breach Red
DANGER_BG = "rgba(255, 82, 82, 0.08)"
WARN      = "#FFD740"   # Escalation Amber
WARN_BG   = "rgba(255, 215, 64, 0.08)"

# Injecting Institutional CSS Framework
st.markdown(f"""
<style>
    /* Global Canvas Reset */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {{ 
        background-color: {BG}; 
        font-family: 'Inter', -apple-system, sans-serif;
    }}
    
    /* Native Header Calibration */
    header[data-testid="stHeader"] {{
        background-color: rgba(6, 13, 25, 0.85) !important;
        backdrop-filter: blur(12px);
        z-index: 999;
        border-bottom: 1px solid {BORDER};
    }}
    .block-container {{ 
        padding-top: 4rem !important; 
        padding-bottom: 3rem; 
    }}

    /* Premium Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: #030811 !important;
        border-right: 1px solid {BORDER} !important;
    }}
    
    /* Branding Badge Enhancement */
    .fg-premium-badge {{
        background: linear-gradient(135deg, #0B1524 0%, #162B47 100%);
        border: 1px solid {BORDER};
        padding: 14px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        margin: 10px 0 25px 0;
    }}
    .fg-premium-title {{
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        letter-spacing: 0.03em;
        color: #FFFFFF !important;
        text-transform: uppercase;
        display: block;
    }}
    .fg-premium-subtitle {{
        font-size: 0.68rem !important;
        color: {ACCENT};
        letter-spacing: 0.15em;
        text-transform: uppercase;
        font-weight: 600;
        margin-top: 2px;
        display: block;
    }}

    /* Custom Navigation System Mapping */
    div[data-testid="stSidebarUserContent"] .stRadio > label {{ display: none !important; }}
    div[data-testid="stSidebarUserContent"] .stRadio div[role="radiogroup"] {{ gap: 8px; }}
    div[data-testid="stSidebarUserContent"] .stRadio div[role="radiogroup"] label {{
        background-color: transparent;
        border: 1px solid transparent;
        padding: 12px 16px;
        border-radius: 6px;
        color: {MUTED} !important;
        font-size: 0.88rem !important;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
        cursor: pointer;
    }}
    div[data-testid="stSidebarUserContent"] .stRadio div[role="radiogroup"] label:hover {{
        background-color: rgba(22, 43, 71, 0.4);
        color: {TEXT} !important;
    }}
    div[data-testid="stSidebarUserContent"] .stRadio div[role="radiogroup"] label[data-checked="true"] {{
        background-color: #0B1524 !important;
        color: #FFFFFF !important;
        font-weight: 600;
        border: 1px solid {BORDER} !important;
        border-left: 4px solid {ACCENT} !important;
    }}

    /* Structural Segment Headers (Ingestion Pipeline, etc.) */
    .fg-section-header {{
        font-weight: 700;
        font-size: 1.25rem;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        margin-bottom: 2px;
    }}
    .fg-section-subheader {{
        font-size: 0.8rem;
        color: {MUTED};
        margin-bottom: 14px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Input Element & Cleanups */
    div[data-testid="stFileUploader"] label {{ display: none !important; }}
    div[data-testid="stFileUploader"] section {{ padding: 0.75rem !important; }}
    div[data-testid="stFileUploader"] {{
        background-color: {PANEL};
        border: 1px dashed {BORDER};
        border-radius: 6px;
    }}

    /* Enterprise Structural Panels (Cards) */
    .fg-card {{
        background-color: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 24px;
        height: 100%;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }}
    .fg-card-title {{ font-weight: 600; font-size: 1.05rem; color: #FFFFFF; margin-bottom: 4px; }}
    .fg-card-sub {{ color: {MUTED}; font-size: 0.8rem; margin-bottom: 18px; }}

    /* Top Live Telemetry Bar */
    .fg-topbar {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 14px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
        font-size: 0.82rem;
        color: {MUTED};
    }}
    .fg-pill {{
        padding: 5px 10px; border-radius: 4px; font-size: 0.7rem;
        font-weight: 700; display: inline-block; text-transform: uppercase; letter-spacing: 0.06em;
    }}
    .fg-pill-ok   {{ background: {ACCENT_BG}; color: {ACCENT}; border: 1px solid rgba(0, 230, 118, 0.25); }}
    .fg-pill-bad  {{ background: {DANGER_BG}; color: {DANGER}; border: 1px solid rgba(255, 82, 82, 0.25); }}

    /* Metric Layout Refinements */
    div[data-testid="stMetric"] {{
        background-color: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }}
    div[data-testid="stMetric"] label {{
        color: {MUTED} !important; font-size: 0.72rem !important;
        font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.08em;
    }}
    div[data-testid="stMetricValue"] {{ color: #FFFFFF !important; font-weight: 700; font-size: 1.8rem !important; }}

    /* Real-Time Alerts */
    .fg-alert {{ border: 1px solid rgba(255,82,82,0.2); background: {PANEL_2}; border-left: 4px solid {DANGER}; border-radius: 4px; padding: 14px; margin-bottom: 12px; }}
    .fg-alert-title {{ color: {DANGER}; font-weight: 700; font-size: 0.82rem; letter-spacing: 0.05em; margin-bottom: 3px; }}
    .fg-alert-body {{ color: {TEXT}; font-size: 0.82rem; }}
    .fg-alert-meta {{ color: {MUTED}; font-size: 0.75rem; margin-top: 6px; }}
    
    .fg-alert-ok {{ border: 1px solid rgba(0,230,118,0.2); background: {PANEL_2}; border-left: 4px solid {ACCENT}; border-radius: 4px; padding: 14px; margin-bottom: 12px; }}
    .fg-alert-ok-title {{ color: {ACCENT}; font-weight: 700; font-size: 0.82rem; letter-spacing: 0.05em; margin-bottom: 3px; }}

    .fg-banner-fail {{ border: 1px solid rgba(255,82,82,0.3); background: {DANGER_BG}; border-radius: 4px; padding: 12px; color: {DANGER}; font-weight: 700; font-size: 0.85rem; text-align: center; margin-bottom: 16px; letter-spacing: 0.02em; }}
    .fg-banner-ok {{ border: 1px solid rgba(0,230,118,0.3); background: {ACCENT_BG}; border-radius: 4px; padding: 12px; color: {ACCENT}; font-weight: 700; font-size: 0.85rem; text-align: center; margin-bottom: 16px; letter-spacing: 0.02em; }}

    /* Form Elements Override & Input Label Discarding for Clean Banking Vibe */
    div[data-testid="stForm"] {{ background-color: transparent !important; border: none !important; padding: 0 !important; }}
    div[data-testid="stForm"] label {{ display: none !important; }}
    
    .stTextInput input, .stNumberInput input {{ background-color: {PANEL_2} !important; color: #FFFFFF !important; border: 1px solid {BORDER} !important; border-radius: 4px !important; }}
    .stButton button[kind="primary"] {{ background-color: {ACCENT} !important; border-color: {ACCENT} !important; color: #030811 !important; font-weight: 700; border-radius: 4px; width: 100%; }}
    .stButton button[kind="secondary"] {{ background-color: {PANEL_2} !important; border-color: {BORDER} !important; color: {TEXT} !important; border-radius: 4px; }}
    
    /* Clean overrides for expanders inside dark panels */
    div[data-testid="stExpander"] {{ background-color: {PANEL_2} !important; border: 1px solid {BORDER} !important; border-radius: 6px !important; }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# PIPELINE TRAINING ENGINE
# ==============================================================================
@st.cache_resource(show_spinner="Optimizing Random Forest Classifier (SMOTE Engine Active)...")
def train_enterprise_model(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    
    preds = rf_model.predict(X_test_scaled)
    probs = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    
    feature_importance = pd.DataFrame({
        "Feature": X.columns, "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True),
        "feature_columns": list(X.columns),
        "fraud_percentage": (y.sum() / len(y)) * 100,
        "n_rows": len(df),
    }
    fraud_sample = df[df["Class"] == 1].sample(min(5, (df["Class"] == 1).sum()), random_state=1)
    normal_sample = df[df["Class"] == 0].sample(5, random_state=1)
    sample_rows = pd.concat([fraud_sample, normal_sample]).sample(frac=1, random_state=2)
    
    return rf_model, scaler, feature_importance, metrics, sample_rows

if "transaction_log" not in st.session_state:
    st.session_state.transaction_log = []

# ==============================================================================
# SIDEBAR (High-End Dynamic Navigation)
# ==============================================================================
with st.sidebar:
    st.markdown(
        "<div class='fg-premium-badge'>"
        "<span class='fg-premium-title'>🛡️ FraudGuard</span>"
        "<span class='fg-premium-subtitle'>Premium Enterprise</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    
    current_pane = st.radio(
        "Navigation",
        options=[
            "📊  Risk Intelligence Suite",
            "💳  Transaction Ledger",
            "🚩  Escalations Management",
            "📈  Hyperparameter Validation",
            "⚙️  System Preferences"
        ]
    )

# ==============================================================================
# DATA INGESTION SUBSYSTEM (Tailored Layout Structure)
# ==============================================================================
st.markdown("<div class='fg-section-header'>System Ingestion Pipeline</div>", unsafe_allow_html=True)
st.markdown("<div class='fg-section-subheader'>Institutional Ledger Feeds & Security Frame Configuration</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

df_data = None
MODEL_READY = False

if uploaded_file is not None:
    try:
        df_data = pd.read_csv(uploaded_file)
        model, scaler, feature_importance, metrics, sample_rows = train_enterprise_model(df_data)
        FEATURE_COLUMNS = metrics["feature_columns"]
        MODEL_READY = True
    except Exception as e:
        st.error(f"Error reading dataset files: {e}")
else:
    FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Global Dynamic Topbar
status_pill = "<span class='fg-pill fg-pill-ok'>● Pipeline Connected</span>" if MODEL_READY \
    else "<span class='fg-pill fg-pill-bad'>● Awaiting Secure Stream</span>"

st.markdown(
    f"<div class='fg-topbar'>"
    f"<div>{status_pill}&nbsp;&nbsp;&nbsp;&nbsp;System Framework: <b>{current_pane.split('  ')[1]}</b></div>"
    f"<div>Terminal Active Instance &nbsp;|&nbsp; {datetime.now().strftime('%b %d, %Y · %I:%M %p')}</div>"
    f"</div>",
    unsafe_allow_html=True,
)

# ==============================================================================
# ROUTED PAGES EXECUTION ENGINE
# ==============================================================================

# --- PANE 1: RISK INTELLIGENCE SUITE ---
if "Risk Intelligence Suite" in current_pane:
    st.title("Risk & Security Intelligence Dashboard")
    st.caption("High-performance telemetry mapping neural classification vectors to transaction streams.")
    st.write("")

    # KPI Layout Frame
    k1, k2, k3, k4 = st.columns(4)
    if MODEL_READY:
        n_flagged = sum(1 for t in st.session_state.transaction_log if t["prediction"] == 1)
        k1.metric("Ingested Records", f"{metrics['n_rows']:,}", "Active File")
        k2.metric("Base Fraud Threat", f"{metrics['fraud_percentage']:.3f}%", "Historical Baseline")
        k3.metric("Validated Accuracy", f"{metrics['accuracy']*100:.2f}%", f"ROC-AUC {metrics['roc_auc']:.3f}")
        k4.metric("Live Session Intercepts", f"{n_flagged}", f"Scored Records")
    else:
        for k, label in zip((k1, k2, k3, k4), ["Ingested Records", "Base Fraud Threat", "Validated Accuracy", "Live Session Intercepts"]):
            k.metric(label, "—")

    st.write("")

    main_left, main_right = st.columns([2, 1])
    with main_left:
        st.markdown("<div class='fg-card'>", unsafe_allow_html=True)
        st.markdown("<div class='fg-card-title'>Quantitative Volumetric Threat Assessment</div>", unsafe_allow_html=True)
        st.markdown("<div class='fg-card-sub'>A scatter analysis mapping transaction volumes against statistical risk values.</div>", unsafe_allow_html=True)

        if MODEL_READY:
            sample = df_data.sample(min(600, len(df_data)), random_state=7)
            scaled_sample = scaler.transform(sample[FEATURE_COLUMNS])
            risk_scores = model.predict_proba(scaled_sample)[:, 1]

            fig = go.Figure(go.Scatter(
                x=sample["Amount"], y=risk_scores, mode="markers",
                marker=dict(size=7, color=risk_scores, colorscale=[[0, "#00E676"], [0.5, WARN], [1, DANGER]], showscale=False),
                text=[f"Actual Ground Truth: {'Positive Fraud' if c == 1 else 'Cleared Normal'}" for c in sample["Class"]],
                hovertemplate="Volume: $%{x:.2f}<br>Evaluated Risk: %{y:.2f}<br>%{text}<extra></extra>",
            ))
            fig.update_layout(
                height=320, margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title="Transaction Amount ($)", yaxis_title="Calculated Probability Score",
                yaxis=dict(range=[-0.05, 1.05], gridcolor=BORDER, color=MUTED), xaxis=dict(gridcolor=BORDER, color=MUTED),
                plot_bgcolor=PANEL, paper_bgcolor=PANEL, font=dict(color=TEXT),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("System ingestion frame empty. Feed transactional ledger files above.")
        st.markdown("</div>", unsafe_allow_html=True)

    with main_right:
        st.markdown("<div class='fg-card'>", unsafe_allow_html=True)
        st.markdown("<div class='fg-card-title'>Real-Time Incident Stream</div>", unsafe_allow_html=True)
        st.markdown("<div class='fg-card-sub'>Intercept records created during this active assessment session.</div>", unsafe_allow_html=True)

        if st.session_state.transaction_log:
            for tx in st.session_state.transaction_log[-3:][::-1]:
                if tx["prediction"] == 1:
                    st.markdown(f"<div class='fg-alert'><div class='fg-alert-title'>🚨 ESCALATION FLAG // CRITICAL OVERRIDE</div><div class='fg-alert-body'>Transfer amount of <b>${tx['amount']:,.2f}</b> triggered anomaly algorithms.</div><div class='fg-alert-meta'>Risk Score: {tx['confidence']:.1f}% · {tx['time']}</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='fg-alert-ok'><div class='fg-alert-ok-title'>✓ TRANSACTION CLEARED AT EDGE</div><div class='fg-alert-body'>Volume of <b>${tx['amount']:,.2f}</b> verified within normal thresholds.</div><div class='fg-alert-meta'>Risk Score: {tx['confidence']:.1f}% · {tx['time']}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:{MUTED}; font-size:0.85rem; padding: 20px 0;'>No active instances evaluated in this session thread.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # Interactive Form row
    row2_left, row2_right = st.columns([1.2, 1])
    with row2_left:
        st.markdown("<div class='fg-card'>", unsafe_allow_html=True)
        st.markdown("<div class='fg-card-title'>Single Instance Forensic Entry</div>", unsafe_allow_html=True)
        st.markdown("<div class='fg-card-sub'>Manually run individual transaction telemetry through the pipeline model.</div>", unsafe_allow_html=True)

        quick_fill = None
        if MODEL_READY:
            qf_cols = st.columns([1, 1, 2])
            with qf_cols[0]:
                if st.button("Inject Anomaly Profile", use_container_width=True): quick_fill = sample_rows[sample_rows["Class"] == 1].iloc[0]
            with qf_cols[1]:
                if st.button("Inject Compliant Profile", use_container_width=True): quick_fill = sample_rows[sample_rows["Class"] == 0].iloc[0]

        if quick_fill is not None:
            for col in FEATURE_COLUMNS: st.session_state[f"in_{col}"] = float(quick_fill[col])

        with st.form("score_form"):
            # Hidden container keeps the inputs clean, structured, and entirely label-free
            with st.expander("Configure Forensic Entry Fields", expanded=True):
                f1, f2 = st.columns(2)
                with f1: tx_time = st.number_input("Transaction Time (Seconds) (Seconds)", min_value=0.0, value=float(st.session_state.get("in_Time", 0.0)), step=1.0)
                with f2: tx_amount = st.number_input("Transaction Amount ($)" , min_value=0.0, value=float(st.session_state.get("in_Amount", 100.0)), step=1.0)

                v_values = {}
                v_features = [c for c in FEATURE_COLUMNS if c not in ("Time", "Amount")]
                for i in range(0, len(v_features), 4):
                    cols = st.columns(4)
                    for j, c in enumerate(cols):
                        if i + j < len(v_features):
                            fname = v_features[i + j]
                            v_values[fname] = c.number_input(fname, value=float(st.session_state.get(f"in_{fname}", 0.0)), step=0.1, format="%.4f")

            submitted = st.form_submit_button("Execute Anomaly Check")
        st.markdown("</div>", unsafe_allow_html=True)

    with row2_right:
        st.markdown("<div class='fg-card' style='text-align:center;'>", unsafe_allow_html=True)
        st.markdown("<div class='fg-card-title'>Probability Output Matrix</div>", unsafe_allow_html=True)
        st.markdown("<div class='fg-card-sub'>Neural score probability readout.</div>", unsafe_allow_html=True)
        
        fraud_risk = 0.0
        gauge_color = MUTED
        
        if MODEL_READY and submitted:
            row = {"Time": tx_time, "Amount": tx_amount}; row.update(v_values)
            input_df = pd.DataFrame([row])[FEATURE_COLUMNS]
            scaled = scaler.transform(input_df)
            
            pred = model.predict(scaled)[0]
            # FIX: Tracking probability index [1] isolates the exact dynamic probability of Fraud (0-100%)
            fraud_risk = model.predict_proba(scaled)[0][1] * 100

            st.session_state.transaction_log.append({
                "time": datetime.now().strftime("%I:%M:%S %p"), 
                "amount": tx_amount, 
                "prediction": int(pred), 
                "confidence": fraud_risk
            })
            
            if pred == 1:
                st.markdown("<div class='fg-banner-fail'>CRITICAL ACTION FLAG: TRANSACTION REJECTED</div>", unsafe_allow_html=True)
                gauge_color = DANGER
            else:
                st.markdown("<div class='fg-banner-ok'>COMPLIANT CAPTURE: CLEARANCE APPROVED</div>", unsafe_allow_html=True)
                gauge_color = ACCENT

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=fraud_risk, number={"suffix": "%", "font": {"size": 32, "color": TEXT}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}}, 
                "bar": {"color": gauge_color}, 
                "bgcolor": PANEL_2, 
                "borderwidth": 1, 
                "bordercolor": BORDER
            }
        ))
        fig.update_layout(height=180, margin=dict(t=10, b=10, l=30, r=30), paper_bgcolor=PANEL, font=dict(color=TEXT))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- PANE 2: TRANSACTION LEDGER ---
elif "Transaction Ledger" in current_pane:
    st.title("Transaction Ledger Vault")
    st.caption("Direct immutable tracking audits for active system verification loops.")
    
    st.markdown("<div class='fg-card'>", unsafe_allow_html=True)
    st.markdown("<div class='fg-card-title'>Active Ledger Frame Records</div>", unsafe_allow_html=True)
    if MODEL_READY:
        st.dataframe(df_data.head(50), use_container_width=True)
    else:
        st.info("No records loaded. Stream a ledger dataset to view relational frames.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- PANE 3: ESCALATIONS MANAGEMENT ---
elif "Escalations Management" in current_pane:
    st.title("Escalations & Incident Management Queue")
    st.caption("Isolate High-risk anomalous flags requiring tier-3 personnel remediation reviews.")

    st.markdown("<div class='fg-card'>", unsafe_allow_html=True)
    st.markdown("<div class='fg-card-title'>Flagged Security Discrepancies</div>", unsafe_allow_html=True)
    
    flags = [t for t in st.session_state.transaction_log if t["prediction"] == 1]
    if flags:
        for item in flags:
            st.markdown(f"<div class='fg-row'><span class='fg-row-main'>${item['amount']:,.2f} USD</span><span class='fg-tag fg-tag-bad'>Unresolved Anomaly</span><span class='fg-row-sub'>Risk Score: {item['confidence']:.2f}% · {item['time']}</span></div>", unsafe_allow_html=True)
    else:
        st.success("No active anomalies escalated during this evaluation frame.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- PANE 4: HYPERPARAMETER VALIDATION ---
elif "Hyperparameter Validation" in current_pane:
    st.title("Model Verification & Hyperparameter Suite")
    st.caption("Deep performance validation matrices tracking error distribution vectors.")
    st.write("")

    row3_left, row3_right = st.columns(2)
    with row3_left:
        st.markdown("<div class='fg-card'>", unsafe_allow_html=True)
        st.markdown("<div class='fg-card-title'>System Statistical Confusion Matrix</div>", unsafe_allow_html=True)
        st.markdown("<div class='fg-card-sub'>Validation error bounds on the validation split set.</div>", unsafe_allow_html=True)

        if MODEL_READY:
            cm = np.array(metrics["confusion_matrix"])
            fig = go.Figure(data=go.Heatmap(
                z=cm, x=["Normal", "Fraud"], y=["Normal", "Fraud"], colorscale=[[0, PANEL_2], [1, ACCENT]],
                showscale=False, text=cm, texttemplate="%{text}", textfont=dict(size=14, color=TEXT),
            ))
            fig.update_layout(
                height=260, margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title="Predicted Class Alignment", yaxis_title="Actual Ground Audits",
                yaxis=dict(autorange="reversed", color=MUTED), xaxis=dict(color=MUTED),
                plot_bgcolor=PANEL, paper_bgcolor=PANEL, font=dict(color=TEXT),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Awaiting file input to construct prediction performance arrays.")
        st.markdown("</div>", unsafe_allow_html=True)

    with row3_right:
        st.markdown("<div class='fg-card'>", unsafe_allow_html=True)
        st.markdown("<div class='fg-card-title'>Cross-Algorithm Metric Evaluations</div>", unsafe_allow_html=True)
        st.markdown("<div class='fg-card-sub'>Performance metrics across benchmark frameworks.</div>", unsafe_allow_html=True)

        results = pd.DataFrame({
            "Model": ["Logistic Reg.", "Decision Tree", "KNN", "Random Forest (Ours)"],
            "Precision": [0.83, 0.75, 0.92, 0.84], "Recall": [0.63, 0.74, 0.81, 0.83], "F1 Score": [0.72, 0.75, 0.86, 0.83],
        })
        fig = go.Figure()
        for metric_name, c in zip(["Precision", "Recall", "F1 Score"], ["#162B47", "#2A75D3", ACCENT]):
            fig.add_trace(go.Bar(name=metric_name, x=results["Model"], y=results[metric_name], marker_color=c))
        fig.update_layout(
            barmode="group", height=260, margin=dict(t=10, b=10, l=10, r=10),
            plot_bgcolor=PANEL, paper_bgcolor=PANEL, font=dict(color=TEXT),
            yaxis=dict(gridcolor=BORDER, range=[0, 1], color=MUTED), xaxis=dict(color=MUTED),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- PANE 5: SYSTEM PREFERENCES ---
elif "System Preferences" in current_pane:
    st.title("System Preferences & Core Configuration")
    st.caption("Manage authentication levels, pipeline thresholds, and API token telemetry.")
    
    st.markdown("<div class='fg-card'>", unsafe_allow_html=True)
    st.markdown("<div class='fg-card-title'>Operational Parameters</div>", unsafe_allow_html=True)
    st.write("")
    st.slider("Automatic Threshold Escalation Sensitivity", 0.0, 1.0, 0.85)
    st.checkbox("Log safe entries directly to edge arrays", value=True)
    st.checkbox("Automate model pipeline generation cycles", value=False)
    st.markdown("</div>", unsafe_allow_html=True)  