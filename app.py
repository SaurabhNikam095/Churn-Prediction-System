import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Telcom | Advanced Churn Analytics",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS (Upgraded Premium UI)
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    /* ---------------------------
       TOP HEADER BAND (fills blank line)
       --------------------------- */
    .stApp::before{
        content:"";
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 76px;
        background: linear-gradient(90deg, #2563eb 0%, #6366f1 45%, #7c3aed 100%);
        z-index: 0;
        box-shadow: 0 18px 30px rgba(37,99,235,0.20);
    }

    /* Keep content above the band */
    .stApp > header, .stApp > div { position: relative; z-index: 1; }

    /* Add a bit of top padding so content doesn't stick into band */
    .block-container { padding-top: 1.35rem !important; }

    /* ---------------------------
       MAIN BACKGROUND (soft, premium)
       --------------------------- */
    .stApp {
        background:
            radial-gradient(900px 540px at 15% 12%, rgba(99,102,241,0.12), rgba(255,255,255,0) 60%),
            radial-gradient(900px 540px at 85% 10%, rgba(59,130,246,0.10), rgba(255,255,255,0) 60%),
            linear-gradient(180deg, #f8fbff 0%, #f6f7ff 40%, #f8fafc 100%);
        color: #0f172a;
    }

    /* ---------------------------
       SIDEBAR (slightly darker than main, NOT too dark)
       --------------------------- */
    [data-testid="stSidebar"]{
        background: linear-gradient(180deg, #e7ecfb 0%, #d9e3fb 100%) !important;
        border-right: 1px solid rgba(148,163,184,0.35);
        box-shadow: 8px 0 30px rgba(2,6,23,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #0f172a !important;
    }

    /* Sidebar section look */
    .sidebar-card{
        background: rgba(255,255,255,0.70);
        border: 1px solid rgba(148,163,184,0.28);
        border-radius: 16px;
        padding: 14px;
        box-shadow: 0 12px 22px rgba(2,6,23,0.06);
    }

    /* ---------------------------
       TITLES
       --------------------------- */
    .main-title {
        /* responsive size: grows on large screens */
        font-size: clamp(2.6rem, 3.2vw, 4.0rem);
        font-weight: 900;
        line-height: 1.04;
        letter-spacing: -0.6px;
        margin: 0.25rem 0 0.15rem 0;
        background: -webkit-linear-gradient(45deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 10px 30px rgba(2,6,23,0.08);
    }

    .sub-title {
        font-size: 1.08rem;
        color: rgba(71,85,105,0.95);
        font-weight: 600;
        margin-bottom: 1.35rem;
    }

    /* ---------------------------
       CARDS (glass + soft depth)
       --------------------------- */
    .card {
        background: rgba(255,255,255,0.84);
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 20px;
        padding: 18px;
        box-shadow:
            0 20px 50px rgba(2,6,23,0.08),
            0 1px 0 rgba(255,255,255,0.7) inset;
        backdrop-filter: blur(8px);
        margin-bottom: 16px;
    }

    /* ---------------------------
       METRICS
       --------------------------- */
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(148,163,184,0.22);
        box-shadow: 0 16px 34px rgba(2,6,23,0.08);
        padding: 16px;
        border-radius: 18px;
        text-align: center;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 22px 46px rgba(2,6,23,0.10);
    }
    div[data-testid="stMetricValue"] {
        font-weight: 900 !important;
        font-size: 2.05rem !important;
    }

    /* ---------------------------
       BUTTONS (shadow + glow)
       --------------------------- */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #6366f1 45%, #7c3aed 100%);
        color: white !important;
        font-weight: 850;
        border: none;
        border-radius: 16px;
        padding: 0.90rem 1.0rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
        box-shadow:
            0 18px 40px rgba(37,99,235,0.22),
            0 10px 22px rgba(124,58,237,0.16);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        filter: brightness(1.03);
        box-shadow:
            0 24px 54px rgba(37,99,235,0.28),
            0 14px 30px rgba(124,58,237,0.20);
    }
    .stButton>button:active {
        transform: translateY(0px);
        box-shadow:
            0 16px 34px rgba(37,99,235,0.22),
            0 10px 22px rgba(124,58,237,0.16);
    }

    /* Download button gets a premium “success” gradient */
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%) !important;
        box-shadow:
            0 18px 40px rgba(16,185,129,0.18),
            0 10px 22px rgba(6,182,212,0.12) !important;
        border-radius: 16px !important;
    }

    /* ---------------------------
       INPUTS (modern focus glow)
       --------------------------- */
    [data-baseweb="input"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        border-radius: 14px !important;
    }
    [data-baseweb="select"] > div {
        border-radius: 14px !important;
    }

    /* focus ring */
    input:focus, textarea:focus, select:focus {
        outline: none !important;
        box-shadow: 0 0 0 4px rgba(99,102,241,0.20) !important;
        border-color: rgba(99,102,241,0.60) !important;
    }

    /* Plotly card */
    .stPlotlyChart {
        border-radius: 18px;
        overflow: hidden;
        background-color: rgba(255,255,255,0.88);
        border: 1px solid rgba(148,163,184,0.22);
        padding: 10px;
        box-shadow: 0 18px 44px rgba(2,6,23,0.08);
    }

    .muted { color: rgba(71,85,105,0.92); font-size: 0.95rem; }

    /* Risk badge */
    .badge {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        font-weight: 900;
        font-size: 0.85rem;
        border: 1px solid rgba(2,6,23,0.10);
        box-shadow: 0 10px 18px rgba(2,6,23,0.06);
    }
    .badge-high { background: rgba(239,68,68,0.14); color: #991b1b; }
    .badge-med  { background: rgba(245,158,11,0.18); color: #92400e; }
    .badge-low  { background: rgba(16,185,129,0.18); color: #166534; }

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, rgba(148,163,184,0), rgba(148,163,184,0.55), rgba(148,163,184,0));
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# Load models
# =========================================================
@st.cache_resource(show_spinner="Initializing AI Engine...")
def load_models():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    columns = joblib.load("models/model_columns.pkl")
    kmeans = joblib.load("models/kmeans_segmentation.pkl")
    kmeans_scaler = joblib.load("models/kmeans_scaler.pkl")
    bg_data = joblib.load("models/shap_background.pkl")
    return model, scaler, columns, kmeans, kmeans_scaler, bg_data

try:
    model, scaler, columns, kmeans, kmeans_scaler, bg_data = load_models()
except Exception:
    st.error("⚠️ Model artifacts not found. Please ensure `models/` contains required .pkl files.")
    st.stop()

# =========================================================
# Helpers
# =========================================================
def preprocess_input(input_data, raw_columns, fitted_scaler):
    df_processed = input_data.copy()

    binary_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col == 'SeniorCitizen':
            continue
        df_processed[col] = df_processed[col].apply(lambda x: 1 if x in ['Yes', 'Female'] else 0)

    df_processed = pd.get_dummies(df_processed, drop_first=True)

    for col in raw_columns:
        if col not in df_processed.columns and col != 'Churn':
            df_processed[col] = 0

    expected_cols = [c for c in raw_columns if c != 'Churn']
    df_processed = df_processed[expected_cols]

    num_cols = getattr(fitted_scaler, 'feature_names_in_', ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'])
    cols_to_scale = [c for c in num_cols if c in df_processed.columns]
    if cols_to_scale:
        df_processed[cols_to_scale] = fitted_scaler.transform(df_processed[cols_to_scale])

    return df_processed

def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability Risk", 'font': {'size': 18}},
        number={'suffix': "%", 'font': {'size': 44}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "#f1f5f9",
            'borderwidth': 2,
            'steps': [
                {'range': [0, 30], 'color': '#10b981'},
                {'range': [30, 60], 'color': '#f59e0b'},
                {'range': [60, 100], 'color': '#ef4444'}
            ],
            'threshold': {
                'line': {'color': "#0f172a", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def risk_badge_html(prob):
    if prob >= 0.70:
        return '<span class="badge badge-high">High Risk</span>'
    if prob >= 0.40:
        return '<span class="badge badge-med">Medium Risk</span>'
    return '<span class="badge badge-low">Low Risk</span>'

def confidence_from_prob(p):
    return max(p, 1 - p)

def build_input_df_from_session():
    input_data = {
        'gender': st.session_state.get('gender', 'Male'),
        'SeniorCitizen': 1 if st.session_state.get('senior', 'No') == 'Yes' else 0,
        'Partner': st.session_state.get('partner', 'Yes'),
        'Dependents': st.session_state.get('dependents', 'Yes'),
        'tenure': st.session_state.get('tenure', 24),
        'PhoneService': st.session_state.get('phoneservice', 'Yes'),
        'MultipleLines': st.session_state.get('multiplelines', 'No phone service'),
        'InternetService': st.session_state.get('internetservice', 'DSL'),
        'OnlineSecurity': st.session_state.get('onlinesecurity', 'No'),
        'OnlineBackup': st.session_state.get('onlinebackup', 'No'),
        'DeviceProtection': st.session_state.get('deviceprotection', 'No'),
        'TechSupport': st.session_state.get('techsupport', 'No'),
        'StreamingTV': st.session_state.get('streamingtv', 'No'),
        'StreamingMovies': st.session_state.get('streamingmovies', 'No'),
        'Contract': st.session_state.get('contract', 'Month-to-month'),
        'PaperlessBilling': st.session_state.get('paperless', 'Yes'),
        'PaymentMethod': st.session_state.get('payment', 'Electronic check'),
        'MonthlyCharges': st.session_state.get('monthlycharges', 75.0),
        'TotalCharges': st.session_state.get(
            'totalcharges',
            float(st.session_state.get('tenure', 24)) * float(st.session_state.get('monthlycharges', 75.0))
        ),
    }
    return pd.DataFrame(input_data, index=[0])

# =========================================================
# Session defaults
# =========================================================
DEFAULTS = {
    'gender': 'Male',
    'senior': 'No',
    'partner': 'Yes',
    'dependents': 'Yes',
    'phoneservice': 'Yes',
    'multiplelines': 'No phone service',
    'internetservice': 'DSL',
    'onlinesecurity': 'No',
    'onlinebackup': 'No',
    'deviceprotection': 'No',
    'techsupport': 'No',
    'streamingtv': 'No',
    'streamingmovies': 'No',
    'contract': 'Month-to-month',
    'paperless': 'Yes',
    'payment': 'Electronic check',
    'tenure': 24,
    'monthlycharges': 75.0,
    'totalcharges': 24 * 75.0,
}
SAMPLE = {
    'gender': 'Female',
    'senior': 'No',
    'partner': 'No',
    'dependents': 'No',
    'phoneservice': 'Yes',
    'multiplelines': 'Yes',
    'internetservice': 'Fiber optic',
    'onlinesecurity': 'No',
    'onlinebackup': 'Yes',
    'deviceprotection': 'No',
    'techsupport': 'No',
    'streamingtv': 'Yes',
    'streamingmovies': 'Yes',
    'contract': 'Month-to-month',
    'paperless': 'Yes',
    'payment': 'Electronic check',
    'tenure': 6,
    'monthlycharges': 99.0,
    'totalcharges': 6 * 99.0,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# =========================================================
# Sidebar (Telecom Data) - aligned buttons
# =========================================================
st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
st.sidebar.markdown("## 📡 Telecom Data")
st.sidebar.markdown('<p class="muted">Quick actions</p>', unsafe_allow_html=True)

csa, csb = st.sidebar.columns(2)
with csa:
    if st.button("🎯 Load Sample", use_container_width=True):
        for k, v in SAMPLE.items():
            st.session_state[k] = v
        st.session_state["last_result"] = None
        st.rerun()

with csb:
    if st.button("🔄 Reset", use_container_width=True):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.session_state["last_result"] = None
        st.rerun()

st.sidebar.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# Main header
# =========================================================

# =========================================================
# 1) INPUT DATA SECTION (FIRST)
# =========================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🧾 Input Data")
st.markdown('<p class="muted">Fill customer details below, then click <b>Input Data</b> to run the prediction model.</p>', unsafe_allow_html=True)

with st.form("main_input_form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Demographics")
        st.selectbox("Gender", ['Male', 'Female'], key="gender")
        st.selectbox("Senior Citizen", ['No', 'Yes'], key="senior")
        st.selectbox("Partner", ['Yes', 'No'], key="partner")
        st.selectbox("Dependents", ['Yes', 'No'], key="dependents")

    with col2:
        st.markdown("#### 🔌 Services")
        st.selectbox("Phone Service", ['Yes', 'No'], key="phoneservice")
        st.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'], key="multiplelines")
        st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'], key="internetservice")
        st.selectbox("Online Security", ['No', 'Yes', 'No internet service'], key="onlinesecurity")
        st.selectbox("Online Backup", ['No', 'Yes', 'No internet service'], key="onlinebackup")
        st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'], key="deviceprotection")

    with col3:
        st.markdown("#### 💳 Billing")
        st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'], key="techsupport")
        st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'], key="streamingtv")
        st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'], key="streamingmovies")

        st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'], key="contract")
        st.selectbox("Paperless Billing", ['Yes', 'No'], key="paperless")
        st.selectbox(
            "Payment Method",
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
            key="payment"
        )

        st.slider("Tenure (Months)", 0, 72, key="tenure")
        st.slider("Monthly Charges ($)", 18.0, 120.0, key="monthlycharges")

        # totalcharges stays editable but default is tenure*monthly
        default_total = float(st.session_state["tenure"]) * float(st.session_state["monthlycharges"])
        if "totalcharges" not in st.session_state or st.session_state["totalcharges"] == DEFAULTS["totalcharges"]:
            st.session_state["totalcharges"] = default_total
        st.slider("Total Charges ($)", 0.0, 8000.0, key="totalcharges")

    st.markdown("")
    run_now = st.form_submit_button("📥 Input Data")

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 2) MODEL EXECUTION + RESULTS (same page)
# =========================================================
if run_now:
    input_df = build_input_df_from_session()

    with st.spinner("Running prediction model..."):
        X_pred = preprocess_input(input_df, columns, scaler)
        prediction_proba = model.predict_proba(X_pred)[0][1]
        prediction = 1 if prediction_proba > 0.5 else 0

        segment_features = input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
        scaled_seg_fw = kmeans_scaler.transform(segment_features)
        segment = kmeans.predict(scaled_seg_fw)[0]
        segment_names = {0: "Low/Mid Tier", 1: "High Value (VIP)", 2: "At Risk (Newer)"}
        segment_label = segment_names.get(segment, "Standard")

        conf = confidence_from_prob(prediction_proba)

        result_row = input_df.copy()
        result_row["churn_probability"] = prediction_proba
        result_row["prediction"] = "Churn" if prediction == 1 else "No Churn"
        result_row["segment"] = segment_label
        result_row["risk"] = "High" if prediction_proba >= 0.70 else ("Medium" if prediction_proba >= 0.40 else "Low")
        st.session_state["last_result"] = result_row

# =========================================================
# 3) SHOW RESULTS IF AVAILABLE
# =========================================================
if st.session_state["last_result"] is not None:
    last = st.session_state["last_result"]
    prob = float(last["churn_probability"].iloc[0])
    pred_label = str(last["prediction"].iloc[0])
    seg_label = str(last["segment"].iloc[0])
    conf = confidence_from_prob(prob)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Predicted Status", "🚨 Churner" if pred_label == "Churn" else "✅ Retained")
    with k2:
        st.metric("Confidence", f"{conf*100:.1f}%")
    with k3:
        st.metric("Segment", seg_label)
    with k4:
        st.metric("Estimated LTV", f"${float(last['TotalCharges'].iloc[0]):,.2f}")

    st.markdown("")
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📌 Risk Summary")
        st.markdown(
            f"**Churn Probability:** `{prob:.2%}` &nbsp;&nbsp; {risk_badge_html(prob)}",
            unsafe_allow_html=True
        )
        st.markdown('<p class="muted">Use this to prioritize retention actions.</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧾 Customer Snapshot")
        cA, cB = st.columns(2)
        with cA:
            st.write(f"**Tenure:** {int(last['tenure'].iloc[0])} months")
            st.write(f"**Contract:** {last['Contract'].iloc[0]}")
            st.write(f"**Payment:** {last['PaymentMethod'].iloc[0]}")
        with cB:
            st.write(f"**Internet:** {last['InternetService'].iloc[0]}")
            st.write(f"**Monthly Charges:** ${float(last['MonthlyCharges'].iloc[0]):.2f}")
            st.write(f"**Total Charges:** ${float(last['TotalCharges'].iloc[0]):.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📉 Probability Gauge")
        st.plotly_chart(create_gauge_chart(prob), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Explainability
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Explainable AI Diagnostic")
    st.markdown('<p class="muted">Feature impact for this specific customer.</p>', unsafe_allow_html=True)

    with st.expander("Show SHAP Waterfall (customer-level)", expanded=True):
        try:
            input_df = build_input_df_from_session()
            X_pred = preprocess_input(input_df, columns, scaler)

            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(9, 4))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')

            model_class = type(model).__name__
            if model_class in ['RandomForestClassifier', 'XGBClassifier', 'XGBRegressor']:
                explainer = shap.TreeExplainer(model)
                shap_val = explainer(X_pred)
                shap_obj = shap_val[1][0] if isinstance(shap_val, list) else shap_val[0]
            else:
                explainer = shap.LinearExplainer(model, bg_data)
                shap_val = explainer(X_pred)
                shap_obj = shap_val[0]

            shap.plots.waterfall(shap_obj, show=False, max_display=8)
            st.pyplot(fig, bbox_inches='tight')
            plt.close()

        except Exception as e:
            st.error(f"Failed to render SHAP diagnostic: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Export + Data
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📁 Export & Data")
    st.markdown('<p class="muted">Review input + download the prediction row.</p>', unsafe_allow_html=True)

    current_input = build_input_df_from_session()
    st.markdown("#### Current Input")
    st.dataframe(current_input, use_container_width=True)

    st.markdown("#### Latest Prediction Output")
    st.dataframe(last, use_container_width=True)

    csv_bytes = last.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Prediction (CSV)",
        data=csv_bytes,
        file_name="nexus_churn_prediction.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✅ Ready")
    st.markdown("- Fill the form in **Input Data** above\n- Click **📥 Input Data** to run the model\n- Results will appear below")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with Streamlit • Telecom Churn Analytics • © 2026")