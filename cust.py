"""
======================================================================
 CUSTOMER LIFETIME VALUE (CLV) PREDICTION SYSTEM  -  ROYAL EDITION
 A dynamic, animated Streamlit dashboard
----------------------------------------------------------------------
 HOW TO RUN:
   1. pip install streamlit pandas numpy scikit-learn seaborn matplotlib openpyxl
   2. streamlit run clv_app.py
----------------------------------------------------------------------
 WHAT IT DOES:
   - Upload ANY retail transaction dataset (CSV / Excel)
   - Map your columns -> auto-computes RFM, CLV, Segments (your notebook logic)
   - Trains RandomForest (regression + classification) + a Deep Learning net
   - Cohort retention heatmap + 6 power features
   - Royal purple/gold animated, professional UI
======================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi
from datetime import timedelta

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="CLV Prediction System", page_icon="👑", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# CSS  (Royal Purple + Gold theme + animations/transitions)
# ============================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0d0015 0%, #1a0030 50%, #0d0015 100%); }

    @keyframes fadeUp { from {opacity:0; transform:translateY(18px);} to {opacity:1; transform:translateY(0);} }
    @keyframes glow { 0%,100%{box-shadow:0 0 18px rgba(255,215,0,0.15);} 50%{box-shadow:0 0 34px rgba(255,215,0,0.45);} }
    @keyframes shimmer { 0%{background-position:-500px 0;} 100%{background-position:500px 0;} }
    @keyframes floatY { 0%,100%{transform:translateY(0);} 50%{transform:translateY(-8px);} }

    .nav-container {
        background: linear-gradient(90deg, #1a0030, #2d0050, #1a0030);
        border-bottom: 2px solid #FFD700; padding: 15px 30px; display: flex;
        justify-content: space-between; align-items: center; margin-bottom: 30px;
        border-radius: 0 0 20px 20px; animation: fadeUp .6s ease;
    }
    .nav-logo { font-family: 'Playfair Display', serif; font-size: 1.5rem; color: #FFD700; font-weight: 700; }

    .metric-card {
        background: linear-gradient(135deg, #1a0030, #2d0050); border: 1px solid #FFD700;
        border-radius: 20px; padding: 25px; text-align: center;
        box-shadow: 0 0 20px rgba(255,215,0,0.15); margin: 10px 0; animation: fadeUp .7s ease;
        transition: transform .35s ease, box-shadow .35s ease;
    }
    .metric-card:hover { transform: translateY(-8px) scale(1.02); box-shadow: 0 0 40px rgba(255,215,0,0.45); }
    .metric-value { font-size: 2.2rem; font-weight: bold; color: #FFD700; }
    .metric-label { font-size: 0.85rem; color: #cc99ff; margin-top: 5px; letter-spacing: 1px; text-transform: uppercase; }
    .metric-icon { font-size: 2rem; margin-bottom: 10px; animation: floatY 4s ease-in-out infinite; }

    .royal-title {
        font-family: 'Playfair Display', serif; font-size: 3rem; font-weight: 900;
        background: linear-gradient(90deg, #FFD700, #cc99ff, #FFD700); background-size: 200% auto;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 10px; animation: shimmer 3s linear infinite, fadeUp .6s ease;
    }
    .royal-subtitle { text-align: center; color: #cc99ff; font-size: 1rem; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 30px; animation: fadeUp .8s ease; }
    .section-header { font-size: 1.5rem; color: #FFD700; border-left: 4px solid #FFD700; padding-left: 15px; margin: 30px 0 20px 0; font-weight: 600; animation: fadeUp .6s ease; }
    .gold-divider { height: 1px; background: linear-gradient(90deg, transparent, #FFD700, transparent); margin: 30px 0; }

    .predict-card {
        background: linear-gradient(135deg, #1a0030, #2d0050); border: 2px solid #FFD700;
        border-radius: 20px; padding: 30px; text-align: center;
        box-shadow: 0 0 40px rgba(255,215,0,0.2); animation: fadeUp .6s ease, glow 3s ease-in-out infinite;
    }
    .predict-value { font-size: 3rem; font-weight: bold; color: #FFD700; }
    .predict-label { color: #cc99ff; font-size: 1rem; letter-spacing: 2px; text-transform: uppercase; }

    .rec-card {
        background: linear-gradient(135deg, #1a0030, #2d0050); border-radius: 15px; padding: 20px;
        margin: 10px 0; border-left: 4px solid #FFD700; color: white;
        animation: fadeUp .6s ease; transition: transform .3s ease;
    }
    .rec-card:hover { transform: translateX(6px); }

    .dna-card {
        background: linear-gradient(135deg, #2d0050 0%, #1a0030 60%, #3d0066 100%);
        border: 1px solid #FFD700; border-radius: 22px; padding: 28px; color: #fff;
        box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 30px rgba(255,215,0,0.2);
        position: relative; overflow: hidden; min-height: 240px; animation: fadeUp .6s ease;
    }
    .dna-card::after {
        content:""; position:absolute; top:-50%; left:-50%; width:200%; height:200%;
        background: linear-gradient(120deg, transparent 40%, rgba(255,215,0,0.12) 50%, transparent 60%);
        animation: shimmer 4s linear infinite;
    }
    .dna-chip { width:46px; height:34px; border-radius:7px; background:linear-gradient(135deg,#FFD700,#b8860b); margin-bottom:18px; }
    .dna-id { font-family:'Playfair Display',serif; font-size:1.6rem; color:#FFD700; letter-spacing:1px; }
    .dna-row { display:flex; justify-content:space-between; margin-top:18px; }
    .dna-k { color:#cc99ff; font-size:.72rem; text-transform:uppercase; letter-spacing:1px; }
    .dna-v { color:#fff; font-size:1.05rem; font-weight:700; }

    .pill { display:inline-block; padding:5px 16px; border-radius:50px; font-weight:700; font-size:.8rem; letter-spacing:1px; }

    .stButton > button {
        background: linear-gradient(90deg, #FFD700, #cc99ff); color: #0d0015; font-weight: bold;
        border: none; border-radius: 50px; padding: 14px 40px; font-size: 1.05rem; width: 100%;
        transition: transform .25s ease, box-shadow .25s ease;
    }
    .stButton > button:hover { transform: translateY(-3px); box-shadow: 0 8px 26px rgba(255,215,0,0.45); }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] { background: #1a0030; border-radius: 12px 12px 0 0; color: #cc99ff; transition: all .3s ease; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg,#2d0050,#1a0030); color: #FFD700 !important; border-bottom: 2px solid #FFD700; }

    [data-testid="stSidebar"] { background: linear-gradient(180deg,#0d0015,#1a0030); border-right: 1px solid #FFD700; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

CUR = "£"


# ============================================================
# DATA ENGINE  (fully dynamic - any dataset works)
# ============================================================
def build_sample_data(n=900, seed=42):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp("2011-12-09 12:50:00")
    rows = []
    for cid in range(12346, 12346 + n):
        for _ in range(int(rng.integers(1, 15))):
            day = int(rng.integers(0, 374))
            rows.append({
                "Invoice": f"INV{int(rng.integers(100000,999999))}",
                "Customer ID": float(cid),
                "InvoiceDate": end - timedelta(days=day, hours=int(rng.integers(0, 23))),
                "Quantity": int(rng.integers(1, 25)),
                "Price": round(float(rng.uniform(1.5, 60)), 2),
            })
    return pd.DataFrame(rows)


def segment_customer(score):
    if score == "333": return "Champion"
    elif score in ["323", "332", "331"]: return "Loyal Customer"
    elif score in ["311", "312", "313"]: return "Potential Loyalist"
    elif score in ["133", "233"]: return "At Risk"
    elif score in ["111", "211", "112"]: return "Lost"
    return "Regular"


# FIX: leading underscore on the DataFrame argument (_df) tells Streamlit's
# @st.cache_data NOT to hash it. Streamlit was raising
# "Cannot hash argument 'df' (of type pandas.DataFrame)" because by default
# it tries to hash every argument to decide whether to reuse a cached result.
# DataFrames (especially big/mixed-dtype ones) can't always be hashed
# reliably, so the underscore prefix is the documented escape hatch -
# Streamlit will still cache based on the OTHER arguments (cust_col, date_col,
# etc.), it just trusts that _df itself didn't silently change between calls.
@st.cache_data(show_spinner=False)
def compute_rfm(_df, cust_col, date_col, inv_col, qty_col, price_col):
    df = _df.copy()
    df = df.dropna(subset=[cust_col]).drop_duplicates()
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[qty_col, price_col])
    df = df[(df[qty_col] > 0) & (df[price_col] > 0)]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["TotalPrice"] = df[qty_col] * df[price_col]

    reference_date = df[date_col].max()

    rfm = df.groupby(cust_col).agg(
        Recency=(date_col, lambda x: (reference_date - x.max()).days),
        Frequency=(inv_col, "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index().rename(columns={cust_col: "Customer ID"})

    rfm["CLV"] = rfm["Monetary"] * rfm["Frequency"]
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 3, labels=[3, 2, 1], duplicates="drop").astype(str)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 3, labels=[1, 2, 3], duplicates="drop").astype(str)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 3, labels=[1, 2, 3], duplicates="drop").astype(str)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    rfm["Segment"] = rfm["RFM_Score"].apply(segment_customer)
    rfm["CLV_Segment"] = pd.qcut(rfm["CLV"], 3, labels=["Low", "Medium", "High"], duplicates="drop").astype(str)

    # Cohort retention
    df["InvoiceMonth"] = df[date_col].dt.to_period("M")
    df["CohortMonth"] = df.groupby(cust_col)[date_col].transform("min").dt.to_period("M")
    df["CohortIndex"] = (df["InvoiceMonth"] - df["CohortMonth"]).apply(lambda x: x.n)
    cd = df.groupby(["CohortMonth", "CohortIndex"])[cust_col].nunique().reset_index()
    ct = cd.pivot_table(index="CohortMonth", columns="CohortIndex", values=cust_col)
    cohort = ct.divide(ct.iloc[:, 0], axis=0) * 100 if not ct.empty else None
    if cohort is not None:
        cohort.index = cohort.index.astype(str)

    return rfm, cohort, reference_date


# FIX: same hashing issue applies here - "rfm" is a DataFrame, so it becomes
# "_rfm" to skip Streamlit's hashing attempt on it.
@st.cache_resource(show_spinner=False)
def train_models(_rfm, use_deep=True):
    rfm = _rfm
    X = rfm[["Recency", "Frequency", "Monetary"]].values
    y_clv = rfm["CLV"].values
    y_seg = rfm["CLV_Segment"].astype(str).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xs, y_clv, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(Xtr, ytr)
    rf_r2 = r2_score(yte, rf.predict(Xte))
    rf_mae = mean_absolute_error(yte, rf.predict(Xte))

    le = LabelEncoder()
    y_enc = le.fit_transform(y_seg)
    rfc_acc = float("nan")
    rfc = RandomForestClassifier(n_estimators=100, max_depth=5,
                                 min_samples_split=10, min_samples_leaf=5, random_state=42)
    if len(np.unique(y_enc)) > 1:
        Xtr2, Xte2, ytr2, yte2 = train_test_split(Xs, y_enc, test_size=0.2, random_state=42)
        rfc.fit(Xtr2, ytr2)
        rfc_acc = accuracy_score(yte2, rfc.predict(Xte2))
    else:
        rfc.fit(Xs, y_enc)

    dl, dl_r2 = None, None
    if use_deep:
        dl = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                          solver="adam", max_iter=400, random_state=42, early_stopping=True)
        dl.fit(Xtr, ytr)
        dl_r2 = r2_score(yte, dl.predict(Xte))

    metrics = {"rf_r2": rf_r2, "rf_mae": rf_mae, "rfc_acc": rfc_acc, "dl_r2": dl_r2}
    return rf, rfc, dl, scaler, le, metrics


# ============================================================
# MAIN-PAGE DATA LOADER  - dynamic dataset upload + column mapping
# (placed on the main page, not just the sidebar, so it can never be
#  missed even if the sidebar is collapsed)
# ============================================================
st.markdown("""
    <div style="background:linear-gradient(90deg,#1a0030,#2d0050,#1a0030);
                border:2px solid #FFD700; border-radius:18px; padding:22px 28px;
                margin-bottom:20px; box-shadow:0 0 25px rgba(255,215,0,0.15);">
        <span style="font-family:'Playfair Display',serif; color:#FFD700; font-size:1.3rem; font-weight:700;">
            📂 Load Your Dataset
        </span>
        <div style="color:#cc99ff; font-size:0.88rem; margin-top:4px;">
            Upload any company's retail/transaction CSV or Excel file below — RFM, CLV and every model
            recalculates instantly for YOUR data. No upload? The built-in sample dataset is used automatically.
        </div>
    </div>
""", unsafe_allow_html=True)

loader_col1, loader_col2 = st.columns([2, 1])
with loader_col1:
    uploaded = st.file_uploader(
        "Upload CSV / Excel — drag & drop or browse",
        type=["csv", "xlsx", "xls"],
        key="main_uploader",
    )
with loader_col2:
    use_deep = st.checkbox("🧠 Train Deep Learning model", value=True, key="main_use_deep")

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(("xlsx", "xls")):
            raw = pd.read_excel(uploaded)
        else:
            raw = pd.read_csv(uploaded, encoding="ISO-8859-1", on_bad_lines="skip")
        st.success(f"✅ Loaded **{len(raw):,}** rows from **{uploaded.name}**")
        data_source = "uploaded"
    except Exception as e:
        st.error(f"Could not read file: {e}")
        raw = build_sample_data(); data_source = "sample"
else:
    raw = build_sample_data(); data_source = "sample"
    st.info("ℹ️ Using built-in sample data. Upload a file above to use your own dataset.")


def guess(cols, *keywords):
    for kw in keywords:
        for c in cols:
            if kw.lower() in str(c).lower():
                return c
    return cols[0]

cols = list(raw.columns)

with st.expander("🔧 Column Mapping — confirm or adjust which column is which", expanded=(data_source == "uploaded")):
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        cust_col = st.selectbox("Customer ID", cols, index=cols.index(guess(cols, "customer", "cust", "client")), key="map_cust")
    with m2:
        date_col = st.selectbox("Invoice Date", cols, index=cols.index(guess(cols, "date", "time")), key="map_date")
    with m3:
        inv_col = st.selectbox("Invoice / Order", cols, index=cols.index(guess(cols, "invoice", "order", "transaction")), key="map_inv")
    with m4:
        qty_col = st.selectbox("Quantity", cols, index=cols.index(guess(cols, "quantity", "qty", "units")), key="map_qty")
    with m5:
        price_col = st.selectbox("Unit Price", cols, index=cols.index(guess(cols, "price", "unitprice", "amount", "cost")), key="map_price")

with st.spinner("👑 Crunching RFM, CLV & training models..."):
    try:
        rfm, cohort, reference_date = compute_rfm(raw, cust_col, date_col, inv_col, qty_col, price_col)
        if len(rfm) < 10:
            raise ValueError("Too few valid customers detected after cleaning.")
        rf, rfc, dl, scaler, le, metrics = train_models(rfm, use_deep=use_deep)
        data_ok = True
    except Exception as e:
        data_ok = False
        err_msg = str(e)

if data_ok:
    st.markdown(
        f"<div style='color:#7CFC00;font-weight:700;font-size:1.05rem;margin:6px 0 20px 0;'>"
        f"✅ R, F, M & CLV calculated for {len(rfm):,} customers — see the full table in the Home tab below.</div>",
        unsafe_allow_html=True,
    )

# Short status note in the sidebar too, in case it's open
st.sidebar.markdown("<h2 style='color:#FFD700;font-family:Playfair Display;'>👑 Data Control</h2>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<p style='color:#cc99ff;font-size:.85rem;'>The dataset uploader is on the main page "
    "(top of the screen). This sidebar just shows status.</p>",
    unsafe_allow_html=True,
)
if data_ok:
    st.sidebar.success(f"{len(rfm):,} customers calculated")
    st.sidebar.caption(f"Source: {'Your uploaded file' if data_source == 'uploaded' else 'Sample data'}")


# ============================================================
# NAV BAR
# ============================================================
st.markdown(f"""
    <div class="nav-container">
        <div class="nav-logo">👑 CLV Prediction System</div>
        <div style="color:#cc99ff;font-size:0.85rem;">{'Your Dataset' if data_source=='uploaded' else 'Sample Data'} | RFM • ML • Deep Learning</div>
    </div>
""", unsafe_allow_html=True)

if not data_ok:
    st.error(f"Could not process this dataset with the selected columns. Please re-check the mapping in the sidebar.\n\nDetails: {err_msg}")
    st.stop()


# ============================================================
# TABS
# ============================================================
tabs = st.tabs([
    "🏠 Home", "📊 Segmentation", "🔮 CLV Prediction",
    "🕰️ Time Machine", "🎯 Retention Score", "💌 AI Email",
    "📈 Revenue Simulator", "🏆 Customer DNA",
])

# ---------------- HOME ----------------
with tabs[0]:
    st.markdown('<div class="royal-title">Customer Lifetime Value</div>', unsafe_allow_html=True)
    st.markdown('<div class="royal-subtitle">✦ Powered by Machine Learning & Deep Learning ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="rec-card" style="border-left:4px solid #7CFC00;">
    ✅ <b style="color:#7CFC00;">Calculation complete</b> — Recency, Frequency, Monetary &amp; CLV have been
    computed for all <b style="color:#FFD700;">{len(rfm):,}</b> customers in this dataset. Full results below.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📄 Full RFM + CLV Results — Every Customer</div>', unsafe_allow_html=True)

    search_col, sort_col = st.columns([2, 1])
    with search_col:
        search_id = st.text_input("🔍 Search by Customer ID (leave blank to see all)", "")
    with sort_col:
        sort_by = st.selectbox("Sort by", ["CLV", "Recency", "Frequency", "Monetary"], index=0)

    display_rfm = rfm.copy()
    display_rfm["Customer ID"] = display_rfm["Customer ID"].astype(str).str.replace(".0", "", regex=False)
    if search_id.strip():
        display_rfm = display_rfm[display_rfm["Customer ID"].str.contains(search_id.strip(), case=False)]
    display_rfm = display_rfm.sort_values(sort_by, ascending=False)

    show_cols = ["Customer ID", "Recency", "Frequency", "Monetary", "CLV", "Segment", "CLV_Segment"]
    st.dataframe(
        display_rfm[show_cols].rename(columns={
            "Recency": "Recency (days)", "Frequency": "Frequency (orders)",
            "Monetary": f"Monetary ({CUR})", "CLV": f"CLV ({CUR})", "CLV_Segment": "Value Tier",
        }),
        use_container_width=True, height=420,
    )
    st.caption(f"Showing {len(display_rfm):,} of {len(rfm):,} customers.")

    csv_data = rfm.copy()
    csv_data["Customer ID"] = csv_data["Customer ID"].astype(str).str.replace(".0", "", regex=False)
    st.download_button(
        "⬇️ Download full RFM + CLV table (CSV)",
        csv_data.to_csv(index=False),
        "rfm_clv_results.csv", "text/csv",
    )

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-icon">👥</div><div class="metric-value">{len(rfm):,}</div><div class="metric-label">Total Customers</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-icon">💷</div><div class="metric-value">{CUR}{rfm["Monetary"].sum()/1_000_000:.1f}M</div><div class="metric-label">Total Revenue</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-icon">🔁</div><div class="metric-value">{rfm["Frequency"].mean():.1f}x</div><div class="metric-label">Avg Frequency</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-icon">💎</div><div class="metric-value">{CUR}{rfm["CLV"].mean():,.0f}</div><div class="metric-label">Avg CLV</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="section-header">📌 About This Project</div>', unsafe_allow_html=True)
        st.markdown("""<div class="rec-card">
        This system predicts <b style="color:#FFD700">Customer Lifetime Value</b> using RFM Analysis, Machine Learning & Deep Learning.<br><br>
        ✦ RFM Calculation — Recency, Frequency, Monetary per customer<br>
        ✦ Segmentation — behavioural grouping<br>
        ✦ Regression — exact CLV prediction<br>
        ✦ Classification — High / Medium / Low value<br>
        ✦ Cohort Analysis — retention heatmap
        </div>""", unsafe_allow_html=True)
    with b:
        st.markdown('<div class="section-header">🤖 Live Model Performance</div>', unsafe_allow_html=True)
        dl_line = f'<br><br><b style="color:#FFD700">Deep Learning (Neural Net)</b><br>R² Score: {metrics["dl_r2"]:.3f}' if metrics["dl_r2"] is not None else ""
        acc_txt = f'{metrics["rfc_acc"]*100:.1f}%' if metrics["rfc_acc"] == metrics["rfc_acc"] else "N/A"
        st.markdown(f"""<div class="rec-card">
        <b style="color:#FFD700">Random Forest Regressor</b><br>
        R² Score: {metrics['rf_r2']:.3f} &nbsp;|&nbsp; MAE: {CUR}{metrics['rf_mae']:,.0f}<br><br>
        <b style="color:#FFD700">Random Forest Classifier</b><br>
        Accuracy: {acc_txt}{dl_line}
        </div>""", unsafe_allow_html=True)

# ---------------- SEGMENTATION ----------------
with tabs[1]:
    st.markdown('<div class="royal-title">Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="royal-subtitle">✦ RFM Analysis & Cohort Retention ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">🎯 RFM Segments</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#FFD700', '#cc99ff', '#9b59b6', '#f39c12', '#8e44ad', '#d4ac0d']
        rfm['Segment'].value_counts().plot(kind='bar', ax=ax, color=colors, edgecolor='none')
        fig.patch.set_facecolor('#1a0030'); ax.set_facecolor('#1a0030')
        ax.tick_params(colors='#cc99ff'); ax.set_title('Customer Segments', color='#FFD700', fontsize=14)
        ax.set_xlabel(''); ax.set_ylabel('Count', color='#cc99ff')
        for s in ax.spines.values(): s.set_edgecolor('#FFD700')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    with c2:
        st.markdown('<div class="section-header">💎 CLV Distribution</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        rfm['CLV_Segment'].value_counts().plot(kind='pie', autopct='%1.1f%%',
            colors=['#FFD700', '#cc99ff', '#9b59b6'], ax=ax2,
            textprops={'color': 'white', 'fontsize': 12},
            wedgeprops={'edgecolor': '#0d0015', 'linewidth': 2})
        ax2.set_ylabel(''); fig2.patch.set_facecolor('#1a0030')
        ax2.set_title('CLV Segments', color='#FFD700', fontsize=14)
        st.pyplot(fig2)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔥 Cohort Retention Heatmap</div>', unsafe_allow_html=True)
    if cohort is not None and not cohort.empty:
        fig3, ax3 = plt.subplots(figsize=(20, 8))
        sns.heatmap(cohort.astype(float), annot=True, fmt='.0f', cmap='YlOrRd', ax=ax3,
                    vmin=0, vmax=100, linewidths=0.5, linecolor='#0d0015')
        ax3.set_title('Customer Retention Rate by Cohort (%)', color='#FFD700', fontsize=16)
        ax3.set_xlabel('Months After First Purchase', color='#cc99ff'); ax3.set_ylabel('Cohort Month', color='#cc99ff')
        fig3.patch.set_facecolor('#1a0030'); ax3.tick_params(colors='#cc99ff')
        st.pyplot(fig3)
    else:
        st.info("Not enough monthly history in this dataset to build a cohort heatmap.")

# ---------------- CLV PREDICTION ----------------
with tabs[2]:
    st.markdown('<div class="royal-title">CLV Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="royal-subtitle">✦ Enter Customer Details to Predict ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-header">📅 Recency</div>', unsafe_allow_html=True)
        recency = st.number_input("Days since last purchase", 1, 1000, 30)
    with c2:
        st.markdown('<div class="section-header">🔁 Frequency</div>', unsafe_allow_html=True)
        frequency = st.number_input("Number of purchases", 1, 500, 5)
    with c3:
        st.markdown('<div class="section-header">💷 Monetary</div>', unsafe_allow_html=True)
        monetary = st.number_input(f"Total amount spent ({CUR})", 1.0, 1_000_000.0, 500.0)

    model_choice = st.radio("Prediction model", ["Random Forest", "Deep Learning"] if dl is not None else ["Random Forest"], horizontal=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    if st.button("👑 PREDICT CLV NOW"):
        Xin = scaler.transform([[recency, frequency, monetary]])
        clv_pred = dl.predict(Xin)[0] if (model_choice == "Deep Learning" and dl is not None) else rf.predict(Xin)[0]
        seg_pred = le.inverse_transform(rfc.predict(Xin))[0]

        d1, d2 = st.columns(2)
        d1.markdown(f'<div class="predict-card"><div class="predict-label">Predicted Lifetime Value</div><div class="predict-value">{CUR}{clv_pred:,.0f}</div></div>', unsafe_allow_html=True)
        color = "#FFD700" if seg_pred == "High" else "#cc99ff" if seg_pred == "Medium" else "#ff6b6b"
        emoji = "👑" if seg_pred == "High" else "⭐" if seg_pred == "Medium" else "⚠️"
        d2.markdown(f'<div class="predict-card"><div class="predict-label">Customer Segment</div><div class="predict-value" style="color:{color};">{emoji} {seg_pred}</div></div>', unsafe_allow_html=True)

# ---------------- FEATURE 1: TIME MACHINE ----------------
with tabs[3]:
    st.markdown('<div class="royal-title">🕰️ Customer Time Machine</div>', unsafe_allow_html=True)
    st.markdown('<div class="royal-subtitle">✦ See how this customer evolves into the future ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    tm_r = c1.number_input("Current Recency (days)", 1, 1000, 30, key="tm_r")
    tm_f = c2.number_input("Current Frequency", 1, 500, 5, key="tm_f")
    tm_m = c3.number_input(f"Current Monetary ({CUR})", 1.0, 1_000_000.0, 500.0, key="tm_m")

    months = st.slider("⏩ Travel into the future (months)", 0, 24, 6)
    purchase_rate = st.slider("Expected purchases per month", 0.0, 10.0, 1.0, 0.5)
    spend_per_purchase = tm_m / max(tm_f, 1)

    def project(mm):
        ff = tm_f + purchase_rate * mm
        mmn = tm_m + purchase_rate * mm * spend_per_purchase
        rr = max(1, int(tm_r - mm * 30)) if purchase_rate > 0 else tm_r + mm * 30
        return rf.predict(scaler.transform([[rr, ff, mmn]]))[0]

    now_clv = rf.predict(scaler.transform([[tm_r, tm_f, tm_m]]))[0]
    future_clv = project(months)
    delta = future_clv - now_clv

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    d1.markdown(f'<div class="metric-card"><div class="metric-icon">📍</div><div class="metric-value">{CUR}{now_clv:,.0f}</div><div class="metric-label">CLV Today</div></div>', unsafe_allow_html=True)
    d2.markdown(f'<div class="metric-card"><div class="metric-icon">🔮</div><div class="metric-value">{CUR}{future_clv:,.0f}</div><div class="metric-label">CLV in {months} months</div></div>', unsafe_allow_html=True)
    arrow = "▲" if delta >= 0 else "▼"
    dcolor = "#7CFC00" if delta >= 0 else "#ff6b6b"
    d3.markdown(f'<div class="metric-card"><div class="metric-icon">📈</div><div class="metric-value" style="color:{dcolor};">{arrow} {CUR}{abs(delta):,.0f}</div><div class="metric-label">Projected Change</div></div>', unsafe_allow_html=True)

    xs = list(range(0, months + 1))
    ys = [project(mm) for mm in xs]
    figt, axt = plt.subplots(figsize=(12, 4))
    axt.plot(xs, ys, color='#FFD700', marker='o', linewidth=2.5)
    axt.fill_between(xs, ys, color='#FFD700', alpha=0.12)
    figt.patch.set_facecolor('#1a0030'); axt.set_facecolor('#1a0030')
    axt.tick_params(colors='#cc99ff'); axt.set_title('Projected CLV Trajectory', color='#FFD700')
    axt.set_xlabel('Months from now', color='#cc99ff'); axt.set_ylabel(f'CLV ({CUR})', color='#cc99ff')
    for s in axt.spines.values(): s.set_edgecolor('#FFD700')
    st.pyplot(figt)

# ---------------- FEATURE 2: RETENTION / RISK SCORE ----------------
with tabs[4]:
    st.markdown('<div class="royal-title">🎯 Smart Retention Score</div>', unsafe_allow_html=True)
    st.markdown('<div class="royal-subtitle">✦ Real-time Risk of Leaving (0-100%) ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    rs_r = c1.number_input("Recency (days)", 1, 1000, 90, key="rs_r")
    rs_f = c2.number_input("Frequency", 1, 500, 3, key="rs_f")
    rs_m = c3.number_input(f"Monetary ({CUR})", 1.0, 1_000_000.0, 300.0, key="rs_m")

    r_max = max(rfm["Recency"].quantile(0.95), 1)
    f_max = max(rfm["Frequency"].quantile(0.95), 1)
    m_max = max(rfm["Monetary"].quantile(0.95), 1)
    recency_risk = min(rs_r / r_max, 1) * 100
    freq_risk = (1 - min(rs_f / f_max, 1)) * 100
    money_risk = (1 - min(rs_m / m_max, 1)) * 100
    risk = round(0.5 * recency_risk + 0.3 * freq_risk + 0.2 * money_risk, 1)
    retention = round(100 - risk, 1)

    if risk >= 66: r_color, r_label = "#ff6b6b", "HIGH RISK 🚨"
    elif risk >= 33: r_color, r_label = "#f39c12", "MEDIUM RISK ⚠️"
    else: r_color, r_label = "#7CFC00", "SAFE ✅"

    st.markdown(f"""
    <div class="predict-card">
        <div class="predict-label">Risk of Leaving</div>
        <div class="predict-value" style="color:{r_color};">{risk}%</div>
        <div style="margin-top:10px;"><span class="pill" style="background:{r_color};color:#0d0015;">{r_label}</span></div>
        <div style="margin-top:22px;height:18px;border-radius:50px;background:#0d0015;overflow:hidden;border:1px solid #FFD700;">
            <div style="width:{risk}%;height:100%;background:linear-gradient(90deg,#7CFC00,#f39c12,#ff6b6b);transition:width 1s ease;"></div>
        </div>
        <div style="color:#cc99ff;margin-top:10px;">Retention Likelihood: <b style="color:#FFD700;">{retention}%</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    b1.markdown(f'<div class="metric-card"><div class="metric-label">Recency Risk</div><div class="metric-value" style="font-size:1.6rem;">{recency_risk:.0f}%</div></div>', unsafe_allow_html=True)
    b2.markdown(f'<div class="metric-card"><div class="metric-label">Frequency Risk</div><div class="metric-value" style="font-size:1.6rem;">{freq_risk:.0f}%</div></div>', unsafe_allow_html=True)
    b3.markdown(f'<div class="metric-card"><div class="metric-label">Monetary Risk</div><div class="metric-value" style="font-size:1.6rem;">{money_risk:.0f}%</div></div>', unsafe_allow_html=True)

# ---------------- FEATURE 3: AI EMAIL CAMPAIGN ----------------
with tabs[5]:
    st.markdown('<div class="royal-title">💌 AI Email Campaign</div>', unsafe_allow_html=True)
    st.markdown('<div class="royal-subtitle">✦ Personalized, copy-paste ready emails per segment ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    seg = st.selectbox("Select Segment", sorted(rfm["Segment"].unique()))
    brand = st.text_input("Your Brand Name", "Royal Store")
    offer = st.text_input("Offer / Discount", "20% OFF")

    templates = {
        "Champion": ("👑 You're Royalty, {name}!",
            "Dear Valued Champion,\n\nYou are among our TOP customers and we treasure you. As a thank-you, enjoy an exclusive {offer} plus VIP early access to our newest collection before anyone else.\n\nYour loyalty deserves a crown.\n\nWarm regards,\nThe {brand} Team"),
        "Loyal Customer": ("⭐ A Special Thank You, {name}",
            "Hi there,\n\nYour continued trust means the world to us. Here's a loyalty reward just for you: {offer} on your next order.\n\nThank you for being so loyal.\n\nCheers,\nThe {brand} Team"),
        "Potential Loyalist": ("🌱 Something Special Awaits, {name}",
            "Hello,\n\nWe noticed you love what we do! Here's {offer} to make your next purchase even better. We can't wait to see you again.\n\nBest,\nThe {brand} Team"),
        "At Risk": ("⚠️ We Miss You, {name} — Here's {offer}",
            "Hi,\n\nIt's been a while and we'd hate to see you go. Come back and enjoy {offer} as our way of saying we value you.\n\nThis offer is waiting for you.\n\nWith love,\nThe {brand} Team"),
        "Lost": ("😢 Come Back, {name}? {offer} Inside",
            "Hello,\n\nWe truly miss having you around. Here's an exclusive {offer} to welcome you back home. No strings attached.\n\nHope to see you soon,\nThe {brand} Team"),
        "Regular": ("🛍️ Just For You, {name} — {offer}",
            "Hi,\n\nThanks for shopping with us! Enjoy {offer} on your next visit and discover something you'll love.\n\nSee you soon,\nThe {brand} Team"),
    }
    subject, body = templates.get(seg, templates["Regular"])
    subject = subject.format(name="Valued Customer", offer=offer)
    body = body.format(offer=offer, brand=brand)

    n_in_seg = int((rfm["Segment"] == seg).sum())
    st.markdown(f'<div class="rec-card">📬 This campaign targets <b style="color:#FFD700;">{n_in_seg:,}</b> customers in the <b style="color:#FFD700;">{seg}</b> segment.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Subject Line</div>', unsafe_allow_html=True)
    st.code(subject, language=None)
    st.markdown('<div class="section-header">Email Body</div>', unsafe_allow_html=True)
    st.code(body, language=None)

# ---------------- FEATURE 4: REVENUE SIMULATOR ----------------
with tabs[6]:
    st.markdown('<div class="royal-title">📈 Revenue Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="royal-subtitle">✦ What if we retained our At-Risk customers? ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    at_risk = rfm[rfm["Segment"].isin(["At Risk", "Lost"])]
    at_risk_clv = at_risk["CLV"].sum()
    n_risk = len(at_risk)

    retain_pct = st.slider("% of At-Risk/Lost customers we successfully retain", 0, 100, 30)
    campaign_cost = st.number_input(f"Campaign cost per customer ({CUR})", 0.0, 1000.0, 5.0)

    saved_revenue = at_risk_clv * (retain_pct / 100)
    total_cost = n_risk * (retain_pct / 100) * campaign_cost
    net_gain = saved_revenue - total_cost
    roi = (net_gain / total_cost * 100) if total_cost > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-icon">⚠️</div><div class="metric-value">{n_risk:,}</div><div class="metric-label">At-Risk Customers</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-icon">💰</div><div class="metric-value">{CUR}{saved_revenue/1_000_000:.2f}M</div><div class="metric-label">Revenue Saved</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-icon">💸</div><div class="metric-value">{CUR}{total_cost:,.0f}</div><div class="metric-label">Campaign Cost</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-icon">🚀</div><div class="metric-value">{roi:,.0f}%</div><div class="metric-label">ROI</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="predict-card"><div class="predict-label">Net Business Gain</div><div class="predict-value" style="color:{"#7CFC00" if net_gain>=0 else "#ff6b6b"};">{CUR}{net_gain:,.0f}</div></div>', unsafe_allow_html=True)

    figr, axr = plt.subplots(figsize=(12, 4))
    pcts = list(range(0, 101, 10))
    gains = [at_risk_clv * (p / 100) - n_risk * (p / 100) * campaign_cost for p in pcts]
    axr.plot(pcts, gains, color='#FFD700', marker='o', linewidth=2.5)
    axr.fill_between(pcts, gains, color='#cc99ff', alpha=0.15)
    figr.patch.set_facecolor('#1a0030'); axr.set_facecolor('#1a0030')
    axr.tick_params(colors='#cc99ff'); axr.set_title('Net Gain vs Retention Rate', color='#FFD700')
    axr.set_xlabel('% Retained', color='#cc99ff'); axr.set_ylabel(f'Net Gain ({CUR})', color='#cc99ff')
    for s in axr.spines.values(): s.set_edgecolor('#FFD700')
    st.pyplot(figr)

# ---------------- FEATURE 5: CUSTOMER DNA CARD ----------------
with tabs[7]:
    st.markdown('<div class="royal-title">🏆 Customer DNA Card</div>', unsafe_allow_html=True)
    st.markdown('<div class="royal-subtitle">✦ A premium profile card for any customer ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    cust_id = st.selectbox("Select Customer ID", rfm["Customer ID"].tolist())
    row = rfm[rfm["Customer ID"] == cust_id].iloc[0]

    seg_colors = {"Champion": "#FFD700", "Loyal Customer": "#cc99ff", "Potential Loyalist": "#9b59b6",
                  "At Risk": "#f39c12", "Lost": "#ff6b6b", "Regular": "#8e44ad"}
    sc = seg_colors.get(row["Segment"], "#FFD700")

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown(f"""
        <div class="dna-card">
            <div class="dna-chip"></div>
            <div class="dna-id">#{str(cust_id).replace('.0','')}</div>
            <div style="color:#cc99ff;letter-spacing:2px;font-size:.8rem;margin-top:4px;">CUSTOMER DNA • {CUR} CLV {row['CLV']:,.0f}</div>
            <div class="dna-row">
                <div><div class="dna-k">Recency</div><div class="dna-v">{int(row['Recency'])}d</div></div>
                <div><div class="dna-k">Frequency</div><div class="dna-v">{int(row['Frequency'])}x</div></div>
                <div><div class="dna-k">Monetary</div><div class="dna-v">{CUR}{row['Monetary']:,.0f}</div></div>
            </div>
            <div style="margin-top:22px;display:flex;justify-content:space-between;align-items:center;">
                <span class="pill" style="background:{sc};color:#0d0015;">{row['Segment']}</span>
                <span style="color:#FFD700;font-family:'Playfair Display';font-size:1.1rem;">{row['CLV_Segment']} VALUE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with right:
        r_n = 1 - min(row['Recency'] / max(rfm['Recency'].quantile(0.95), 1), 1)
        f_n = min(row['Frequency'] / max(rfm['Frequency'].quantile(0.95), 1), 1)
        m_n = min(row['Monetary'] / max(rfm['Monetary'].quantile(0.95), 1), 1)
        cats = ['Recency', 'Frequency', 'Monetary']
        vals = [r_n, f_n, m_n]; vals += vals[:1]
        angles = [n / float(len(cats)) * 2 * pi for n in range(len(cats))]; angles += angles[:1]
        figd, axd = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        axd.set_facecolor('#1a0030'); figd.patch.set_facecolor('#1a0030')
        axd.plot(angles, vals, color='#FFD700', linewidth=2)
        axd.fill(angles, vals, color='#FFD700', alpha=0.25)
        axd.set_xticks(angles[:-1]); axd.set_xticklabels(cats, color='#cc99ff')
        axd.set_yticklabels([]); axd.set_title('RFM Strength', color='#FFD700', pad=20)
        st.pyplot(figd)
