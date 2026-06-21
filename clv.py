import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="CLV Prediction System",
    page_icon="👑",
    layout="wide"
)

# Load models aur data
rf = pickle.load(open('rf_regressor.pkl', 'rb'))
rfc = pickle.load(open('rf_classifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
rfm = pd.read_csv('rfm_data.csv')
cohort = pd.read_csv('cohort_data.csv', index_col=0)

# Custom CSS - Purple + Gold Royal Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0d0015 0%, #1a0030 50%, #0d0015 100%);
    }

    /* Top Navigation */
    .nav-container {
        background: linear-gradient(90deg, #1a0030, #2d0050, #1a0030);
        border-bottom: 2px solid #FFD700;
        padding: 15px 30px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 30px;
        border-radius: 0 0 20px 20px;
    }

    .nav-logo {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        color: #FFD700;
        font-weight: 700;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a0030, #2d0050);
        border: 1px solid #FFD700;
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.15);
        margin: 10px 0;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #FFD700;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #cc99ff;
        margin-top: 5px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .metric-icon {
        font-size: 2rem;
        margin-bottom: 10px;
    }

    /* Title */
    .royal-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FFD700, #cc99ff, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }

    .royal-subtitle {
        text-align: center;
        color: #cc99ff;
        font-size: 1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 30px;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        color: #FFD700;
        border-left: 4px solid #FFD700;
        padding-left: 15px;
        margin: 30px 0 20px 0;
        font-weight: 600;
    }

    /* Divider */
    .gold-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #FFD700, transparent);
        margin: 30px 0;
    }

    /* Prediction Box */
    .predict-card {
        background: linear-gradient(135deg, #1a0030, #2d0050);
        border: 2px solid #FFD700;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 40px rgba(255, 215, 0, 0.2);
    }

    .predict-value {
        font-size: 3rem;
        font-weight: bold;
        color: #FFD700;
    }

    .predict-label {
        color: #cc99ff;
        font-size: 1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* Recommendation Cards */
    .rec-card {
        background: linear-gradient(135deg, #1a0030, #2d0050);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #FFD700;
        color: white;
    }

    /* Stmetric override */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a0030, #2d0050);
        border: 1px solid #FFD700;
        border-radius: 15px;
        padding: 15px;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #FFD700, #cc99ff);
        color: #0d0015;
        font-weight: bold;
        border: none;
        border-radius: 50px;
        padding: 15px 40px;
        font-size: 1.1rem;
        letter-spacing: 1px;
        width: 100%;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.5);
    }

    /* Input fields */
    .stNumberInput > div > div > input {
        background: #1a0030;
        border: 1px solid #FFD700;
        color: white;
        border-radius: 10px;
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Top Navigation Bar
st.markdown("""
    <div class="nav-container">
        <div class="nav-logo">👑 CLV Prediction System</div>
        <div style="color: #cc99ff; font-size: 0.85rem;">Online Retail UK | Machine Learning</div>
    </div>
""", unsafe_allow_html=True)

# Navigation Tabs
page = st.tabs(["🏠 Home", "📊 Segmentation", "🔮 CLV Prediction", "💡 Recommendations"])

# ==================== PAGE 1 - HOME ====================
with page[0]:
    st.markdown('<div class="royal-title">Customer Lifetime Value</div>', unsafe_allow_html=True)
    st.markdown('<div class="royal-subtitle">✦ Powered by Machine Learning ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">👥</div>
            <div class="metric-value">{len(rfm):,}</div>
            <div class="metric-label">Total Customers</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">💷</div>
            <div class="metric-value">£{rfm['Monetary'].sum()/1000000:.1f}M</div>
            <div class="metric-label">Total Revenue</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🔁</div>
            <div class="metric-value">{rfm['Frequency'].mean():.1f}x</div>
            <div class="metric-label">Avg Frequency</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">💎</div>
            <div class="metric-value">£{rfm['CLV'].mean():,.0f}</div>
            <div class="metric-label">Avg CLV</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📌 About This Project</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="rec-card">
        This system predicts <b style="color:#FFD700">Customer Lifetime Value</b> using advanced RFM Analysis and Machine Learning on the UK Online Retail dataset.<br><br>
        ✦ RFM Segmentation — Customer behavior grouping<br>
        ✦ Regression Model — Exact CLV prediction<br>
        ✦ Classification — High / Medium / Low value<br>
        ✦ Cohort Analysis — Retention trend heatmap
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">🤖 Model Performance</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="rec-card">
        <b style="color:#FFD700">Random Forest Regressor</b><br>
        R² Score: 0.966 &nbsp;|&nbsp; MAE: £28,770<br><br>
        <b style="color:#FFD700">Random Forest Classifier</b><br>
        Accuracy: 98.8%<br><br>
        <b style="color:#cc99ff">Winner → Random Forest in both tasks!</b>
        </div>
        """, unsafe_allow_html=True)

# ==================== PAGE 2 - SEGMENTATION ====================
with page[1]:
    st.markdown('<div class="royal-title">Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="royal-subtitle">✦ RFM Analysis & Cohort Retention ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">🎯 RFM Segments</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#FFD700', '#cc99ff', '#9b59b6', '#f39c12', '#8e44ad', '#d4ac0d']
        rfm['Segment'].value_counts().plot(kind='bar', ax=ax, color=colors, edgecolor='none')
        fig.patch.set_facecolor('#1a0030')
        ax.set_facecolor('#1a0030')
        ax.tick_params(colors='#cc99ff')
        ax.set_title('Customer Segments Distribution', color='#FFD700', fontsize=14, pad=15)
        ax.set_xlabel('')
        ax.set_ylabel('Count', color='#cc99ff')
        for spine in ax.spines.values():
            spine.set_edgecolor('#FFD700')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    with col2:
        st.markdown('<div class="section-header">💎 CLV Distribution</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        colors_pie = ['#FFD700', '#cc99ff', '#9b59b6']
        rfm['CLV_Segment'].value_counts().plot(
            kind='pie',
            autopct='%1.1f%%',
            colors=colors_pie,
            ax=ax2,
            textprops={'color': 'white', 'fontsize': 12},
            wedgeprops={'edgecolor': '#0d0015', 'linewidth': 2}
        )
        ax2.set_ylabel('')
        fig2.patch.set_facecolor('#1a0030')
        ax2.set_facecolor('#1a0030')
        ax2.set_title('CLV Segment Distribution', color='#FFD700', fontsize=14, pad=15)
        st.pyplot(fig2)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔥 Cohort Retention Heatmap</div>', unsafe_allow_html=True)

    fig3, ax3 = plt.subplots(figsize=(20, 8))
    sns.heatmap(
        cohort.astype(float),
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        ax=ax3,
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor='#0d0015'
    )
    ax3.set_title('Customer Retention Rate by Cohort (%)', color='#FFD700', fontsize=16, pad=20)
    ax3.set_xlabel('Months After First Purchase', color='#cc99ff', fontsize=12)
    ax3.set_ylabel('Cohort Month', color='#cc99ff', fontsize=12)
    fig3.patch.set_facecolor('#1a0030')
    ax3.set_facecolor('#1a0030')
    ax3.tick_params(colors='#cc99ff')
    st.pyplot(fig3)

# ==================== PAGE 3 - CLV PREDICTION ====================
with page[2]:
    st.markdown('<div class="royal-title">CLV Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="royal-subtitle">✦ Enter Customer Details to Predict ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-header">📅 Recency</div>', unsafe_allow_html=True)
        recency = st.number_input("Days since last purchase", min_value=1, max_value=1000, value=30)

    with col2:
        st.markdown('<div class="section-header">🔁 Frequency</div>', unsafe_allow_html=True)
        frequency = st.number_input("Number of purchases", min_value=1, max_value=500, value=5)

    with col3:
        st.markdown('<div class="section-header">💷 Monetary</div>', unsafe_allow_html=True)
        monetary = st.number_input("Total amount spent (£)", min_value=1.0, max_value=100000.0, value=500.0)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    if st.button("👑 PREDICT CLV NOW"):
        input_data = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_data)

        clv_pred = rf.predict(input_scaled)[0]
        segment_pred = rfc.predict(input_scaled)[0]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="predict-card">
                <div class="predict-label">Predicted Lifetime Value</div>
                <div class="predict-value">£{clv_pred:,.0f}</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            color = "#FFD700" if segment_pred == "High" else "#cc99ff" if segment_pred == "Medium" else "#ff6b6b"
            emoji = "👑" if segment_pred == "High" else "⭐" if segment_pred == "Medium" else "⚠️"
            st.markdown(f"""
            <div class="predict-card">
                <div class="predict-label">Customer Segment</div>
                <div class="predict-value" style="color:{color};">{emoji} {segment_pred}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

        if segment_pred == "High":
            st.markdown("""
            <div class="rec-card" style="border-left: 4px solid #FFD700;">
            👑 <b style="color:#FFD700">HIGH VALUE CUSTOMER</b><br><br>
            This customer is your most valuable asset! Offer VIP treatment, exclusive early access, loyalty rewards and personalized service.
            </div>""", unsafe_allow_html=True)
        elif segment_pred == "Medium":
            st.markdown("""
            <div class="rec-card" style="border-left: 4px solid #cc99ff;">
            ⭐ <b style="color:#cc99ff">MEDIUM VALUE CUSTOMER</b><br><br>
            Good potential! Engage with personalized offers, product recommendations and targeted campaigns to upgrade them to High Value.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="rec-card" style="border-left: 4px solid #ff6b6b;">
            ⚠️ <b style="color:#ff6b6b">LOW VALUE CUSTOMER</b><br><br>
            Send re-engagement campaigns with heavy discounts and incentives to increase their purchase frequency and value.
            </div>""", unsafe_allow_html=True)

# ==================== PAGE 4 - RECOMMENDATIONS ====================
with page[3]:
    st.markdown('<div class="royal-title">Business Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="royal-subtitle">✦ Strategic Action Plan ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    champion_revenue = rfm[rfm['Segment'] == 'Champion']['CLV'].sum()
    atrisk_revenue = rfm[rfm['Segment'] == 'At Risk']['CLV'].sum()

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">👑</div>
            <div class="metric-value">£{champion_revenue/1000000:.2f}M</div>
            <div class="metric-label">Champion Customers CLV</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">⚠️</div>
            <div class="metric-value">£{atrisk_revenue:,.0f}</div>
            <div class="metric-label">At Risk CLV — Save This!</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 Action Plan by Segment</div>', unsafe_allow_html=True)

    recommendations = {
        'Champion': ('👑', '#FFD700', 'Reward them with VIP access, early product launches and exclusive loyalty points. These are your most valuable customers!'),
        'Loyal Customer': ('⭐', '#cc99ff', 'Offer membership programs, exclusive discounts and personalized communication to maintain their loyalty.'),
        'Potential Loyalist': ('🌱', '#9b59b6', 'Send personalized recommendations and onboarding emails. Small nudges can convert them to Loyal Customers!'),
        'At Risk': ('⚠️', '#f39c12', 'URGENT ACTION NEEDED! Send win-back campaigns and special offers immediately before they become Lost customers.'),
        'Lost': ('😟', '#ff6b6b', 'Send re-engagement emails with heavy discounts. Focus budget on customers showing any sign of returning.'),
        'Regular': ('📊', '#8e44ad', 'Nurture with consistent engagement, product updates and seasonal offers to gradually increase their value.')
    }

    segments = rfm['Segment'].value_counts().reset_index()
    segments.columns = ['Segment', 'Count']

    for _, row in segments.iterrows():
        segment = row['Segment']
        count = row['Count']
        if segment in recommendations:
            emoji, color, action = recommendations[segment]
            st.markdown(f"""
            <div class="rec-card" style="border-left: 4px solid {color}; margin-bottom: 15px;">
                <b style="color:{color}; font-size: 1.1rem;">{emoji} {segment}</b>
                <span style="color:#cc99ff; float:right;">{count} customers</span><br><br>
                <span style="color:#e0e0e0;">{action}</span>
            </div>""", unsafe_allow_html=True)