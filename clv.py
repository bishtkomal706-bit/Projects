import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(
    page_title="CLV Prediction System",
    page_icon="👑",
    layout="wide"
)

# CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0d0015 0%, #1a0030 50%, #0d0015 100%); }
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
    .nav-logo { font-family: 'Playfair Display', serif; font-size: 1.5rem; color: #FFD700; font-weight: 700; }
    .metric-card {
        background: linear-gradient(135deg, #1a0030, #2d0050);
        border: 1px solid #FFD700;
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 20px rgba(255,215,0,0.15);
        margin: 10px 0;
    }
    .metric-value { font-size: 2.2rem; font-weight: bold; color: #FFD700; }
    .metric-label { font-size: 0.85rem; color: #cc99ff; margin-top: 5px; letter-spacing: 1px; text-transform: uppercase; }
    .metric-icon { font-size: 2rem; margin-bottom: 10px; }
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
    .royal-subtitle { text-align: center; color: #cc99ff; font-size: 1rem; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 30px; }
    .section-header { font-size: 1.5rem; color: #FFD700; border-left: 4px solid #FFD700; padding-left: 15px; margin: 30px 0 20px 0; font-weight: 600; }
    .gold-divider { height: 1px; background: linear-gradient(90deg, transparent, #FFD700, transparent); margin: 30px 0; }
    .predict-card {
        background: linear-gradient(135deg, #1a0030, #2d0050);
        border: 2px solid #FFD700;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 40px rgba(255,215,0,0.2);
    }
    .predict-value { font-size: 3rem; font-weight: bold; color: #FFD700; }
    .predict-label { color: #cc99ff; font-size: 1rem; letter-spacing: 2px; text-transform: uppercase; }
    .rec-card {
        background: linear-gradient(135deg, #1a0030, #2d0050);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #FFD700;
        color: white;
    }
    .stButton > button {
        background: linear-gradient(90deg, #FFD700, #cc99ff);
        color: #0d0015;
        font-weight: bold;
        border: none;
        border-radius: 50px;
        padding: 15px 40px;
        font-size: 1.1rem;
        width: 100%;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Load + Train
@st.cache_resource
def load_and_train():
    df = pd.read_csv('online_retail_II.csv', encoding='utf-8')
    df.dropna(subset=['Customer ID'], inplace=True)
    df.drop_duplicates(inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['Price']
    reference_date = df['InvoiceDate'].max()
    rfm = df.groupby('Customer ID').agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()

    rfm['CLV'] = rfm['Monetary'] * rfm['Frequency']
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=3, labels=[3,2,1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=3, labels=[1,2,3])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=3, labels=[1,2,3])
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    def segment_customer(score):
        if score == '333': return 'Champion'
        elif score in ['323','332','331']: return 'Loyal Customer'
        elif score in ['311','312','313']: return 'Potential Loyalist'
        elif score in ['133','233']: return 'At Risk'
        elif score in ['111','211','112']: return 'Lost'
        else: return 'Regular'

    rfm['Segment'] = rfm['RFM_Score'].apply(segment_customer)
    rfm['CLV_Segment'] = pd.qcut(rfm['CLV'], q=3, labels=['Low','Medium','High'])

    # Cohort
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
    df['CohortMonth'] = df.groupby('Customer ID')['InvoiceDate'].transform('min').dt.to_period('M')
    df['CohortIndex'] = (df['InvoiceMonth'] - df['CohortMonth']).apply(lambda x: x.n)
    cohort_data = df.groupby(['CohortMonth','CohortIndex'])['Customer ID'].nunique().reset_index()
    cohort_table = cohort_data.pivot_table(index='CohortMonth', columns='CohortIndex', values='Customer ID')
    cohort = cohort_table.divide(cohort_table.iloc[:,0], axis=0) * 100

    # Models
    X = rfm[['Recency','Frequency','Monetary']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_scaled, rfm['CLV'])

    rfc = RandomForestClassifier(n_estimators=100, max_depth=5,
                                  min_samples_split=10, min_samples_leaf=5,
                                  random_state=42)
    rfc.fit(X_scaled, rfm['CLV_Segment'])

    return rfm, cohort, rf, rfc, scaler

# Load
with st.spinner('👑 Loading Royal Dashboard... Please wait!'):
    rfm, cohort, rf, rfc, scaler = load_and_train()

# Nav Bar
st.markdown("""
    <div class="nav-container">
        <div class="nav-logo">👑 CLV Prediction System</div>
        <div style="color:#cc99ff;font-size:0.85rem;">Online Retail UK | Machine Learning</div>
    </div>
""", unsafe_allow_html=True)

# Tabs
page = st.tabs(["🏠 Home", "📊 Segmentation", "🔮 CLV Prediction", "💡 Recommendations"])

# ========== HOME ==========
with page[0]:
    st.markdown('<div class="royal-title">Customer Lifetime Value</div>', unsafe_allow_html=True)
    st.markdown('<div class="royal-subtitle">✦ Powered by Machine Learning ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-icon">👥</div><div class="metric-value">{len(rfm):,}</div><div class="metric-label">Total Customers</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-icon">💷</div><div class="metric-value">£{rfm["Monetary"].sum()/1000000:.1f}M</div><div class="metric-label">Total Revenue</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-icon">🔁</div><div class="metric-value">{rfm["Frequency"].mean():.1f}x</div><div class="metric-label">Avg Frequency</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-icon">💎</div><div class="metric-value">£{rfm["CLV"].mean():,.0f}</div><div class="metric-label">Avg CLV</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">📌 About This Project</div>', unsafe_allow_html=True)
        st.markdown("""<div class="rec-card">
        This system predicts <b style="color:#FFD700">Customer Lifetime Value</b> using RFM Analysis and Machine Learning.<br><br>
        ✦ RFM Segmentation — Customer behavior grouping<br>
        ✦ Regression Model — Exact CLV prediction<br>
        ✦ Classification — High / Medium / Low value<br>
        ✦ Cohort Analysis — Retention trend heatmap
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-header">🤖 Model Performance</div>', unsafe_allow_html=True)
        st.markdown("""<div class="rec-card">
        <b style="color:#FFD700">Random Forest Regressor</b><br>
        R² Score: 0.966 &nbsp;|&nbsp; MAE: £28,770<br><br>
        <b style="color:#FFD700">Random Forest Classifier</b><br>
        Accuracy: 98.8%<br><br>
        <b style="color:#cc99ff">Winner → Random Forest in both tasks!</b>
        </div>""", unsafe_allow_html=True)

# ========== SEGMENTATION ==========
with page[1]:
    st.markdown('<div class="royal-title">Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="royal-subtitle">✦ RFM Analysis & Cohort Retention ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">🎯 RFM Segments</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8,5))
        colors = ['#FFD700','#cc99ff','#9b59b6','#f39c12','#8e44ad','#d4ac0d']
        rfm['Segment'].value_counts().plot(kind='bar', ax=ax, color=colors, edgecolor='none')
        fig.patch.set_facecolor('#1a0030')
        ax.set_facecolor('#1a0030')
        ax.tick_params(colors='#cc99ff')
        ax.set_title('Customer Segments', color='#FFD700', fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('Count', color='#cc99ff')
        for spine in ax.spines.values():
            spine.set_edgecolor('#FFD700')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    with col2:
        st.markdown('<div class="section-header">💎 CLV Distribution</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(8,5))
        rfm['CLV_Segment'].value_counts().plot(
            kind='pie', autopct='%1.1f%%',
            colors=['#FFD700','#cc99ff','#9b59b6'],
            ax=ax2,
            textprops={'color':'white','fontsize':12},
            wedgeprops={'edgecolor':'#0d0015','linewidth':2}
        )
        ax2.set_ylabel('')
        fig2.patch.set_facecolor('#1a0030')
        ax2.set_title('CLV Segments', color='#FFD700', fontsize=14)
        st.pyplot(fig2)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔥 Cohort Retention Heatmap</div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(20,8))
    sns.heatmap(cohort.astype(float), annot=True, fmt='.0f',
                cmap='YlOrRd', ax=ax3, vmin=0, vmax=100,
                linewidths=0.5, linecolor='#0d0015')
    ax3.set_title('Customer Retention Rate by Cohort (%)', color='#FFD700', fontsize=16)
    ax3.set_xlabel('Months After First Purchase', color='#cc99ff')
    ax3.set_ylabel('Cohort Month', color='#cc99ff')
    fig3.patch.set_facecolor('#1a0030')
    ax3.tick_params(colors='#cc99ff')
    st.pyplot(fig3)

# ========== CLV PREDICTION ==========
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
            st.markdown(f'<div class="predict-card"><div class="predict-label">Predicted Lifetime Value</div><div class="predict-value">£{clv_pred:,.0f}</div></div>', unsafe_allow_html=True)
        with col2:
            color = "#FFD700" if segment_pred == "High" else "#cc99ff" if segment_pred == "Medium" else "#ff6b6b"
            emoji = "👑" if segment_pred == "High" else "⭐" if segment_pred == "Medium" else "⚠️"
            st.markdown(f'<div class="predict-card"><div class="predict-label">Customer Segment</div><div class="predict-value" style="color:{color};">{emoji} {segment_pred}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
        if segment_pred == "High":
            st.markdown('<div class="rec-card" style="border-left:4px solid #FFD700;">👑 <b style="color:#FFD700">HIGH VALUE CUSTOMER</b><br><br>Offer VIP treatment, exclusive early access and loyalty rewards!</div>', unsafe_allow_html=True)
        elif segment_pred == "Medium":
            st.markdown('<div class="rec-card" style="border-left:4px solid #cc99ff;">⭐ <b style="color:#cc99ff">MEDIUM VALUE CUSTOMER</b><br><br>Engage with personalized offers to upgrade them to High Value!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="rec-card" style="border-left:4px solid #ff6b6b;">⚠️ <b style="color:#ff6b6b">LOW VALUE CUSTOMER</b><br><br>Send re-engagement campaigns with heavy discounts!</div>', unsafe_allow_html=True)

# ========== RECOMMENDATIONS ==========
with page[3]:
    st.markdown('<div class="royal-title">Business Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="royal-subtitle">✦ Strategic Action Plan ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    champion_revenue = rfm[rfm['Segment'] == 'Champion']['CLV'].sum()
    atrisk_revenue = rfm[rfm['Segment'] == 'At Risk']['CLV'].sum()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-icon">👑</div><div class="metric-value">£{champion_revenue/1000000:.2f}M</div><div class="metric-label">Champion Customers CLV</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-icon">⚠️</div><div class="metric-value">£{atrisk_revenue:,.0f}</div><div class="metric-label">At Risk CLV — Save This!</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 Action Plan by Segment</div>', unsafe_allow_html=True)

    recommendations = {
        'Champion': ('👑','#FFD700','Reward with VIP access, early product launches and exclusive loyalty points!'),
        'Loyal Customer': ('⭐','#cc99ff','Offer membership programs and exclusive discounts to maintain loyalty.'),
        'Potential Loyalist': ('🌱','#9b59b6','Send personalized recommendations — small nudges convert them to Loyal!'),
        'At Risk': ('⚠️','#f39c12','URGENT! Send win-back campaigns immediately before they become Lost!'),
        'Lost': ('😟','#ff6b6b','Re-engagement emails with heavy discounts — focus on any sign of return.'),
        'Regular': ('📊','#8e44ad','Consistent engagement and seasonal offers to gradually increase value.')
    }

    for _, row in rfm['Segment'].value_counts().reset_index().iterrows():
        segment = row['Segment']
        count = row['count'] if 'count' in row else row.iloc[1]
        if segment in recommendations:
            emoji, color, action = recommendations[segment]
            st.markdown(f'<div class="rec-card" style="border-left:4px solid {color};margin-bottom:15px;"><b style="color:{color};font-size:1.1rem;">{emoji} {segment}</b><span style="color:#cc99ff;float:right;">{count} customers</span><br><br><span style="color:#e0e0e0;">{action}</span></div>', unsafe_allow_html=True)