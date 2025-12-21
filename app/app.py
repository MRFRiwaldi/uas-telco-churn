import os
import io
import base64
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(
    page_title="Prediksi Pelanggan Berhenti Berlangganan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Initialize Session State
# =========================
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# =========================
# Theme Configuration
# =========================
def get_theme_colors():
    if st.session_state.dark_mode:
        return {
            'bg_gradient': 'radial-gradient(circle at top, #1d4ed8 0, #020617 55%)',
            'card_bg': 'rgba(15, 23, 42, 0.82)',
            'text_primary': '#ffffff',
            'text_secondary': '#cbd5f5',
            'border_color': 'rgba(148, 163, 184, 0.4)',
            'accent_color': '#22c55e',
            'chart_template': 'plotly_dark'
        }
    else:
        return {
            'bg_gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            'card_bg': 'rgba(255, 255, 255, 0.9)',
            'text_primary': '#1a202c',
            'text_secondary': '#4a5568',
            'border_color': 'rgba(99, 102, 241, 0.3)',
            'accent_color': '#667eea',
            'chart_template': 'plotly_white'
        }

theme = get_theme_colors()

# =========================
# CSS Kustom – Tema Modern / Glassmorphism
# =========================
st.markdown(
    f"""
    <style>
    /* Background */
    .stApp {{
        background: {theme['bg_gradient']};
        transition: all 0.3s ease;
    }}

    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}

    /* Headers */
    .header-title {{
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.6s ease;
    }}
    
    .header-subtitle {{
        font-size: 1.1rem;
        color: {theme['text_secondary']};
        max-width: 800px;
        margin-bottom: 2rem;
        animation: fadeIn 0.8s ease;
    }}

    /* Glass Cards */
    .glass-card {{
        background: {theme['card_bg']};
        border-radius: 1.4rem;
        padding: 1.5rem;
        border: 1px solid {theme['border_color']};
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.70);
        backdrop-filter: blur(16px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeInUp 0.5s ease;
    }}

    .glass-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(15, 23, 42, 0.85);
    }}

    /* Metrics Cards */
    .metric-card {{
        background: {theme['card_bg']};
        border-radius: 1rem;
        padding: 1.2rem;
        border: 1px solid {theme['border_color']};
        text-align: center;
        transition: all 0.3s ease;
    }}

    .metric-card:hover {{
        transform: scale(1.05);
        border-color: {theme['accent_color']};
    }}

    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {theme['accent_color']};
        margin: 0.5rem 0;
    }}

    .metric-label {{
        font-size: 0.9rem;
        color: {theme['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }}

    /* Buttons */
    .stButton>button {{
        width: 100%;
        border-radius: 999px;
        padding: 0.9rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border: none;
        font-size: 0.9rem;
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
        box-shadow: 0 15px 35px rgba(16, 185, 129, 0.55);
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(16, 185, 129, 0.70);
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1rem;
        background: transparent;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: {theme['card_bg']};
        border-radius: 1rem 1rem 0 0;
        padding: 0.8rem 2rem;
        border: 1px solid {theme['border_color']};
        color: {theme['text_secondary']};
        font-weight: 600;
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: transparent;
    }}

    /* Result Chips */
    .result-chip-danger {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        background: rgba(248, 113, 113, 0.2);
        border: 2px solid rgba(248, 113, 113, 0.6);
        color: #fca5a5;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        font-weight: 600;
        animation: pulse 2s infinite;
    }}
    
    .result-chip-safe {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        background: rgba(45, 212, 191, 0.2);
        border: 2px solid rgba(45, 212, 191, 0.6);
        color: #5eead4;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        font-weight: 600;
        animation: pulse 2s infinite;
    }}

    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}

    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    @keyframes fadeInDown {{
        from {{
            opacity: 0;
            transform: translateY(-30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
    }}

    /* Dark Mode Toggle */
    .theme-toggle {{
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 999;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Muat Model dan Data
# =========================
@st.cache_resource
def load_model():
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_best.pkl")
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

# =========================
# Sidebar - Theme Toggle & Info
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    
    # Theme Toggle
    if st.button("🌓 Toggle Dark/Light Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    
    st.markdown("---")
    
    # Info
    st.markdown("### 📊 Informasi Dataset")
    st.metric("Total Pelanggan", f"{len(df):,}")
    
    churn_count = len(df[df['Churn'] == 'Yes'])
    churn_rate = (churn_count / len(df)) * 100
    st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    st.markdown("---")
    
    st.markdown("### 📝 Tentang Aplikasi")
    st.info(
        """
        Aplikasi ini menggunakan **Machine Learning** untuk memprediksi 
        kemungkinan pelanggan berhenti berlangganan (churn).
        
        **Fitur:**
        - 🔮 Prediksi real-time
        - 📊 Dashboard interaktif
        - 📈 Analisis feature importance
        - 💾 Export hasil prediksi
        """
    )

# =========================
# Header
# =========================
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown(
        """
        <div class="header-title">📊 Prediksi Customer Churn</div>
        <div class="header-subtitle">
            Platform analitik berbasis AI untuk memprediksi dan mencegah pelanggan berhenti berlangganan
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

# =========================
# Tab Navigation
# =========================
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Dashboard", "📈 Analisis"])

# =========================
# TAB 1: PREDIKSI
# =========================
with tab1:
    left_col, right_col = st.columns([2.2, 1.3])
    
    # Form Input
    with left_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Data Pelanggan")
        st.caption("Isi formulir di bawah untuk memprediksi risiko churn")
        st.markdown("")
        
        c1, c2 = st.columns(2)
        
        with c1:
            gender = st.selectbox("👤 Jenis Kelamin", ["Female", "Male"])
            senior = st.selectbox("👴 Status Senior", ["Bukan Senior (0)", "Senior (1)"])
            partner = st.selectbox("💑 Memiliki Pasangan", ["Yes", "No"])
            dependents = st.selectbox("👶 Memiliki Tanggungan", ["Yes", "No"])
            tenure = st.slider("📅 Lama Berlangganan (bulan)", 0, 100, 12,
                             help="Berapa lama pelanggan sudah berlangganan")
            phone_service = st.selectbox("📞 Layanan Telepon", ["Yes", "No"])
            multiple_lines = st.selectbox("📱 Multiple Lines", 
                                        ["No phone service", "No", "Yes"])
            internet_service = st.selectbox("🌐 Jenis Internet", 
                                          ["DSL", "Fiber optic", "No"])
        
        with c2:
            online_security = st.selectbox("🔒 Keamanan Online", 
                                         ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("💾 Backup Online", 
                                       ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("🛡️ Proteksi Perangkat", 
                                           ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("🔧 Dukungan Teknis", 
                                      ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("📺 Streaming TV", 
                                      ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("🎬 Streaming Film", 
                                          ["No", "Yes", "No internet service"])
            contract = st.selectbox("📄 Jenis Kontrak", 
                                  ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("📧 Tagihan Digital", ["Yes", "No"])
        
        st.markdown("---")
        
        mc1, mc2 = st.columns(2)
        with mc1:
            monthly_charges = st.number_input("💰 Biaya Bulanan (USD)", 
                                            0.0, 1000.0, 70.0, 1.0)
        with mc2:
            total_charges = st.number_input("💵 Total Biaya (USD)", 
                                          0.0, 10000.0, 500.0, 10.0)
        
        payment_method = st.selectbox("💳 Metode Pembayaran",
            ["Electronic check", "Mailed check", 
             "Bank transfer (automatic)", "Credit card (automatic)"])
        
        st.markdown("")
        show_data = st.checkbox("📋 Tampilkan ringkasan data")
        
        # Prepare input
        senior_value = 0 if senior.startswith("Bukan") else 1
        input_dict = {
            "gender": [gender],
            "SeniorCitizen": [senior_value],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
        }
        input_df = pd.DataFrame(input_dict)
        
        if show_data:
            st.dataframe(input_df, use_container_width=True)
        
        st.markdown("")
        predict_clicked = st.button("🚀 Jalankan Prediksi")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Results Panel
    with right_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Hasil Prediksi")
        
        if not predict_clicked:
            st.info("👈 Isi form dan klik **Jalankan Prediksi**")
        else:
            with st.spinner("🔄 Menganalisis data..."):
                proba_churn = model.predict_proba(input_df)[0][1]
                pred = model.predict(input_df)[0]
                
                # Save to history
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'prediction': 'Churn' if pred == 1 else 'No Churn',
                    'probability': proba_churn,
                    'tenure': tenure,
                    'monthly_charges': monthly_charges
                })
                
                # Display result
                if pred == 1:
                    st.markdown(
                        '<div class="result-chip-danger">⚠️ RISIKO TINGGI</div>',
                        unsafe_allow_html=True
                    )
                    st.error(f"**Prediksi:** Pelanggan berpotensi CHURN")
                    
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=proba_churn * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Risiko Churn (%)", 'font': {'size': 20}},
                        delta={'reference': 50, 'increasing': {'color': "red"}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1},
                            'bar': {'color': "darkred"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 30], 'color': 'lightgreen'},
                                {'range': [30, 70], 'color': 'yellow'},
                                {'range': [70, 100], 'color': 'lightcoral'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': proba_churn * 100
                            }
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.markdown("#### 💡 Rekomendasi")
                    st.warning("""
                    - Tawarkan diskon atau promo khusus
                    - Upgrade ke kontrak jangka panjang
                    - Tingkatkan customer service
                    - Berikan loyalty rewards
                    """)
                    
                else:
                    st.markdown(
                        '<div class="result-chip-safe">✅ RISIKO RENDAH</div>',
                        unsafe_allow_html=True
                    )
                    st.success(f"**Prediksi:** Pelanggan cenderung BERTAHAN")
                    
                    # Gauge chart
                    retention_prob = 1 - proba_churn
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=retention_prob * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Tingkat Retensi (%)", 'font': {'size': 20}},
                        delta={'reference': 50, 'increasing': {'color': "green"}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1},
                            'bar': {'color': "darkgreen"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 30], 'color': 'lightcoral'},
                                {'range': [30, 70], 'color': 'yellow'},
                                {'range': [70, 100], 'color': 'lightgreen'}
                            ],
                            'threshold': {
                                'line': {'color': "green", 'width': 4},
                                'thickness': 0.75,
                                'value': retention_prob * 100
                            }
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.markdown("#### 💡 Rekomendasi")
                    st.info("""
                    - Pertahankan kualitas layanan
                    - Tawarkan program referral
                    - Cross-sell layanan tambahan
                    - Minta feedback untuk improvement
                    """)
                
                # Export button
                st.markdown("---")
                csv = input_df.copy()
                csv['Prediction'] = 'Churn' if pred == 1 else 'No Churn'
                csv['Churn_Probability'] = proba_churn
                csv_string = csv.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Hasil (CSV)",
                    data=csv_string,
                    file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TAB 2: DASHBOARD
# =========================
with tab2:
    st.markdown("### 📊 Dashboard Analitik")
    st.caption("Visualisasi dan statistik dataset pelanggan")
    st.markdown("")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df)
    churned = len(df[df['Churn'] == 'Yes'])
    retained = len(df[df['Churn'] == 'No'])
    churn_rate = (churned / total_customers) * 100
    avg_tenure = df['tenure'].mean()
    avg_monthly = df['MonthlyCharges'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Pelanggan</div>
            <div class="metric-value">{total_customers:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Churn Rate</div>
            <div class="metric-value" style="color: #ef4444;">{churn_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Tenure</div>
            <div class="metric-value">{avg_tenure:.0f}</div>
            <div style="font-size: 0.75rem; color: #94a3b8;">Bulan</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Monthly</div>
            <div class="metric-value">${avg_monthly:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn Distribution
        churn_data = df['Churn'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=['Retained', 'Churned'],
            values=[churn_data['No'], churn_data['Yes']],
            hole=0.5,
            marker=dict(colors=['#22c55e', '#ef4444']),
            textinfo='label+percent',
            textfont=dict(size=14)
        )])
        fig.update_layout(
            title="Distribusi Churn vs Retained",
            template=theme['chart_template'],
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Contract Type Distribution
        contract_churn = df.groupby(['Contract', 'Churn']).size().unstack()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Retained',
            x=contract_churn.index,
            y=contract_churn['No'],
            marker_color='#22c55e'
        ))
        fig.add_trace(go.Bar(
            name='Churned',
            x=contract_churn.index,
            y=contract_churn['Yes'],
            marker_color='#ef4444'
        ))
        fig.update_layout(
            title="Churn Rate by Contract Type",
            barmode='stack',
            template=theme['chart_template'],
            height=400,
            xaxis_title="Contract Type",
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        # Tenure Distribution
        fig = px.histogram(
            df,
            x='tenure',
            color='Churn',
            nbins=30,
            title="Distribusi Tenure by Churn Status",
            color_discrete_map={'Yes': '#ef4444', 'No': '#22c55e'},
            template=theme['chart_template']
        )
        fig.update_layout(height=400, xaxis_title="Tenure (months)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly Charges Distribution
        fig = px.box(
            df,
            x='Churn',
            y='MonthlyCharges',
            color='Churn',
            title="Monthly Charges by Churn Status",
            color_discrete_map={'Yes': '#ef4444', 'No': '#22c55e'},
            template=theme['chart_template']
        )
        fig.update_layout(height=400, yaxis_title="Monthly Charges ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Payment Method Analysis
    st.markdown("---")
    payment_churn = df.groupby(['PaymentMethod', 'Churn']).size().unstack()
    payment_churn_rate = (payment_churn['Yes'] / (payment_churn['Yes'] + payment_churn['No']) * 100).sort_values(ascending=True)
    
    fig = go.Figure(go.Bar(
        x=payment_churn_rate.values,
        y=payment_churn_rate.index,
        orientation='h',
        marker=dict(
            color=payment_churn_rate.values,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Churn %")
        ),
        text=[f'{val:.1f}%' for val in payment_churn_rate.values],
        textposition='auto'
    ))
    fig.update_layout(
        title="Churn Rate by Payment Method",
        template=theme['chart_template'],
        height=350,
        xaxis_title="Churn Rate (%)",
        yaxis_title="Payment Method"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3: ANALISIS
# =========================
with tab3:
    st.markdown("### 📈 Feature Importance & Insights")
    st.caption("Analisis faktor-faktor yang mempengaruhi customer churn")
    st.markdown("")
    
    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        feature_names = input_df.columns.tolist()
        importances = model.feature_importances_
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True).tail(15)
        
        # Plot
        fig = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker=dict(
                color=importance_df['Importance'],
                colorscale='Viridis',
                showscale=True
            ),
            text=[f'{val:.3f}' for val in importance_df['Importance']],
            textposition='auto'
        ))
        fig.update_layout(
            title="Top 15 Most Important Features",
            template=theme['chart_template'],
            height=600,
            xaxis_title="Importance Score",
            yaxis_title="Feature"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance tidak tersedia untuk model ini.")
    
    st.markdown("---")
    
    # Comparison Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 Profile: Churned Customers")
        churned_df = df[df['Churn'] == 'Yes']
        st.markdown(f"""
        - **Avg Tenure:** {churned_df['tenure'].mean():.1f} months
        - **Avg Monthly Charges:** ${churned_df['MonthlyCharges'].mean():.2f}
        - **Most Common Contract:** {churned_df['Contract'].mode()[0]}
        - **Most Common Internet:** {churned_df['InternetService'].mode()[0]}
        - **Most Common Payment:** {churned_df['PaymentMethod'].mode()[0]}
        """)
    
    with col2:
        st.markdown("#### 🟢 Profile: Retained Customers")
        retained_df = df[df['Churn'] == 'No']
        st.markdown(f"""
        - **Avg Tenure:** {retained_df['tenure'].mean():.1f} months
        - **Avg Monthly Charges:** ${retained_df['MonthlyCharges'].mean():.2f}
        - **Most Common Contract:** {retained_df['Contract'].mode()[0]}
        - **Most Common Internet:** {retained_df['InternetService'].mode()[0]}
        - **Most Common Payment:** {retained_df['PaymentMethod'].mode()[0]}
        """)
    
    st.markdown("---")
    
    # Key Insights
    st.markdown("### 💡 Key Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.success("""
        **Faktor Proteksi (Menurunkan Risiko Churn):**
        - Kontrak jangka panjang (1-2 tahun)
        - Tenure yang lebih lama (>12 bulan)
        - Layanan keamanan dan backup
        - Payment otomatis
        """)
    
    with insights_col2:
        st.error("""
        **Faktor Risiko (Meningkatkan Churn):**
        - Kontrak month-to-month
        - Tenure pendek (<6 bulan)
        - Electronic check payment
        - Tidak ada layanan proteksi
        """)
    
    # Prediction History
    if st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("### 📜 Riwayat Prediksi Session Ini")
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df['timestamp'] = history_df['timestamp'].dt.strftime('%H:%M:%S')
        history_df['probability'] = history_df['probability'].apply(lambda x: f"{x*100:.1f}%")
        
        st.dataframe(
            history_df[['timestamp', 'prediction', 'probability', 'tenure', 'monthly_charges']],
            use_container_width=True,
            hide_index=True
        )
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

# =========================
# Footer
# =========================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>
        <p>📊 Customer Churn Prediction System | Powered by Machine Learning & Streamlit</p>
        <p>Made with ❤️ using Python, Streamlit, Plotly & Scikit-learn</p>
    </div>
    """,
    unsafe_allow_html=True
)
