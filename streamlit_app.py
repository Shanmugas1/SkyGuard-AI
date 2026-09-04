import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy import stats

# Page config
st.set_page_config(page_title="SkyGuard AI", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main { padding: 2rem; }
    h1 { color: #1f77b4; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🌤️ SkyGuard AI: Weather Station Anomaly Detector")
st.markdown("""
Detect sensor malfunctions using Machine Learning.  
Upload your weather data (CSV) and discover anomalies instantly!
""")

st.markdown("---")

# Sidebar
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Select:", ["Upload & Detect", "About"])

if page == "Upload & Detect":
    
    # File uploader
    uploaded_file = st.file_uploader("📁 Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Records", len(df))
        with col2:
            st.metric("📈 Columns", len(df.columns))
        with col3:
            missing = df.isnull().sum().sum()
            st.metric("⚠️ Missing", missing)
        with col4:
            quality = (1 - missing / (len(df) * len(df.columns))) * 100
            st.metric("✅ Quality", f"{quality:.1f}%")
        
        # Show raw data
        with st.expander("📋 View Raw Data"):
            st.dataframe(df, use_container_width=True)
        
        # Check for required columns
        required_cols = ['Temperature_C', 'Humidity_percent', 'Pressure_hPa']
        
        if all(col in df.columns for col in required_cols):
            
            # Clean data
            df_clean = df[required_cols].copy()
            df_clean = df_clean.interpolate(method='linear').fillna(df_clean.mean())
            
            # Train model
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            predictions = iso_forest.fit_predict(df_clean)
            anomaly_scores = iso_forest.score_samples(df_clean)
            
            df['Anomaly'] = predictions == -1
            df['Anomaly_Score'] = anomaly_scores
            
            # Display results
            st.subheader("📈 Anomaly Detection Results")
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # Temperature
            axes[0].plot(df.index, df['Temperature_C'], color='red', linewidth=2, label='Temperature')
            anomaly_rows = df[df['Anomaly']]
            if len(anomaly_rows) > 0:
                axes[0].scatter(anomaly_rows.index, anomaly_rows['Temperature_C'], 
                              color='red', s=100, marker='X', label='Anomalies', zorder=5)
            axes[0].set_ylabel('Temperature (°C)')
            axes[0].set_title('Temperature with Detected Anomalies')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Humidity
            axes[1].plot(df.index, df['Humidity_percent'], color='blue', linewidth=2, label='Humidity')
            if len(anomaly_rows) > 0:
                axes[1].scatter(anomaly_rows.index, anomaly_rows['Humidity_percent'],
                              color='red', s=100, marker='X', label='Anomalies', zorder=5)
            axes[1].set_ylabel('Humidity (%)')
            axes[1].set_title('Humidity with Detected Anomalies')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Pressure
            axes[2].plot(df.index, df['Pressure_hPa'], color='green', linewidth=2, label='Pressure')
            if len(anomaly_rows) > 0:
                axes[2].scatter(anomaly_rows.index, anomaly_rows['Pressure_hPa'],
                              color='red', s=100, marker='X', label='Anomalies', zorder=5)
            axes[2].set_ylabel('Pressure (hPa)')
            axes[2].set_xlabel('Record Index')
            axes[2].set_title('Pressure with Detected Anomalies')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Summary metrics
            st.subheader("📊 Sensor Health Report")
            
            anomaly_count = df['Anomaly'].sum()
            uptime = (1 - anomaly_count/len(df)) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 Anomalies", int(anomaly_count))
            with col2:
                st.metric("✅ Uptime", f"{uptime:.1f}%")
            with col3:
                st.metric("⚠️ Anomaly Rate", f"{(anomaly_count/len(df))*100:.2f}%")
            
            # Detailed anomalies
            st.subheader("🔍 Detailed Anomaly List")
            if len(anomaly_rows) > 0:
                display_df = df[df['Anomaly']][['Temperature_C', 'Humidity_percent', 'Pressure_hPa', 'Anomaly_Score']].copy()
                display_df = display_df.rename(columns={'Anomaly_Score': 'Suspicion Score'})
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("✅ No anomalies detected! Sensor is healthy.")
            
        else:
            st.error("❌ CSV must have: Temperature_C, Humidity_percent, Pressure_hPa")
    
    else:
        st.info("👆 Upload a CSV file to start")

elif page == "About":
    st.markdown("""
    ## About SkyGuard AI
    
    ### Purpose
    Automatically detect sensor malfunctions in Automatic Weather Stations (AWS).
    
    ### Technology
    - **Algorithm**: Isolation Forest (Unsupervised ML)
    - **Framework**: Streamlit
    - **Libraries**: Scikit-Learn, Pandas, Matplotlib
    
    ### What It Detects
    - 🔴 **Spikes**: Impossible readings (e.g., 60°C)
    - 🔵 **Frozen Values**: Sensor stops responding
    - ⚪ **Missing Data**: Communication failures
    
    ### How to Use
    1. Prepare CSV with 3 columns:
       - `Temperature_C`
       - `Humidity_percent`
       - `Pressure_hPa`
    2. Upload file
    3. View results instantly
    
    ### Sample CSV Format
    Temperature_C,Humidity_percent,Pressure_hPa
    15.2,65.3,1013.2
    16.1,64.8,1013.1
    17.3,63.2,1013.0
        
    ### Reference
    **Smart India Hackathon 2026** - Problem Statement SIH26073
    """)