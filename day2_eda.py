import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("📊 DAY 2: Exploratory Data Analysis & Statistical Detection")
print("="*60)

# Load data
df = pd.read_csv('data_with_anomalies.csv')

# Clean missing data
df['Pressure_hPa'] = df['Pressure_hPa'].interpolate(method='linear')

print("\n✅ Data loaded and cleaned")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

# =============================================
# VISUALIZATION
# =============================================
print("\n📈 Creating visualizations...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Temperature
axes[0].plot(df.index, df['Temperature_C'], color='red', linewidth=2)
axes[0].set_ylabel('Temperature (°C)')
axes[0].set_title('Temperature Over Time')
axes[0].grid(True, alpha=0.3)

# Humidity
axes[1].plot(df.index, df['Humidity_percent'], color='blue', linewidth=2)
axes[1].set_ylabel('Humidity (%)')
axes[1].set_title('Humidity Over Time')
axes[1].grid(True, alpha=0.3)

# Pressure
axes[2].plot(df.index, df['Pressure_hPa'], color='green', linewidth=2)
axes[2].set_ylabel('Pressure (hPa)')
axes[2].set_xlabel('Hour')
axes[2].set_title('Pressure Over Time')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_weather_data_visualization.png', dpi=100)
print("💾 Saved: 01_weather_data_visualization.png")
plt.show()

# =============================================
# ROLLING Z-SCORE DETECTION
# =============================================
print("\n🔍 Training Z-Score Anomaly Detector...")

window = 24
df['Temp_rolling_mean'] = df['Temperature_C'].rolling(window=window, center=True).mean()
df['Temp_rolling_std'] = df['Temperature_C'].rolling(window=window, center=True).std()
df['Temp_rolling_zscore'] = np.abs(
    (df['Temperature_C'] - df['Temp_rolling_mean']) / df['Temp_rolling_std']
)
df['Temp_anomaly'] = df['Temp_rolling_zscore'] > 3

anomaly_count = df['Temp_anomaly'].sum()
print(f"✅ Found {anomaly_count} anomalies in temperature")

# Visualize
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(df.index, df['Temperature_C'], label='Temperature', color='red', linewidth=2, alpha=0.7)
ax.plot(df.index, df['Temp_rolling_mean'], label='24-Hour Average', color='blue', 
       linewidth=2, linestyle='--')
ax.fill_between(df.index, 
                df['Temp_rolling_mean'] - 3*df['Temp_rolling_std'],
                df['Temp_rolling_mean'] + 3*df['Temp_rolling_std'],
                alpha=0.2, color='gray', label='Normal Range (±3σ)')

anomaly_idx = df[df['Temp_anomaly']].index
ax.scatter(anomaly_idx, df.loc[anomaly_idx, 'Temperature_C'],
          color='red', s=150, marker='X', label='Detected Anomalies', zorder=5)

ax.set_xlabel('Hour')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Rolling Z-Score Anomaly Detection')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02_zscore_detection.png', dpi=100)
print("💾 Saved: 02_zscore_detection.png")
plt.show()

print("\n" + "="*60)
print("✅ DAY 2 COMPLETE!")
print("="*60)
print("\nYou learned:")
print("  ✓ How to visualize time series data")
print("  ✓ What Z-scores measure")
print("  ✓ How rolling windows adapt to time-series patterns")