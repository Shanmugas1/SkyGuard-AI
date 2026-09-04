import pandas as pd
import numpy as np

print("🌤️ Creating sample weather data...")

# Create sample weather data (7 days, hourly readings)
dates = pd.date_range(start='2024-01-01', periods=168, freq='H')
np.random.seed(42)

# Normal weather patterns
temp = 15 + 10*np.sin(np.arange(168)*2*np.pi/24) + np.random.normal(0, 1, 168)
humidity = 60 + 15*np.sin(np.arange(168)*2*np.pi/24 + 2) + np.random.normal(0, 2, 168)
pressure = 1013 + 5*np.sin(np.arange(168)*2*np.pi/24 + 4) + np.random.normal(0, 1, 168)

# Create DataFrame
df_clean = pd.DataFrame({
    'Timestamp': dates,
    'Temperature_C': temp,
    'Humidity_percent': humidity,
    'Pressure_hPa': pressure
})

print("✅ Created clean data")
print(df_clean.head(10))
print(f"\nDataset shape: {df_clean.shape}")

# Save clean data
df_clean.to_csv('clean_data.csv', index=False)
print("\n💾 Saved: clean_data.csv")

# ==================================================
# NOW INJECT ANOMALIES
# ==================================================

print("\n" + "="*50)
print("Injecting anomalies...")
print("="*50)

df_anomaly = df_clean.copy()

# ANOMALY 1: Spikes
print("\n🔴 Injecting SPIKES (impossible readings)...")
spike_indices = np.random.choice(df_anomaly.index, size=5, replace=False)
for idx in spike_indices:
    df_anomaly.loc[idx, 'Temperature_C'] = 55
    print(f"   Row {idx}: Temperature set to 55°C")

# ANOMALY 2: Frozen values
print("\n🔵 Injecting FROZEN VALUES (sensor stops)...")
frozen_start = 50
frozen_duration = 10
df_anomaly.loc[frozen_start:frozen_start+frozen_duration, 'Humidity_percent'] = 45
print(f"   Rows {frozen_start}-{frozen_start+frozen_duration}: Humidity frozen at 45%")

# ANOMALY 3: Missing data
print("\n⚪ Injecting MISSING DATA (communication drops)...")
missing_indices = np.random.choice(df_anomaly.index, size=8, replace=False)
df_anomaly.loc[missing_indices, 'Pressure_hPa'] = np.nan
print(f"   Injected {len(missing_indices)} missing pressure readings")

# Save anomalous data
df_anomaly.to_csv('data_with_anomalies.csv', index=False)
print("\n💾 Saved: data_with_anomalies.csv")

print("\n" + "="*50)
print("✅ DAY 1 COMPLETE!")
print("="*50)
print("\nYou now have:")
print("  ✓ clean_data.csv (normal weather data)")
print("  ✓ data_with_anomalies.csv (broken sensor data)")
print("\nReady for Day 2!")