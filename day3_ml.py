import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

print("🤖 DAY 3: Machine Learning with Isolation Forest")
print("="*60)

# Load and clean data
df = pd.read_csv('data_with_anomalies.csv')
df['Pressure_hPa'] = df['Pressure_hPa'].interpolate(method='linear')

# Prepare features
features = ['Temperature_C', 'Humidity_percent', 'Pressure_hPa']
X = df[features].copy().fillna(df[features].mean())

print(f"✅ Features prepared: {X.shape[0]} samples, {X.shape[1]} features")

# =============================================
# TRAIN ISOLATION FOREST
# =============================================
print("\n🔨 Training Isolation Forest...")

iso_forest = IsolationForest(contamination=0.05, random_state=42)
predictions = iso_forest.fit_predict(X)
anomaly_scores = iso_forest.score_samples(X)

df['ML_prediction'] = predictions
df['ML_anomaly_score'] = anomaly_scores
df['ML_is_anomaly'] = predictions == -1

anomaly_count = (predictions == -1).sum()
print(f"✅ Found {anomaly_count} anomalies")
print(f"\nAnomalous rows:")
print(df[df['ML_is_anomaly']][['Temperature_C', 'Humidity_percent', 'Pressure_hPa', 'ML_anomaly_score']])

# =============================================
# VISUALIZATION
# =============================================
print("\n📊 Creating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Prediction (Red=Anomaly, Green=Normal)
scatter1 = axes[0].scatter(df['Temperature_C'], df['Humidity_percent'],
                          c=df['ML_prediction'], cmap='RdYlGn', s=100, alpha=0.6)
axes[0].set_xlabel('Temperature (°C)')
axes[0].set_ylabel('Humidity (%)')
axes[0].set_title('Isolation Forest Results\n(Red=Anomaly, Green=Normal)')
plt.colorbar(scatter1, ax=axes[0])

# Right: Anomaly Score (darker=more suspicious)
scatter2 = axes[1].scatter(df['Temperature_C'], df['Humidity_percent'],
                          c=df['ML_anomaly_score'], cmap='RdYlGn_r', s=100, alpha=0.6)
axes[1].set_xlabel('Temperature (°C)')
axes[1].set_ylabel('Humidity (%)')
axes[1].set_title('Anomaly Scores\n(Red=Highly Suspicious, Green=Normal)')
plt.colorbar(scatter2, ax=axes[1], label='Anomaly Score')

plt.tight_layout()
plt.savefig('03_ml_isolation_forest.png', dpi=100)
print("💾 Saved: 03_ml_isolation_forest.png")
plt.show()

print("\n" + "="*60)
print("✅ DAY 3 COMPLETE!")
print("="*60)
print("\nYou learned:")
print("  ✓ How Isolation Forest works")
print("  ✓ How to train an ML model")
print("  ✓ How ML catches multivariate anomalies")