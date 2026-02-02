
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns # Removed to avoid dependency issue
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.saving import register_keras_serializable
import joblib
import os

# Define custom objects for loading models
@register_keras_serializable()
class MCDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

print("="*60)
print("🚀 AQI MODEL COMPARISON: LSTM vs GRU vs ADVANCED GRU")
print("="*60)

# ===================== STEP 1: LOAD DATA & PREPARE =====================
print("\n📥 Step 1: Loading & Preparing Data...")
# Re-using logic from train_gru.py to ensure identical test set
# Re-using logic from train_gru.py to ensure identical test set
try:
    data = pd.read_csv("data/aqi_india_1990_2015_cleaned.csv", low_memory=False)
except:
    print("❌ New dataset not found, using old one")
    data = pd.read_csv("aqi_data_preprocessed.csv")

# Filter for Bangalore and Sort by Date
if 'location' in data.columns:
    print("✅ Filtering for City: Bangalore")
    data = data[data['location'].str.contains('Bangalore|Bengaluru', case=False, na=False)]
elif 'City' in data.columns:
    print("✅ Filtering for City: Bengaluru")
    data = data[data['City'] == 'Bengaluru']

if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    print("✅ Sorted by Date")
elif 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    print("✅ Sorted by Date")
aqi_column = None
for col in data.columns:
    if 'aqi' in col.lower():
        aqi_column = col
        break

if not aqi_column: aqi_column = data.columns[1]

# Clean
data = data.dropna(subset=[aqi_column])
# Imputation for misses
data[aqi_column] = data[aqi_column].fillna(method='ffill')
data = data[data[aqi_column] > 0]
data = data[data[aqi_column] <= 500]

# Add features
n_samples = len(data)
time_idx = np.arange(n_samples)
# Synthetic weather features (SMART CORRELATION)
# Normalized AQI for correlation (0 to 1)
norm_aqi = (data[aqi_column] - data[aqi_column].min()) / (data[aqi_column].max() - data[aqi_column].min())

# Temperature: Periodic + correlation (Cooler often = higher AQI due to inversion)
temp_base = 25 + 5 * np.sin(time_idx * (2*np.pi/24)) 
data['Temperature'] = temp_base - (norm_aqi * 5) + np.random.normal(0, 1, n_samples)

# Wind Speed: Strong inverse correlation (Wind disperses pollutants)
wind_base = np.abs(np.random.normal(10, 5, n_samples))
data['WindSpeed'] = np.clip(wind_base * (1 - norm_aqi * 0.8), 2, 30)

# Humidity: Positive correlation (High humidity/fog traps pollutants)
hum_base = 60 - 20 * np.sin(time_idx * (2*np.pi/24))
data['Humidity'] = np.clip(hum_base + (norm_aqi * 20) + np.random.normal(0, 3, n_samples), 20, 100)

data[aqi_column] = data[aqi_column].interpolate(method='linear')
data = data.fillna(method='bfill').fillna(method='ffill')

data = data.reset_index(drop=True)
data['TimeIndex'] = data.index
data['Hour'] = data['TimeIndex'] % 24
data['DayOfWeek'] = (data['TimeIndex'] // 24) % 7
data['Month'] = (data['TimeIndex'] // (24 * 30)) % 12

data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)

data['AQI_rolling_mean_3h'] = data[aqi_column].rolling(window=3, min_periods=1).mean()
data['AQI_rolling_mean_6h'] = data[aqi_column].rolling(window=6, min_periods=1).mean()
data['AQI_rolling_mean_12h'] = data[aqi_column].rolling(window=12, min_periods=1).mean() # NEW FEATURE
data['AQI_rolling_mean_24h'] = data[aqi_column].rolling(window=24, min_periods=1).mean()

data = data.ffill().bfill()

# STL placeholders (training scripts add these)
data['aqi_trend'] = data[aqi_column]
data['aqi_seasonal'] = 0
data['aqi_residual'] = 0

feature_columns = [
    aqi_column, 
    'so2', 'no2',  # Full pollutant features
    'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
    'AQI_rolling_mean_3h', 'AQI_rolling_mean_6h', 'AQI_rolling_mean_12h', 'AQI_rolling_mean_24h',
    'Temperature', 'Humidity', 'WindSpeed',
    'aqi_trend', 'aqi_seasonal', 'aqi_residual'  # STL components
]

# Filter to only existing columns
feature_columns = [c for c in feature_columns if c in data.columns]

feature_data = data[feature_columns].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(feature_data)

LOOKBACK = 72  # Extended to match training
FORECAST_HORIZON = 24

def create_sequences(data, lookback, forecast_horizon):
    X, y = [], []
    for i in range(lookback, len(data) - forecast_horizon):
        X.append(data[i-lookback:i])
        y.append(data[i:i+forecast_horizon, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, LOOKBACK, FORECAST_HORIZON)
split_index = int(len(X) * 0.8)
X_test = X[split_index:]
# y_test = y[split_index:] # Not strictly needed if we simulate prediction fully
y_test_actual_scaled = y[split_index:]

# Inverse transform actuals
dummy_test = np.zeros((y_test_actual_scaled.shape[0], y_test_actual_scaled.shape[1], X.shape[2]))
dummy_test[:, :, 0] = y_test_actual_scaled
y_test_actual = scaler.inverse_transform(dummy_test.reshape(-1, X.shape[2]))[:, 0].reshape(y_test_actual_scaled.shape)

print(f"✅ Test Set Size: {len(X_test)} samples")

# ===================== STEP 2: LOAD MODELS =====================
print("\n🤖 Step 2: Loading Models...")

dataset_results = {}

models_to_test = {
    "LSTM": "models/aqi_lstm_best.keras",
    "GRU": "models/aqi_gru_model.keras",
    "Advanced GRU": "models/aqi_attention_gru_best.keras"  # Attention-based GRU
}

results = {"Model": [], "MAE": [], "RMSE": []}
preds = {}

for name, path in models_to_test.items():
    if not os.path.exists(path):
        print(f"❌ {name} model not found at {path}")
        continue
        
    try:
        print(f"   Loading {name}...")
        model = load_model(path, custom_objects={"MCDropout": MCDropout})
        
        print(f"   Predicting with {name}...")
        y_pred = model.predict(X_test, verbose=0)
        
        # Inverse transform
        dummy_pred = np.zeros((y_pred.shape[0], y_pred.shape[1], X.shape[2]))
        dummy_pred[:, :, 0] = y_pred
        y_pred_actual = scaler.inverse_transform(dummy_pred.reshape(-1, X.shape[2]))[:, 0].reshape(y_pred.shape)
        
        # Metrics
        mae = np.mean(np.abs(y_test_actual - y_pred_actual))
        rmse = np.sqrt(np.mean((y_test_actual - y_pred_actual)**2))
        
        results["Model"].append(name)
        results["MAE"].append(mae)
        results["RMSE"].append(rmse)
        preds[name] = y_pred_actual
        
        print(f"   Shape: {y_pred_actual.shape}")
        
    except Exception as e:
        print(f"❌ Error evaluating {name}: {e}")

# ===================== STEP 3: VISUALIZATION =====================
print("\n📊 Step 3: Generating Comparison Graphs...")

if len(results["Model"]) > 0:
    results_df = pd.DataFrame(results)
    print("\n🏆 Results Summary:")
    print(results_df)

    plt.figure(figsize=(14, 10))
    
    # 1. MAE Comparison
    plt.subplot(2, 2, 1)
    bars = plt.bar(results_df["Model"], results_df["MAE"], color=['skyblue', 'lightgreen'])
    plt.title("Mean Absolute Error (Lower is Better)")
    plt.ylabel("MAE (AQI Points)")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    # 2. RMSE Comparison
    plt.subplot(2, 2, 2)
    bars2 = plt.bar(results_df["Model"], results_df["RMSE"], color=['salmon', 'orange'])
    plt.title("Root Mean Square Error (Lower is Better)")
    plt.ylabel("RMSE")
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    # 3. Prediction Overlay (Sample)
    plt.subplot(2, 1, 2)
    sample_idx = 0 # First sample in test set
    
    plt.plot(y_test_actual[sample_idx], 'k-', label="Actual", linewidth=2.5)
    
    colors = {"LSTM": "blue", "GRU": "red"}
    for name, pred_vals in preds.items():
        plt.plot(pred_vals[sample_idx], '--', label=f"{name} Prediction", color=colors.get(name, 'green'), linewidth=2)
        
    plt.title(f"24-Hour Forecast Comparison (Sample #{sample_idx})")
    plt.xlabel("Hours Ahead")
    plt.ylabel("AQI")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    print("\n✅ Comparison graph saved as: model_comparison.png")
    
else:
    print("❌ No models evaluated.")

