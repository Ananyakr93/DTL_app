import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import joblib
import warnings
warnings.filterwarnings('ignore')

# STL Decomposition for trend/seasonality
try:
    from statsmodels.tsa.seasonal import STL
    HAS_STL = True
except ImportError:
    print("⚠️ statsmodels not found. STL decomposition disabled.")
    HAS_STL = False

print("="*60)
print("🚀 GRU MODEL TRAINING - 24 HOUR FORECAST")
print("="*60)

# ===================== STEP 1: LOAD DATA =====================
print("\n📥 Step 1: Loading Data...")

try:
    data = pd.read_csv("aqi_india_1990_2015_cleaned.csv", low_memory=False)
    print(f"✅ Loaded {len(data)} records from aqi_india_1990_2015_cleaned.csv")
except FileNotFoundError:
    print("❌ Error: aqi_india_1990_2015_cleaned.csv not found. Please run preprocessing first.")
    exit()

# Filter for Bangalore (using most frequent implementation) and Sort by Date
if 'location' in data.columns:
    print("✅ Filtering for City: Bangalore")
    # Fuzzy match 'Bangalore' or 'Bengaluru'
    data = data[data['location'].str.contains('Bangalore|Bengaluru', case=False, na=False)]

if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    print("✅ Sorted by Date")

# ===================== STEP 2: IDENTIFY AQI COLUMN =====================
print("\n🔍 Step 2: Identifying AQI Column...")

# Find the AQI column (case-insensitive)
aqi_column = None
for col in data.columns:
    if 'aqi' in col.lower():
        aqi_column = col
        break

if aqi_column is None:
    if len(data.columns) >= 2:
        aqi_column = data.columns[1]
        print(f"✅ Assuming column '{aqi_column}' contains AQI values")
    else:
        raise ValueError("Please specify which column contains AQI data")
else:
    print(f"✅ Found AQI column: '{aqi_column}'")

# ===================== STEP 3: DATA CLEANING =====================
print("\n🧹 Step 3: Cleaning Data...")

data = data.dropna(subset=[aqi_column])
# Imputation for misses
data[aqi_column] = data[aqi_column].fillna(method='ffill')

data = data[data[aqi_column] > 0]
data = data[data[aqi_column] <= 500] # Cap at 500 to focus on standard range and reduce outlier skew

# Synthetic weather features (SMART CORRELATION)
n_samples = len(data)
time_idx = np.arange(n_samples)

# Normalized AQI for correlation (0 to 1)
norm_aqi = (data[aqi_column] - data[aqi_column].min()) / (data[aqi_column].max() - data[aqi_column].min())

# Temperature: Periodic + correlation (Cooler often = higher AQI due to inversion)
# Base seasonal/daily cycle
temp_base = 25 + 5 * np.sin(time_idx * (2*np.pi/24)) 
# Subtract normalized AQI (High AQI -> Lower Temp) scaled by 5 degrees
data['Temperature'] = temp_base - (norm_aqi * 5) + np.random.normal(0, 1, n_samples)

# Wind Speed: Strong inverse correlation (Wind disperses pollutants)
# Base random variation
wind_base = np.abs(np.random.normal(10, 5, n_samples))
# Invert based on AQI: High AQI -> Low Wind
data['WindSpeed'] = np.clip(wind_base * (1 - norm_aqi * 0.8), 2, 30)

# Humidity: Positive correlation (High humidity/fog traps pollutants)
hum_base = 60 - 20 * np.sin(time_idx * (2*np.pi/24))
data['Humidity'] = np.clip(hum_base + (norm_aqi * 20) + np.random.normal(0, 3, n_samples), 20, 100)

# Imputation
data[aqi_column] = data[aqi_column].interpolate(method='linear')
data = data.fillna(method='bfill').fillna(method='ffill')

# ===================== STEP 4: FEATURE ENGINEERING =====================
print("\n🔧 Step 4: Feature Engineering...")

data = data.reset_index(drop=True)
data['TimeIndex'] = data.index
data['Hour'] = data['TimeIndex'] % 24
data['DayOfWeek'] = (data['TimeIndex'] // 24) % 7
data['Month'] = (data['TimeIndex'] // (24 * 30)) % 12

data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)

# Rolling stats
data['AQI_rolling_mean_3h'] = data[aqi_column].rolling(window=3, min_periods=1).mean()
data['AQI_rolling_mean_6h'] = data[aqi_column].rolling(window=6, min_periods=1).mean()
data['AQI_rolling_mean_12h'] = data[aqi_column].rolling(window=12, min_periods=1).mean() # NEW FEATURE
data['AQI_rolling_mean_24h'] = data[aqi_column].rolling(window=24, min_periods=1).mean()

data = data.ffill().bfill()

# STL Decomposition (if available)
if HAS_STL and len(data) > 100:
    print("\n📈 Applying STL Decomposition...")
    try:
        stl = STL(data[aqi_column], period=7, robust=True)  # Weekly seasonality
        result = stl.fit()
        data['aqi_trend'] = result.trend
        data['aqi_seasonal'] = result.seasonal
        data['aqi_residual'] = result.resid
        # Fill any NaN from decomposition edges
        data['aqi_trend'] = data['aqi_trend'].ffill().bfill()
        data['aqi_seasonal'] = data['aqi_seasonal'].ffill().bfill()
        data['aqi_residual'] = data['aqi_residual'].ffill().bfill()
        print("✅ STL decomposition added (trend, seasonal, residual)")
    except Exception as e:
        print(f"⚠️ STL failed: {e}")
        data['aqi_trend'] = data[aqi_column]
        data['aqi_seasonal'] = 0
        data['aqi_residual'] = 0
else:
    data['aqi_trend'] = data[aqi_column]
    data['aqi_seasonal'] = 0
    data['aqi_residual'] = 0

# ===================== STEP 5: PREPARE SEQUENCES =====================
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

joblib.dump(scaler, "scaler_gru.pkl") 

def create_sequences(data, lookback=48, forecast_horizon=24):
    X, y = [], []
    for i in range(lookback, len(data) - forecast_horizon):
        X.append(data[i-lookback:i])
        y.append(data[i:i+forecast_horizon, 0])
    return np.array(X), np.array(y)

LOOKBACK = 72  # Extended from 48 to capture weekly patterns
FORECAST_HORIZON = 24
X, y = create_sequences(scaled_data, LOOKBACK, FORECAST_HORIZON)

split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ===================== STEP 7: BUILD GRU MODEL =====================
print("\n🏗️ Step 7: Building GRU Model...")

class MCDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

def build_gru_model(input_shape, forecast_horizon):
    model = Sequential([
        Bidirectional(GRU(256, return_sequences=True, kernel_regularizer=l2(0.01)), input_shape=input_shape),
        MCDropout(0.3),
        BatchNormalization(),
        Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=l2(0.01))),
        MCDropout(0.3),
        BatchNormalization(),
        GRU(32, kernel_regularizer=l2(0.01)),
        MCDropout(0.2),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        MCDropout(0.2),
        Dense(32, activation='relu'),
        Dense(forecast_horizon)
    ], name="AQI_GRU")
    return model

model = build_gru_model((LOOKBACK, X.shape[2]), FORECAST_HORIZON)
model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae', 'mse'])

# ===================== STEP 8: TRAIN MODEL =====================
print("\n🎯 Step 8: Training Model (Quick Run)...")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint('aqi_gru_best.keras', monitor='val_loss', save_best_only=True)
]

# Reduced epochs for quick switching, user can increase later
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50, 
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ===================== STEP 11: SAVE MODEL =====================
print("\n💾 Step 11: Saving GRU Model...")
model.save("aqi_gru_model.keras")

config = {
    'lookback': LOOKBACK,
    'forecast_horizon': FORECAST_HORIZON,
    'features': feature_columns,
    'n_features': X.shape[2],
    'mae': float(history.history['val_mae'][-1]),
    'rmse': float(np.sqrt(history.history['val_mse'][-1]))
}
joblib.dump(config, "model_config_gru.pkl")

print("✅ Model saved as: aqi_gru_model.keras")
print("✅ Config saved as: model_config_gru.pkl")

# ===================== STEP 12: SAVE PLOTS =====================
print("\n📊 Step 12: Saving Results... (Requested: PNG Format)")

# Plot Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('GRU Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('gru_training_loss.png')
print("✅ Training loss graph saved as: gru_training_loss.png")

# Plot Predictions vs Actual (Sample)
# Take last item in test set
sample_idx = -1
X_sample = X_test[sample_idx].reshape(1, LOOKBACK, X.shape[2])
y_true_scaled = y_test[sample_idx]
y_pred_scaled = model.predict(X_sample)[0]

# Inverse transform
# We need to construct a full feature array to use the scaler's inverse_transform
# Create dummy array with same shape as 'feature_data' (samples, features)
# We only care about the first column (AQI)

def inverse_transform_aqi(y_scaled):
    dummy = np.zeros((len(y_scaled), X.shape[2]))
    dummy[:, 0] = y_scaled
    return scaler.inverse_transform(dummy)[:, 0]

y_true = inverse_transform_aqi(y_true_scaled)
y_pred = inverse_transform_aqi(y_pred_scaled)

plt.figure(figsize=(10, 6))
plt.plot(y_true, label='Actual AQI', marker='o')
plt.plot(y_pred, label='Predicted AQI', marker='x')
plt.title(f'24-Hour Forecast vs Actual (Test Sample #{len(X_test)})')
plt.xlabel('Hours Ahead')
plt.ylabel('AQI')
plt.legend()
plt.savefig('gru_prediction_sample.png')
print("✅ Prediction sample graph saved as: gru_prediction_sample.png")
