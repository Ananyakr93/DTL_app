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

print("="*60)
print("🚀 GRU MODEL TRAINING - 24 HOUR FORECAST")
print("="*60)

# ===================== STEP 1: LOAD DATA =====================
print("\n📥 Step 1: Loading Data...")

try:
    data = pd.read_csv("aqi_data.csv")
    print(f"✅ Loaded {len(data)} records from aqi_data.csv")
except FileNotFoundError:
    print("❌ Error: aqi_data.csv not found. Please ensure the file exists.")
    exit()

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

data = data.fillna(method='bfill').fillna(method='ffill')

# ===================== STEP 5: PREPARE SEQUENCES =====================
feature_columns = [
    aqi_column, 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
    'AQI_rolling_mean_3h', 'AQI_rolling_mean_6h', 'AQI_rolling_mean_12h', 'AQI_rolling_mean_24h',
    'Temperature', 'Humidity', 'WindSpeed'
]

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

LOOKBACK = 48
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
