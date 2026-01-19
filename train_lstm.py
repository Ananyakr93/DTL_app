import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 IMPROVED AQI LSTM MODEL TRAINING - 24 HOUR FORECAST")
print("="*60)

# ===================== STEP 1: LOAD DATA =====================
print("\n📥 Step 1: Loading Data...")

data = pd.read_csv("aqi_data.csv")
print(f"✅ Loaded {len(data)} records from aqi_data.csv")
print(f"📊 Data shape: {data.shape}")
print(f"📋 Columns: {list(data.columns)}")
print(f"\nFirst few rows:")
print(data.head())

# ===================== STEP 2: IDENTIFY AQI COLUMN =====================
print("\n🔍 Step 2: Identifying AQI Column...")

# Find the AQI column (case-insensitive)
aqi_column = None
for col in data.columns:
    if 'aqi' in col.lower():
        aqi_column = col
        break

if aqi_column is None:
    print("⚠️ No 'AQI' column found. Checking available columns...")
    print(f"Available columns: {list(data.columns)}")
    # If only 2 columns and no 'AQI', assume second column is AQI
    if len(data.columns) == 2:
        aqi_column = data.columns[1]
        print(f"✅ Assuming column '{aqi_column}' contains AQI values")
    else:
        raise ValueError("Please specify which column contains AQI data")
else:
    print(f"✅ Found AQI column: '{aqi_column}'")

print(f"📈 AQI range: {data[aqi_column].min():.1f} to {data[aqi_column].max():.1f}")
print(f"📊 AQI mean: {data[aqi_column].mean():.1f}")
print(f"📊 AQI median: {data[aqi_column].median():.1f}")

# ===================== STEP 3: DATA CLEANING =====================
print("\n🧹 Step 3: Cleaning Data...")

# Remove any NaN or invalid values
original_len = len(data)
data = data.dropna(subset=[aqi_column])

# ===================== STEP 3.5: AUGMENT WITH WEATHER FEATURES =====================
print("\n🌤️ Step 3.5: Augmenting with Synthetic Weather Features...")
print("(In production, use fetch_weather_features() from app.py to get real history)")

# Generate synthetic weather data correlated with AQI for training
# We assume this dataset has no timestamps, so we create a dummy time index
n_samples = len(data)
time_idx = np.arange(n_samples)

# Temperature: Periodic (daily cycle) + noise
# Cooler temps often correlate with higher AQI (inversion)
temp = 25 + 5 * np.sin(time_idx * (2*np.pi/24)) + np.random.normal(0, 2, n_samples)

# Humidity: Periodic + noise
humidity = 60 - 20 * np.sin(time_idx * (2*np.pi/24)) + np.random.normal(0, 5, n_samples)
humidity = np.clip(humidity, 20, 100)

# Wind Speed: Random walk
# Lower wind speed = higher AQI (stagnation)
wind = 10 + np.random.normal(0, 3, n_samples)
wind = np.clip(wind, 0, 30)

data['Temperature'] = temp
data['Humidity'] = humidity
data['WindSpeed'] = wind

print("✅ Added 'Temperature', 'Humidity', 'WindSpeed' features")
data = data[data[aqi_column] > 0]  # Remove zero/negative AQI
data = data[data[aqi_column] < 1000]  # Remove unrealistic values

# Imputation for missing values
data[aqi_column] = data[aqi_column].interpolate(method='linear')
data = data.fillna(method='bfill').fillna(method='ffill')

print(f"✅ Removed {original_len - len(data)} invalid records")
print(f"✅ Remaining records: {len(data)}")

# ===================== STEP 4: FEATURE ENGINEERING =====================
print("\n🔧 Step 4: Feature Engineering...")

# Create time-based features (since we don't have dates, create synthetic ones)
# Assuming data is sequential and hourly
data = data.reset_index(drop=True)
data['TimeIndex'] = data.index

# Create hour of day (assuming hourly data cycling through 24 hours)
data['Hour'] = data['TimeIndex'] % 24

# Create day of week (assuming data cycles through 7 days)
data['DayOfWeek'] = (data['TimeIndex'] // 24) % 7

# Create month (assuming data cycles through 12 months)
data['Month'] = (data['TimeIndex'] // (24 * 30)) % 12

# Cyclical encoding for better neural network understanding
data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)
data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

# Rolling statistics to capture trends
data['AQI_rolling_mean_3h'] = data[aqi_column].rolling(window=3, min_periods=1).mean()
data['AQI_rolling_mean_6h'] = data[aqi_column].rolling(window=6, min_periods=1).mean()
data['AQI_rolling_std_6h'] = data[aqi_column].rolling(window=6, min_periods=1).std()
data['AQI_rolling_mean_24h'] = data[aqi_column].rolling(window=24, min_periods=1).mean()
data['AQI_rolling_max_24h'] = data[aqi_column].rolling(window=24, min_periods=1).max()
data['AQI_rolling_min_24h'] = data[aqi_column].rolling(window=24, min_periods=1).min()

# Fill any NaN values from rolling operations
data = data.fillna(method='bfill').fillna(method='ffill')

print(f"✅ Created {len(data.columns)} features")
print(f"📋 Features: {list(data.columns)}")

# ===================== STEP 5: PREPARE SEQUENCES =====================
print("\n📦 Step 5: Creating Training Sequences...")

# Select features for model
feature_columns = [
    aqi_column,  # Main target
    'Hour_sin', 'Hour_cos',  # Time of day
    'DayOfWeek_sin', 'DayOfWeek_cos',  # Day of week
    'AQI_rolling_mean_3h', # Immediate trend
    'AQI_rolling_mean_6h',  # Short-term trend
    'AQI_rolling_mean_24h',  # Daily trend
    'Temperature', 'Humidity', 'WindSpeed' # Weather features
]

print(f"📋 Using features: {feature_columns}")

# Extract feature data
feature_data = data[feature_columns].values

# Scale the data (very important for neural networks!)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(feature_data)

# Save scaler
joblib.dump(scaler, "scaler_improved.pkl")
print("✅ Scaler saved as: scaler_improved.pkl")

# Create sequences
# Create sequences
def create_sequences(data, lookback=48, forecast_horizon=24):
    """
    Creates input-output sequences for training
    lookback: how many past hours to look at (increased to 48h)
    forecast_horizon: how many future hours to predict (now 24!)
    """
    X, y = [], []
    
    for i in range(lookback, len(data) - forecast_horizon):
        # Input: past 'lookback' hours with all features
        X.append(data[i-lookback:i])
        
        # Output: next 'forecast_horizon' hours (only AQI, column 0)
        y.append(data[i:i+forecast_horizon, 0])
    
    return np.array(X), np.array(y)

# Configuration - UPDATED FOR 24 HOUR FORECAST
LOOKBACK = 48  # Use past 48 hours for better context
FORECAST_HORIZON = 24  # Predict next 24 hours (CHANGED FROM 12!)

print(f"\n⚙️ Configuration:")
print(f"   Lookback window: {LOOKBACK} hours")
print(f"   Forecast horizon: {FORECAST_HORIZON} hours (EXTENDED!)")

X, y = create_sequences(scaled_data, LOOKBACK, FORECAST_HORIZON)

print(f"\n✅ Created sequences:")
print(f"   Input shape (X): {X.shape}")  # (samples, 24 hours, features)
print(f"   Output shape (y): {y.shape}")  # (samples, 24 predictions)
print(f"   Number of features: {X.shape[2]}")

# ===================== STEP 6: TRAIN-TEST SPLIT =====================
print("\n✂️ Step 6: Splitting Data...")

# Use 80% for training, 20% for validation
# DON'T shuffle - time series data must maintain order!
split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Validation samples: {len(X_test)}")
print(f"✅ Train/Test split: {len(X_train)/(len(X_train)+len(X_test))*100:.1f}% / {len(X_test)/(len(X_train)+len(X_test))*100:.1f}%")

# ===================== STEP 7: BUILD MODEL WITH MC DROPOUT =====================
# ===================== STEP 7: BUILD MODEL (LSTM or TRANSFORMER) =====================
print("\n🏗️ Step 7: Building Model architecture...")

# Custom Dropout layer that stays active during inference for MC Dropout
class MCDropout(Dropout):
    """Monte Carlo Dropout - stays active during inference for uncertainty estimation"""
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)  # Always in training mode

def build_transformer_model(input_shape, forecast_horizon):
    """Builds a Transformer-based model for time series forecasting"""
    from tensorflow.keras.layers import Input, GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention, Add
    from tensorflow.keras.models import Model
    
    inputs = Input(shape=input_shape)
    
    # Positional encoding projection
    x = Dense(256)(inputs)
    
    # Transformer Encoder Block 1
    # Multi-head attention
    attention_output = MultiHeadAttention(num_heads=4, key_dim=256)(x, x)
    # Add & Norm
    x2 = Add()([x, attention_output])
    x2 = LayerNormalization(epsilon=1e-6)(x2)
    
    # Feed Forward Network
    ffn = Dense(512, activation="relu")(x2)
    ffn = MCDropout(0.2)(ffn)
    ffn = Dense(256)(ffn)
    
    # Add & Norm
    x3 = Add()([x2, ffn])
    x3 = LayerNormalization(epsilon=1e-6)(x3)
    
    # Transformer Encoder Block 2
    attention_output = MultiHeadAttention(num_heads=4, key_dim=256)(x3, x3)
    x4 = Add()([x3, attention_output])
    x4 = LayerNormalization(epsilon=1e-6)(x4)
    
    ffn = Dense(512, activation="relu")(x4)
    ffn = MCDropout(0.2)(ffn)
    ffn = Dense(256)(ffn)
    
    x5 = Add()([x4, ffn])
    x5 = LayerNormalization(epsilon=1e-6)(x5)
    
    # Global average pooling to flatten time dimension
    x_gap = GlobalAveragePooling1D()(x5)
    x_gap = MCDropout(0.2)(x_gap)
    
    # Output projection
    outputs = Dense(forecast_horizon)(x_gap)
    
    model = Model(inputs=inputs, outputs=outputs, name="AQI_Transformer")
    return model

def build_lstm_model(input_shape, forecast_horizon):
    """Builds the improved LSTM model"""
    model = Sequential([
        Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.01)), input_shape=input_shape),
        MCDropout(0.3),
        BatchNormalization(),
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))),
        MCDropout(0.3),
        BatchNormalization(),
        LSTM(32, kernel_regularizer=l2(0.01)),
        MCDropout(0.2),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        MCDropout(0.2),
        Dense(32, activation='relu'),
        Dense(forecast_horizon)
    ], name="AQI_LSTM")
    return model

# CHOOSE MODEL TYPE HERE
MODEL_TYPE = "transformer"  # Options: "lstm", "transformer"

print(f"Selecting model architecture: {MODEL_TYPE.upper()}")

if MODEL_TYPE == "transformer":
    model = build_transformer_model((LOOKBACK, X.shape[2]), FORECAST_HORIZON)
else:
    model = build_lstm_model((LOOKBACK, X.shape[2]), FORECAST_HORIZON)

# Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer, 
    loss='huber',  # Better for outliers than MSE
    metrics=['mae', 'mse']
)

print("\n📐 Model Architecture:")
model.summary()

# ===================== STEP 8: TRAIN MODEL =====================
print("\n🎯 Step 8: Training Model...")
print("⏳ This may take 10-30 minutes depending on your hardware...")

# Callbacks for smart training
callbacks = [
    # Stop if no improvement for 15 epochs
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate if stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    ),
    
    # Save best model
    ModelCheckpoint(
        'aqi_lstm_best.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Train!
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,  # Will stop early if not improving
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training completed!")

# ===================== STEP 9: EVALUATE =====================
print("\n📊 Step 9: Evaluating Model Performance...")

# Make predictions on test set
y_pred = model.predict(X_test)

# Inverse transform to get actual AQI values
# Need to create dummy array for inverse transform
dummy_test = np.zeros((y_test.shape[0], y_test.shape[1], X.shape[2]))
dummy_test[:, :, 0] = y_test
y_test_actual = scaler.inverse_transform(dummy_test.reshape(-1, X.shape[2]))[:, 0].reshape(y_test.shape)

dummy_pred = np.zeros((y_pred.shape[0], y_pred.shape[1], X.shape[2]))
dummy_pred[:, :, 0] = y_pred
y_pred_actual = scaler.inverse_transform(dummy_pred.reshape(-1, X.shape[2]))[:, 0].reshape(y_pred.shape)

# Calculate metrics
mae = np.mean(np.abs(y_test_actual - y_pred_actual))
rmse = np.sqrt(np.mean((y_test_actual - y_pred_actual)**2))
mape = np.mean(np.abs((y_test_actual - y_pred_actual) / (y_test_actual + 1e-10))) * 100

print(f"\n{'='*50}")
print(f"📈 MODEL PERFORMANCE METRICS (24-HOUR FORECAST)")
print(f"{'='*50}")
print(f"MAE (Mean Absolute Error):  {mae:.2f} AQI points")
print(f"RMSE (Root Mean Squared):   {rmse:.2f} AQI points")
print(f"MAPE (Mean Abs % Error):    {mape:.2f}%")
print(f"{'='*50}")

# Performance interpretation
if mae < 20:
    print("🎉 EXCELLENT! Model is very accurate for 24-hour forecast!")
elif mae < 35:
    print("✅ GOOD! Model performance is acceptable for extended forecast.")
elif mae < 50:
    print("⚠️  FAIR. Consider collecting more data or tuning hyperparameters.")
else:
    print("❌ POOR. Model needs improvement. Try more data or different architecture.")

# ===================== STEP 10: UNCERTAINTY ESTIMATION =====================
print("\n🔮 Step 10: Estimating Uncertainty with Monte Carlo Dropout...")

# Perform multiple forward passes for uncertainty estimation
n_mc_samples = 20
mc_predictions = []

print(f"   Running {n_mc_samples} Monte Carlo samples...")
for _ in range(n_mc_samples):
    mc_pred = model.predict(X_test, verbose=0)
    mc_predictions.append(mc_pred)

mc_predictions = np.array(mc_predictions)
mc_mean = np.mean(mc_predictions, axis=0)
mc_std = np.std(mc_predictions, axis=0)

avg_uncertainty = np.mean(mc_std)
print(f"   Average prediction uncertainty: ±{avg_uncertainty:.3f} (scaled)")

# ===================== STEP 11: SAVE MODEL =====================
print("\n💾 Step 11: Saving Model...")

model.save("aqi_lstm_model_improved.keras")
print("✅ Model saved as: aqi_lstm_model_improved.keras")

# Save configuration
config = {
    'lookback': LOOKBACK,
    'forecast_horizon': FORECAST_HORIZON,
    'features': feature_columns,
    'n_features': X.shape[2],
    'aqi_column': aqi_column,
    'mae': float(mae),
    'rmse': float(rmse),
    'mape': float(mape),
    'avg_uncertainty': float(avg_uncertainty),
    'mc_samples': n_mc_samples
}
joblib.dump(config, "model_config.pkl")
print("✅ Config saved as: model_config.pkl")

# ===================== STEP 12: VISUALIZATIONS =====================
print("\n📊 Step 12: Creating Visualizations...")

# Create figure with multiple subplots
fig = plt.figure(figsize=(18, 12))

# Plot 1: Training Loss
ax1 = plt.subplot(2, 3, 1)
ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax1.set_title('Model Loss During Training', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: MAE
ax2 = plt.subplot(2, 3, 2)
ax2.plot(history.history['mae'], label='Training MAE', linewidth=2)
ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
ax2.set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Sample 24-Hour Prediction
ax3 = plt.subplot(2, 3, 3)
sample_idx = 0
ax3.plot(range(FORECAST_HORIZON), y_test_actual[sample_idx], 'o-', label='Actual', 
         linewidth=2, markersize=6)
ax3.plot(range(FORECAST_HORIZON), y_pred_actual[sample_idx], 's-', label='Predicted', 
         linewidth=2, markersize=6)
ax3.fill_between(range(FORECAST_HORIZON), 
                  y_pred_actual[sample_idx] - mc_std[sample_idx] * 50,
                  y_pred_actual[sample_idx] + mc_std[sample_idx] * 50,
                  alpha=0.3, label='Uncertainty')
ax3.set_title('Sample 24-Hour Forecast with Uncertainty', fontsize=12, fontweight='bold')
ax3.set_xlabel('Hour Ahead')
ax3.set_ylabel('AQI')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Multiple Predictions Comparison
ax4 = plt.subplot(2, 3, 4)
for i in range(min(10, len(y_test_actual))):
    ax4.plot(y_test_actual[i], alpha=0.3, color='blue', linewidth=1)
    ax4.plot(y_pred_actual[i], alpha=0.3, color='red', linewidth=1)
ax4.plot([], [], color='blue', label='Actual', linewidth=2)
ax4.plot([], [], color='red', label='Predicted', linewidth=2)
ax4.set_title('Predictions Overlay (10 samples)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Hour Ahead')
ax4.set_ylabel('AQI')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Error Distribution
ax5 = plt.subplot(2, 3, 5)
errors = (y_test_actual - y_pred_actual).flatten()
ax5.hist(errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax5.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
ax5.set_xlabel('Error (Actual - Predicted)')
ax5.set_ylabel('Frequency')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Scatter Plot - Predicted vs Actual
ax6 = plt.subplot(2, 3, 6)
ax6.scatter(y_test_actual.flatten(), y_pred_actual.flatten(), 
           alpha=0.5, s=10, color='steelblue')
# Perfect prediction line
min_val = min(y_test_actual.min(), y_pred_actual.min())
max_val = max(y_test_actual.max(), y_pred_actual.max())
ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
        label='Perfect Prediction')
ax6.set_title('Scatter Plot: Predicted vs Actual AQI', fontsize=12, fontweight='bold')
ax6.set_xlabel('Actual AQI')
ax6.set_ylabel('Predicted AQI')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
print("✅ Saved: training_results.png")

plt.show()

# ===================== STEP 13: TEST PREDICTION =====================
print("\n🧪 Step 13: Testing Prediction Function...")

def predict_future(current_sequence):
    """Test prediction with a sample sequence"""
    if len(current_sequence.shape) == 2:
        current_sequence = current_sequence.reshape(1, LOOKBACK, X.shape[2])
    
    prediction = model.predict(current_sequence, verbose=0)
    
    # Inverse transform
    dummy = np.zeros((1, FORECAST_HORIZON, X.shape[2]))
    dummy[0, :, 0] = prediction[0]
    prediction_actual = scaler.inverse_transform(dummy.reshape(-1, X.shape[2]))[:, 0]
    
    return prediction_actual

# Test with random sample
test_idx = np.random.randint(0, len(X_test))
test_sequence = X_test[test_idx:test_idx+1]
predicted = predict_future(test_sequence)
actual = y_test_actual[test_idx]

print(f"\n🎯 Sample Prediction Test (Index: {test_idx}):")
print(f"{'Hour':<10} {'Predicted':<15} {'Actual':<15} {'Error':<10}")
print("-" * 50)
for i in range(FORECAST_HORIZON):
    error = abs(predicted[i] - actual[i])
    print(f"+{i+1}h{'':<6} {predicted[i]:<15.1f} {actual[i]:<15.1f} {error:<10.1f}")

# ===================== FINAL SUMMARY =====================
print("\n" + "="*60)
print("✅ TRAINING COMPLETE - 24 HOUR FORECAST MODEL!")
print("="*60)
print(f"""
📁 Generated Files:
   ✅ aqi_lstm_model_improved.keras  - Main trained model
   ✅ aqi_lstm_best.keras            - Best checkpoint
   ✅ scaler_improved.pkl            - Data scaler
   ✅ model_config.pkl               - Model configuration
   ✅ training_results.png           - Training visualizations

📊 Final Performance (24-Hour Forecast):
   • MAE:  {mae:.2f} AQI points
   • RMSE: {rmse:.2f} AQI points  
   • MAPE: {mape:.2f}%
   • Avg Uncertainty: ±{avg_uncertainty:.3f}

🎯 Key Improvements:
   • Extended forecast from 12 → 24 hours
   • Monte Carlo Dropout for uncertainty estimation
   • Better data imputation

💡 Model Insights:
   • Trained on {len(data)} AQI records
   • Uses {LOOKBACK}-hour lookback window
   • Predicts {FORECAST_HORIZON} hours ahead
   • {X.shape[2]} features per timestep
   • {len(X_train)} training samples
   • {len(X_test)} validation samples

🚀 Ready to deploy with app.py!
""")
print("="*60)