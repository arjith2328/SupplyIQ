"""
Module 3: Demand Forecasting
Trains and compares XGBoost, Prophet, and LSTM for daily order volume forecasting.
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pickle
import traceback

# Import specifically for LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def load_data(db_path="outputs/supplyiq.db"):
    """Load daily order volume from database."""
    print("Loading data from database...")
    conn = sqlite3.connect(db_path)
    query = """
    SELECT date(order_purchase_timestamp) as ds, COUNT(order_id) as y
    FROM olist_orders_dataset
    WHERE order_status != 'canceled'
    GROUP BY ds
    ORDER BY ds
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['ds'] = pd.to_datetime(df['ds'])
    # Sort chronologically
    df = df.sort_values('ds').reset_index(drop=True)
    return df

def feature_engineering(df):
    """Adds lag and rolling features for XGBoost."""
    print("Performing feature engineering...")
    df_feat = df.copy()
    df_feat['dayofweek'] = df_feat['ds'].dt.dayofweek
    df_feat['month'] = df_feat['ds'].dt.month
    df_feat['day'] = df_feat['ds'].dt.day
    df_feat['is_weekend'] = df_feat['dayofweek'].isin([5,6]).astype(int)
    
    # Lag features
    for i in [1, 7, 14]:
        df_feat[f'lag_{i}'] = df_feat['y'].shift(i)
        
    # Rolling averages
    df_feat['rolling_mean_7'] = df_feat['y'].rolling(window=7).mean()
    df_feat['rolling_mean_30'] = df_feat['y'].rolling(window=30).mean()
    
    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat

def calculate_metrics(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # Adding small epsilon to avoid div by zero
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    print(f"--- {model_name} Metrics ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")
    return rmse, mae, mape

def train_xgboost(df_feat):
    print("Training XGBoost...")
    features = [c for c in df_feat.columns if c not in ['ds', 'y']]
    
    # Simple temporal split (last 30 days as test)
    train_size = len(df_feat) - 30
    train, test = df_feat.iloc[:train_size], df_feat.iloc[train_size:]
    
    X_train, y_train = train[features], train['y']
    X_test, y_test = test[features], test['y']
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse, mae, mape = calculate_metrics(y_test, preds, "XGBoost")
    
    return model, mape, (test['ds'], y_test, preds)

def train_prophet(df):
    print("Training Prophet...")
    train_size = len(df) - 30
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(train)
    
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    preds = forecast['yhat'].iloc[-30:].values
    y_test = test['y'].values
    rmse, mae, mape = calculate_metrics(y_test, preds, "Prophet")
    
    return model, mape, (test['ds'], y_test, preds)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def train_lstm(df):
    print("Training LSTM...")
    try:
        data = df['y'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        seq_length = 30
        X, y = create_sequences(scaled_data, seq_length)
        
        train_size = len(X) - 30
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        # Train with limited epochs for speed in demonstration
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        
        scaled_preds = model.predict(X_test, verbose=0)
        preds = scaler.inverse_transform(scaled_preds).flatten()
        
        real_y = scaler.inverse_transform(y_test).flatten()
        rmse, mae, mape = calculate_metrics(real_y, preds, "LSTM")
        
        test_ds = df['ds'].iloc[-len(preds):]
        return model, mape, (test_ds, real_y, preds)
    except Exception as e:
        print(f"Error training LSTM: {e}")
        return None, float('inf'), None

def main():
    try:
        df = load_data()
        if df.empty:
            print("No data loaded. Ensure the database matches the expected schema.")
            return

        df_feat = feature_engineering(df)
        
        xgb_model, xgb_mape, xgb_plot_data = train_xgboost(df_feat)
        prophet_model, prophet_mape, prophet_plot_data = train_prophet(df)
        lstm_model, lstm_mape, lstm_plot_data = train_lstm(df)
        
        # Compare and save best model
        models = {'XGBoost': xgb_mape, 'Prophet': prophet_mape, 'LSTM': lstm_mape}
        best_name = min(models, key=models.get)
        print(f"\\nBest Model: {best_name} with MAPE: {models[best_name]:.2f}%")
        
        os.makedirs('outputs', exist_ok=True)
        # We will save XGBoost/Prophet as pickle, LSTM as h5
        if best_name == 'XGBoost':
            best_model = xgb_model
            with open('outputs/best_forecast_model.pkl', 'wb') as f:
                pickle.dump(best_model, f)
            plot_data = xgb_plot_data
        elif best_name == 'Prophet':
            best_model = prophet_model
            with open('outputs/best_forecast_model.pkl', 'wb') as f:
                pickle.dump(best_model, f)
            plot_data = prophet_plot_data
        else:
            best_model = lstm_model
            best_model.save('outputs/best_forecast_model.h5')
            plot_data = lstm_plot_data
            
        print("Saved best model to outputs/")

        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        ds, y_true, y_pred = plot_data
        plt.plot(ds, y_true, label='Actual', marker='o')
        plt.plot(ds, y_pred, label=f'Predicted ({best_name})', marker='x')
        plt.title(f'Demand Forecasting: Actual vs Predicted (Last 30 Days) - {best_name}')
        plt.xlabel('Date')
        plt.ylabel('Order Volume')
        plt.legend()
        plt.tight_layout()
        plt.savefig('outputs/forecast_comparison.png')
        print("Saved forecast plot to outputs/forecast_comparison.png")
        
    except Exception as e:
        print(f"Error in forecasting pipeline: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
