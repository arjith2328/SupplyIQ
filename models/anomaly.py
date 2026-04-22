import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.keras.layers as layers

import sqlite3
import pandas as pd
import numpy as np
import traceback
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

print("Starting Anomaly Detection...")

def load_data(db_path="outputs/supplyiq.db"):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Please run setup_db.py first.")
        
    conn = sqlite3.connect(db_path)
    query = """
    SELECT date(order_purchase_timestamp) as date, COUNT(*) as order_count
    FROM olist_orders_dataset 
    WHERE order_status != 'canceled'
    GROUP BY date 
    ORDER BY date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def run_isolation_forest(df):
    model = IsolationForest(contamination=0.05, random_state=42)
    X = df['order_count'].values.reshape(-1, 1)
    
    preds = model.fit_predict(X)
    
    df_if = pd.DataFrame()
    df_if['date'] = df['date']
    df_if['value'] = df['order_count']
    df_if['is_anomaly'] = [1 if x == -1 else 0 for x in preds]
    
    return df_if

def main():
    try:
        os.makedirs('outputs', exist_ok=True)
        
        # Load Data
        df = load_data()
        print(f"Data loaded. Shape: {df.shape}")
        print("First 5 rows:")
        print(df.head())
        
        # Isolation Forest
        df_if = run_isolation_forest(df)
        print(f"Isolation Forest found {df_if['is_anomaly'].sum()} anomalies")
        df_if.to_csv("outputs/anomaly_isolation.csv", index=False)
        
        # Autoencoder
        try:
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_ae = scaler.fit_transform(df[['order_count']])
            
            model = tf.keras.Sequential([
                layers.Dense(16, activation='relu', input_dim=1),
                layers.Dense(8, activation='relu'),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_ae, X_ae, epochs=30, batch_size=8, verbose=0, validation_split=0.1)
            
            reconstructions = model.predict(X_ae, verbose=0)
            mse = np.mean(np.square(X_ae - reconstructions), axis=1)
            
            threshold = np.percentile(mse, 95)
            
            df_ae = pd.DataFrame()
            df_ae['date'] = df['date']
            df_ae['value'] = df['order_count']
            df_ae['is_anomaly'] = (mse > threshold).astype(int)
            
            print(f"Autoencoder found {df_ae['is_anomaly'].sum()} anomalies")
            df_ae.to_csv("outputs/anomaly_autoencoder.csv", index=False)
            
        except Exception as e:
            traceback.print_exc()
             
        print("Anomaly Detection Complete! Files saved to outputs/")
        
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    main()
