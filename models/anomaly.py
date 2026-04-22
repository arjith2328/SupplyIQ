"""
Module 4: Anomaly Detection
Uses Isolation Forest and Autoencoder to flag anomalous daily order volumes.
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import traceback

def load_data(db_path="outputs/supplyiq.db"):
    """Load daily order volume from database."""
    print("Loading data for anomaly detection...")
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
    df = df.sort_values('ds').reset_index(drop=True)
    return df

def train_isolation_forest(df):
    print("Training Isolation Forest...")
    model = IsolationForest(contamination=0.05, random_state=42)
    # Using 'y' as the single feature for anomaly detection
    df_if = df.copy()
    X = df_if[['y']].values
    
    df_if['anomaly_if'] = model.fit_predict(X)
    # IF returns -1 for anomaly, 1 for normal. Convert to 1 (anomaly) and 0 (normal)
    df_if['anomaly_if'] = df_if['anomaly_if'].apply(lambda x: 1 if x == -1 else 0)
    return df_if

def train_autoencoder(df):
    print("Training Autoencoder...")
    try:
        df_ae = df.copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_ae[['y']])
        
        # Build simple Autoencoder
        input_dim = X_scaled.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(8, activation='relu')(input_layer)
        encoded = Dense(4, activation='relu')(encoded)
        decoded = Dense(8, activation='relu')(encoded)
        output_layer = Dense(input_dim, activation='linear')(decoded)
        
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        
        autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=0, validation_split=0.1)
        
        # Predict and calculate reconstruction error
        reconstructions = autoencoder.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
        
        # Threshold (e.g., 95th percentile)
        threshold = np.percentile(mse, 95)
        df_ae['anomaly_autoencoder'] = (mse > threshold).astype(int)
        
        return df_ae
    except Exception as e:
        print(f"Error training Autoencoder: {e}")
        df_ae = df.copy()
        df_ae['anomaly_autoencoder'] = 0
        return df_ae

def plot_anomalies(df, col_name, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], df['y'], color='blue', label='Order Volume', alpha=0.6)
    
    anomalies = df[df[col_name] == 1]
    plt.scatter(anomalies['ds'], anomalies['y'], color='red', label='Anomaly', zorder=5)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Order Volume')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved anomaly plot to {filename}")

def main():
    try:
        df = load_data()
        if df.empty:
            print("No data loaded. Ensure the database matches the expected schema.")
            return
            
        df_if = train_isolation_forest(df)
        df_ae = train_autoencoder(df)
        
        # Merge results
        df['anomaly_isolation_forest'] = df_if['anomaly_if']
        df['anomaly_autoencoder'] = df_ae['anomaly_autoencoder']
        
        os.makedirs('outputs', exist_ok=True)
        
        plot_anomalies(df, 'anomaly_isolation_forest', 'Anomalies Detected by Isolation Forest', 'outputs/anomalies_isolation_forest.png')
        plot_anomalies(df, 'anomaly_autoencoder', 'Anomalies Detected by Autoencoder', 'outputs/anomalies_autoencoder.png')
        
        # Save summary report
        report_path = 'outputs/anomaly_summary.csv'
        anomalies_combined = df[(df['anomaly_isolation_forest'] == 1) | (df['anomaly_autoencoder'] == 1)]
        anomalies_combined.to_csv(report_path, index=False)
        print(f"Saved anomaly summary report to {report_path} with {len(anomalies_combined)} anomalies flagged.")
        
    except Exception as e:
        print(f"Error in anomaly detection pipeline: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
