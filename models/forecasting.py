import sqlite3
import pandas as pd
import numpy as np
import os
import json
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import traceback

def load_data(db_path="outputs/supplyiq.db"):
    print("Step 1/5: Loading data from database...")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Please run setup_db.py first.")
        
    conn = sqlite3.connect(db_path)
    query = """
    SELECT date(order_purchase_timestamp) as date, COUNT(order_id) as actual
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

def feature_engineering(df):
    print("Step 2/5: Engineering lag and rolling features...")
    df_feat = df.copy()
    df_feat['dayofweek'] = df_feat['date'].dt.dayofweek
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['day'] = df_feat['date'].dt.day
    
    for i in [1, 7, 14]:
        df_feat[f'lag_{i}'] = df_feat['actual'].shift(i)
        
    df_feat['rolling_mean_7'] = df_feat['actual'].rolling(window=7).mean()
    df_feat['rolling_mean_30'] = df_feat['actual'].rolling(window=30).mean()
    
    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat

def train_xgboost(df_feat):
    print("Step 3/5: Training XGBoost Model...")
    features = [c for c in df_feat.columns if c not in ['date', 'actual']]
    
    train_size = len(df_feat) - 30
    train, test = df_feat.iloc[:train_size], df_feat.iloc[train_size:]
    
    X_train, y_train = train[features], train['actual']
    X_test, y_test = test[features], test['actual']
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-10))) * 100
    
    metrics = {
        "RMSE": float(round(rmse, 2)),
        "MAE": float(round(mae, 2)),
        "MAPE": float(round(mape, 2))
    }
    
    results_df = pd.DataFrame({
        'date': test['date'],
        'actual': y_test,
        'predicted': preds
    })
    
    return results_df, metrics

def main():
    try:
        os.makedirs('outputs', exist_ok=True)
        df = load_data()
        df_feat = feature_engineering(df)
        
        results_df, metrics = train_xgboost(df_feat)
        
        print("Step 4/5: Saving prediction results to CSV...")
        results_df.to_csv("outputs/forecast_results.csv", index=False)
        
        print("Step 5/5: Saving metrics to JSON...")
        with open("outputs/forecast_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        print(f"Success! Metrics: {metrics}")
        print("Forecasting model execution complete.")
        
    except Exception as e:
        print(f"Error in forecasting pipeline: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
