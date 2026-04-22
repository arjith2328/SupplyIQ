#!/bin/bash
# SupplyIQ Setup Script for Streamlit Cloud Deployment

echo "Starting SupplyIQ environment setup..."

# Ensure outputs directory exists
mkdir -p outputs

# Step 1: Create Database
if [ ! -f "outputs/supplyiq.db" ]; then
    echo "Database not found. Running sql/setup_db.py..."
    python sql/setup_db.py
else
    echo "Database outputs/supplyiq.db already exists. Skipping."
fi

# Step 2: Run Forecasting Model
if [ ! -f "outputs/forecast_results.csv" ]; then
    echo "Forecast results not found. Running models/forecasting.py..."
    python models/forecasting.py
else
    echo "Forecast results outputs/forecast_results.csv already exists. Skipping."
fi

# Step 3: Run Anomaly Detection Model
if [ ! -f "outputs/anomaly_isolation.csv" ]; then
    echo "Anomaly isolation results not found. Running models/anomaly.py..."
    python models/anomaly.py
else
    echo "Anomaly isolation results outputs/anomaly_isolation.csv already exists. Skipping."
fi

echo "Setup complete! Ready to run the Streamlit app."
