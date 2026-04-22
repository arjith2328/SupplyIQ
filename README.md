# SupplyIQ — Demand Forecasting & Supply Chain Optimization

## Project Overview
SupplyIQ is an end-to-end supply chain analytics and forecasting system designed to provide deep insights into e-commerce operations. Using the Olist Brazilian E-Commerce dataset, it features a complete pipeline from a SQL-based Data Warehouse to Exploratory Data Analysis, Machine Learning Demand Forecasting, Anomaly Detection, and an interactive Streamlit Dashboard.

## Features
- **Data Warehouse:** SQLite DB with normalized tables for orders, products, sellers, and customers. Complex SQL queries to extract KPIs.
- **Exploratory Data Analysis (EDA):** Insightful visualizations uncovering sales trends, top product categories, and delivery bottlenecks.
- **Demand Forecasting:** Time series forecasting comparing XGBoost, Facebook Prophet, and LSTM Deep Learning models.
- **Anomaly Detection:** Outlier detection using Isolation Forest and Autoencoders to find unusual demand spikes and delivery anomalies.
- **Interactive Dashboard:** Streamlit web application providing a clean, dynamic KPI overview and model insights.

## Project Structure
- `data/` : Raw CSV files (must be downloaded from Kaggle Olist Dataset).
- `sql/` : Scripts to initialize SQLite database and execute analytical queries.
- `notebooks/` : Jupyter Notebooks for Exploratory Data Analysis.
- `models/` : Python scripts for ML Forecasting and Anomaly Detection models.
- `dashboard/` : Streamlit app for visualizing KPIs and Model results.
- `outputs/` : Saved visualizations, databases, and trained model files.

## Setup Instructions
1. Download the [Olist E-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle and place the CSV files into the `data/` folder. At a minimum, ensure you have:
   - `olist_orders_dataset.csv`
   - `olist_order_items_dataset.csv`
   - `olist_products_dataset.csv`
   - `olist_customers_dataset.csv`
   - `olist_sellers_dataset.csv`
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Initialize the database and run the data warehouse script:
   ```bash
   python sql/setup_db.py
   ```
4. Run the Streamlit Dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```
