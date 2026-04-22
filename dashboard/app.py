"""
Module 5: Streamlit Dashboard
Multi-page application visualizing KPIs, forecasting, anomalies, and SQL insights.
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add parent directory to path so we can import sql queries
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from sql.queries import (
        get_top_product_categories, get_monthly_revenue_trend, 
        get_order_fulfillment_rate, get_avg_delivery_time_by_state,
        get_customer_retention, get_payment_method_distribution
    )
except ImportError:
    pass

st.set_page_config(page_title="SupplyIQ Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- Custom Styling for Dark Theme & Professional UI ---
st.markdown("""
<style>
    .reportview-container {
        background: #121212;
        color: #E0E0E0;
    }
    .sidebar .sidebar-content {
        background: #1E1E1E;
    }
    h1, h2, h3 {
        color: #4DB6AC !important;
    }
    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load raw data for aggregations
@st.cache_data
def load_base_data():
    db_path = "outputs/supplyiq.db"
    if not os.path.exists(db_path):
        return pd.DataFrame(), pd.DataFrame()
    conn = sqlite3.connect(db_path)
    
    # Load basic order data for filtering
    query_orders = "SELECT order_id, customer_id, order_status, date(order_purchase_timestamp) as order_date FROM olist_orders_dataset WHERE order_status != 'canceled'"
    df_o = pd.read_sql_query(query_orders, conn)
    
    query_items = "SELECT order_id, product_id, price, freight_value FROM olist_order_items_dataset"
    df_i = pd.read_sql_query(query_items, conn)
    conn.close()
    
    df_o['order_date'] = pd.to_datetime(df_o['order_date'])
    return df_o, df_i

df_orders, df_items = load_base_data()

# --- Sidebar ---
st.sidebar.title("SupplyIQ 📈")
st.sidebar.markdown("Demand Forecasting & Supply Chain Optimization")
page = st.sidebar.radio("Navigation", ["KPI Overview", "Demand Forecasting", "Anomaly Detection", "SQL Insights"])

if not df_orders.empty:
    min_date = df_orders['order_date'].min().date()
    max_date = df_orders['order_date'].max().date()
    st.sidebar.subheader("Global Filters")
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# --- KPI Overview ---
if page == "KPI Overview":
    st.title("KPI Overview")
    if df_orders.empty:
        st.warning("Database not found or empty. Please run sql/setup_db.py first after adding data to data/ folder.")
    else:
        # Filter data based on date range
        mask = (df_orders['order_date'].dt.date >= date_range[0]) & (df_orders['order_date'].dt.date <= date_range[1])
        filtered_orders = df_orders.loc[mask]
        merged_df = pd.merge(filtered_orders, df_items, on='order_id', how='inner')
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Orders", f"{len(filtered_orders):,}")
        with col2:
            st.metric("Total Revenue", f"${merged_df['price'].sum():,.2f}")
        with col3:
            st.metric("Average Order Value", f"${merged_df['price'].sum() / len(filtered_orders) if len(filtered_orders) > 0 else 0:,.2f}")
        with col4:
            st.metric("Total Freight Cost", f"${merged_df['freight_value'].sum():,.2f}")
        with col5:
            st.metric("Items Sold", f"{len(merged_df):,}")

        st.markdown("### Revenue Over Time")
        daily_rev = merged_df.groupby('order_date')['price'].sum().reset_index()
        fig = px.area(daily_rev, x='order_date', y='price', title='Daily Revenue', template="plotly_dark", color_discrete_sequence=['#4DB6AC'])
        st.plotly_chart(fig, use_container_width=True)

# --- Demand Forecasting ---
elif page == "Demand Forecasting":
    st.title("Demand Forecasting")
    st.markdown("Comparing XGBoost, Prophet, and LSTM models for the last 30 days of available data.")
    
    img_path = "outputs/forecast_comparison.png"
    if os.path.exists(img_path):
        st.image(img_path, caption="Actual vs Predicted Demand")
        
        st.info("The forecasting pipeline automatically selects the best model based on Mean Absolute Percentage Error (MAPE) and saves it to the `outputs` directory.")
        
        if os.path.exists("outputs/best_forecast_model.pkl"):
            st.success("Best Model: XGBoost or Prophet (.pkl detected)")
        elif os.path.exists("outputs/best_forecast_model.h5"):
            st.success("Best Model: LSTM (.h5 detected)")
            
    else:
        st.warning("Forecasting output not found. Please run models/forecasting.py.")

# --- Anomaly Detection ---
elif page == "Anomaly Detection":
    st.title("Anomaly Detection")
    st.markdown("Detecting unusual spikes or drops in demand using **Isolation Forest** and **Autoencoders**.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Isolation Forest Anomalies")
        if os.path.exists("outputs/anomalies_isolation_forest.png"):
            st.image("outputs/anomalies_isolation_forest.png")
        else:
            st.warning("Plot not found.")
            
    with col2:
        st.markdown("### Autoencoder Anomalies")
        if os.path.exists("outputs/anomalies_autoencoder.png"):
            st.image("outputs/anomalies_autoencoder.png")
        else:
            st.warning("Plot not found.")
            
    if os.path.exists("outputs/anomaly_summary.csv"):
        st.markdown("### Detected Anomalies Data")
        df_anomalies = pd.read_csv("outputs/anomaly_summary.csv")
        st.dataframe(df_anomalies, use_container_width=True)

# --- SQL Insights ---
elif page == "SQL Insights":
    st.title("SQL Insights")
    st.markdown("Complex query results powering our data warehouse.")
    
    if not os.path.exists("outputs/supplyiq.db"):
        st.warning("Database not found.")
    else:
        tab1, tab2, tab3 = st.tabs(["Top Categories", "Customer Retention", "Fulfillment Rate"])
        
        with tab1:
            try:
                df_cat = get_top_product_categories()
                fig = px.bar(df_cat, x='total_revenue', y='product_category_name', orientation='h', 
                             title="Top 10 Product Categories by Revenue", template="plotly_dark",
                             color='total_revenue', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_cat)
            except Exception as e:
                st.error("Error executing query.")

        with tab2:
            try:
                df_ret = get_customer_retention()
                fig = px.pie(df_ret, values='total_customers', names='customer_type', 
                             title="Customer Retention Mix", template="plotly_dark", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error("Error executing query.")
                
        with tab3:
            try:
                df_ful = get_order_fulfillment_rate()
                fig = px.line(df_ful, x='order_month', y='fulfillment_rate_pct', 
                              title="Monthly Fulfillment Rate (%)", template="plotly_dark", markers=True)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_ful)
            except Exception as e:
                st.error("Error executing query.")

st.sidebar.markdown("---")
st.sidebar.info("Developed as part of the WWT Data Science Internship Application Project: SupplyIQ.")
