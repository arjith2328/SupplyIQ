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
import json

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
    
    csv_path = "outputs/forecast_results.csv"
    json_path = "outputs/forecast_metrics.json"
    
    if os.path.exists(csv_path) and os.path.exists(json_path):
        df_forecast = pd.read_csv(csv_path)
        with open(json_path, 'r') as f:
            metrics = json.load(f)
            
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", metrics.get("RMSE", "N/A"))
        c2.metric("MAE", metrics.get("MAE", "N/A"))
        c3.metric("MAPE", f'{metrics.get("MAPE", "N/A")}%')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_forecast['date'], y=df_forecast['actual'], name='Actual', mode='lines+markers', marker_color='cyan'))
        fig.add_trace(go.Scatter(x=df_forecast['date'], y=df_forecast['predicted'], name='Predicted', mode='lines+markers', marker_color='orange'))
        fig.update_layout(title="Demand Forecasting: Actual vs Predicted", template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("Please run this command in terminal: `python models/forecasting.py`")

# --- Anomaly Detection ---
elif page == "Anomaly Detection":
    st.title("Anomaly Detection")
    
    if_path = "outputs/anomaly_isolation.csv"
    ae_path = "outputs/anomaly_autoencoder.csv"
    
    if os.path.exists(if_path) and os.path.exists(ae_path):
        df_if = pd.read_csv(if_path)
        df_ae = pd.read_csv(ae_path)
        
        c1, c2 = st.columns(2)
        c1.metric("Anomalies (Isolation Forest)", df_if['is_anomaly'].sum())
        c2.metric("Anomalies (Autoencoder)", df_ae['is_anomaly'].sum())
        
        st.markdown("### Isolation Forest Results")
        fig1 = go.Figure()
        normal_if = df_if[df_if['is_anomaly'] == 0]
        anomalies_if = df_if[df_if['is_anomaly'] == 1]
        fig1.add_trace(go.Scatter(x=normal_if['date'], y=normal_if['value'], name='Normal', mode='markers+lines', line=dict(color='blue', width=1), marker=dict(color='blue', size=4)))
        fig1.add_trace(go.Scatter(x=anomalies_if['date'], y=anomalies_if['value'], name='Anomaly', mode='markers', marker=dict(color='red', size=10)))
        fig1.update_layout(template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("#### Top 10 Anomalous Values (Isolation Forest)")
        st.dataframe(anomalies_if.sort_values(by='value', ascending=False).head(10)[['date', 'value']])
        
        st.markdown("### Autoencoder Results")
        fig2 = go.Figure()
        normal_ae = df_ae[df_ae['is_anomaly'] == 0]
        anomalies_ae = df_ae[df_ae['is_anomaly'] == 1]
        fig2.add_trace(go.Scatter(x=normal_ae['date'], y=normal_ae['value'], name='Normal', mode='markers+lines', line=dict(color='blue', width=1), marker=dict(color='blue', size=4)))
        fig2.add_trace(go.Scatter(x=anomalies_ae['date'], y=anomalies_ae['value'], name='Anomaly', mode='markers', marker=dict(color='red', size=10)))
        fig2.update_layout(template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("#### Top 10 Anomalous Values (Autoencoder)")
        st.dataframe(anomalies_ae.sort_values(by='value', ascending=False).head(10)[['date', 'value']])
            
    else:
        st.warning("Please run this command in terminal: `python models/anomaly.py`")

# --- SQL Insights ---
elif page == "SQL Insights":
    st.title("SQL Insights")
    
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
