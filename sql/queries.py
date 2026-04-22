"""
Module 1: SQL Queries
Functions to perform data analysis using SQLite database.
"""

import pandas as pd
from sqlalchemy import create_engine
import traceback

def get_engine(db_path="outputs/supplyiq.db"):
    """Returns a SQLAlchemy engine connected to the SQLite database."""
    return create_engine(f"sqlite:///{db_path}")

def execute_query(query, db_path="outputs/supplyiq.db"):
    """Executes a SQL query and returns a pandas DataFrame."""
    try:
        engine = get_engine(db_path)
        return pd.read_sql_query(query, engine)
    except Exception as e:
        print(f"Error executing query: {e}")
        traceback.print_exc()
        return None

# 1. Which product categories have the highest sales?
def get_top_product_categories():
    query = """
    SELECT 
        p.product_category_name, 
        COUNT(oi.order_id) as total_units_sold,
        SUM(oi.price) as total_revenue
    FROM olist_order_items_dataset oi
    JOIN olist_products_dataset p ON oi.product_id = p.product_id
    GROUP BY p.product_category_name
    ORDER BY total_revenue DESC
    LIMIT 10;
    """
    return execute_query(query)

# 2. Which sellers have the most delays?
def get_most_delayed_sellers():
    query = """
    WITH DelayedOrders AS (
        SELECT 
            order_id, 
            order_estimated_delivery_date, 
            order_delivered_customer_date
        FROM olist_orders_dataset
        WHERE order_status = 'delivered' 
          AND order_delivered_customer_date > order_estimated_delivery_date
    )
    SELECT 
        oi.seller_id,
        COUNT(d.order_id) as delayed_orders_count
    FROM DelayedOrders d
    JOIN olist_order_items_dataset oi ON d.order_id = oi.order_id
    GROUP BY oi.seller_id
    ORDER BY delayed_orders_count DESC
    LIMIT 10;
    """
    return execute_query(query)

# 3. Monthly revenue trend per category (Overall revenue trend since join is heavy)
def get_monthly_revenue_trend():
    query = """
    SELECT 
        strftime('%Y-%m', o.order_purchase_timestamp) as order_month,
        SUM(oi.price) as monthly_revenue
    FROM olist_orders_dataset o
    JOIN olist_order_items_dataset oi ON o.order_id = oi.order_id
    WHERE o.order_status != 'canceled'
    GROUP BY order_month
    ORDER BY order_month;
    """
    return execute_query(query)

# 4. Top 10 customers by order value
def get_top_customers():
    query = """
    SELECT 
        c.customer_unique_id,
        COUNT(DISTINCT o.order_id) as total_orders,
        SUM(oi.price + oi.freight_value) as total_spent
    FROM olist_customers_dataset c
    JOIN olist_orders_dataset o ON c.customer_id = o.customer_id
    JOIN olist_order_items_dataset oi ON o.order_id = oi.order_id
    GROUP BY c.customer_unique_id
    ORDER BY total_spent DESC
    LIMIT 10;
    """
    return execute_query(query)

# 5. Average delivery time by destination state
def get_avg_delivery_time_by_state():
    query = """
    SELECT 
        c.customer_state,
        AVG(julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp)) as avg_delivery_days
    FROM olist_orders_dataset o
    JOIN olist_customers_dataset c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL
    GROUP BY c.customer_state
    ORDER BY avg_delivery_days DESC;
    """
    return execute_query(query)

# 6. Order fulfillment rate per month (Total Orders vs Delivered Orders)
def get_order_fulfillment_rate():
    query = """
    SELECT 
        strftime('%Y-%m', order_purchase_timestamp) as order_month,
        COUNT(order_id) as total_orders,
        SUM(CASE WHEN order_status = 'delivered' THEN 1 ELSE 0 END) as delivered_orders,
        (SUM(CASE WHEN order_status = 'delivered' THEN 1.0 ELSE 0.0 END) / COUNT(order_id)) * 100 as fulfillment_rate_pct
    FROM olist_orders_dataset
    GROUP BY order_month
    ORDER BY order_month;
    """
    return execute_query(query)

# 7. Distribution of payment methods
def get_payment_method_distribution():
    query = """
    SELECT 
        payment_type,
        COUNT(order_id) as transactions_count,
        SUM(payment_value) as total_value
    FROM olist_order_payments_dataset
    GROUP BY payment_type
    ORDER BY total_value DESC;
    """
    return execute_query(query)

# 8. Highest freight value by product category
def get_freight_value_by_category():
    query = """
    SELECT 
        p.product_category_name,
        AVG(oi.freight_value) as avg_freight_value,
        MAX(oi.freight_value) as max_freight_value
    FROM olist_order_items_dataset oi
    JOIN olist_products_dataset p ON oi.product_id = p.product_id
    GROUP BY p.product_category_name
    ORDER BY avg_freight_value DESC
    LIMIT 10;
    """
    return execute_query(query)

# 9. Customer retention (Repeat vs One-time customers)
def get_customer_retention():
    # Number of users who made 1 order vs >1 orders
    query = """
    WITH CustomerOrders AS (
        SELECT 
            c.customer_unique_id,
            COUNT(o.order_id) as order_count
        FROM olist_customers_dataset c
        JOIN olist_orders_dataset o ON c.customer_id = o.customer_id
        GROUP BY c.customer_unique_id
    )
    SELECT 
        CASE 
            WHEN order_count = 1 THEN 'One-Time Customer'
            ELSE 'Repeat Customer' 
        END as customer_type,
        COUNT(customer_unique_id) as total_customers
    FROM CustomerOrders
    GROUP BY customer_type;
    """
    return execute_query(query)

# 10. Top rated product categories (Average Review Score)
def get_top_rated_categories():
    query = """
    SELECT 
        p.product_category_name,
        AVG(r.review_score) as avg_review_score,
        COUNT(r.review_id) as total_reviews
    FROM olist_order_reviews_dataset r
    JOIN olist_order_items_dataset oi ON r.order_id = oi.order_id
    JOIN olist_products_dataset p ON oi.product_id = p.product_id
    GROUP BY p.product_category_name
    HAVING total_reviews > 50
    ORDER BY avg_review_score DESC, total_reviews DESC
    LIMIT 10;
    """
    return execute_query(query)

if __name__ == "__main__":
    print("Testing sample query...")
    df = get_top_product_categories()
    if df is not None:
        print(df.head())
