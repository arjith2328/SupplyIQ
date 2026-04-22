"""
Module 1: Data Warehouse Setup
Loads Olist CSV dataset into a SQLite database using Pandas and SQLAlchemy.
"""

import pandas as pd
import os
import glob
from sqlalchemy import create_engine
import traceback

def setup_database(data_folder="data", db_path="outputs/supplyiq.db"):
    """
    Reads all CSV files from data_folder and loads them into a SQLite database.
    """
    print(f"Setting up database at {db_path}...")
    
    # Ensure outputs directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Create SQLite Engine
    try:
        engine = create_engine(f"sqlite:///{db_path}")
    except Exception as e:
        print(f"Error creating database engine: {e}")
        return False
        
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_folder}. Please download the Olist dataset.")
        return False
        
    for file_path in csv_files:
        # Extract table name from filename, e.g., olist_orders_dataset.csv -> olist_orders_dataset
        table_name = os.path.basename(file_path).replace(".csv", "")
        print(f"Processing {table_name}...")
        
        try:
            # Read CSV in chunks or entirely if small enough
            df = pd.read_csv(file_path)
            # Write to database (replace if exists initially)
            df.to_sql(table_name, con=engine, if_exists='replace', index=False)
            print(f"Successfully loaded {len(df)} records into '{table_name}'.")
        except Exception as e:
            print(f"Error loading {file_path} into database:")
            traceback.print_exc()

    print("Database setup complete.")
    return True

if __name__ == "__main__":
    setup_database()
